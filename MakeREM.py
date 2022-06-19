#!/usr/bin/env python
import os
import numpy as np
import gdal
import osr
import ogr
import osmnx  # for querying OpenStreetMaps data to get river centerlines
from scipy.spatial import cKDTree as KDTree  # for finding nearest neighbors/interpolating
import seaborn as sn  # for nice color palettes
import subprocess
import time
from RasterViz import RasterViz


start = time.time()


class REMMaker(object):
    """
    An attempt to automatically make river REM from DEM

    TODO:
        - make into CLI util
        - test guess_k method, see if it works well on a variety of datasets
        - crop rivers shapefile to DEM extent
        - error handling if rivers but no named rivers?
        - handling geographic coord system?
    """
    def __init__(self, dem, eps=0.1, workers=4):
        """
        :param dem: str, path to DEM raster
        :param eps: float, fractional tolerance for errors in kd tree query
        :param workers: int, number of CPU threads to use for interpolation. -1 = all threads.
        """
        self.dem = dem
        self.dem_name = os.path.basename(dem).split('.')[0]
        self.proj, self.epsg_code = self.get_projection()
        # bbox (n_lat, s_lat, e_lon, w_lon) of DEM
        self.bbox = self.get_bbox()
        self.eps = eps
        self.workers = workers

    @property
    def dem(self):
        return self._dem

    @dem.setter
    def dem(self, dem):
        if not os.path.exists(dem):
            raise FileNotFoundError(f"Cannot find input DEM: {dem}")
        self._dem = dem
        return self._dem

    def get_projection(self):
        """Get EPSG code for DEM raster projection."""
        gtif = gdal.Open(self.dem, gdal.GA_ReadOnly)
        proj = osr.SpatialReference(wkt=gtif.GetProjection())
        epsg_code = proj.GetAttrValue('AUTHORITY', 1)
        return proj, epsg_code

    def get_bbox(self):
        """Get lat/long extent for DEM raster."""
        src = gdal.Open(self.dem, gdal.GA_ReadOnly)
        source_crs = src.GetSpatialRef()
        ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
        lrx = ulx + (src.RasterXSize * xres)
        lry = uly + (src.RasterYSize * yres)
        target_crs = osr.SpatialReference()
        target_crs.ImportFromEPSG(4326)  # WGS84 Geographic Coordinate System
        transform = osr.CoordinateTransformation(source_crs, target_crs)
        ul_lat, ul_long = transform.TransformPoint(ulx, uly)[:2]
        lr_lat, lr_long = transform.TransformPoint(lrx, lry)[:2]
        bbox = [ul_lat, lr_lat, lr_long, ul_long]
        return bbox

    def get_river_centerline(self):
        """Find centerline of river(s) within DEM area using OSM Ways"""
        print("\nFinding river centerline.")
        # get OSM Ways within bbox of DEM (returns geopandas geodataframe)
        self.rivers = osmnx.geometries_from_bbox(*self.bbox, tags={'waterway': ['river', 'stream', 'tidal channel']})
        if len(self.rivers) == 0:
            raise Exception("No rivers found within the DEM domain. Ensure the target river is on OpenStreetMaps.")
        # read into geodataframe with same CRS as DEM
        self.rivers = self.rivers.to_crs(epsg=self.epsg_code)
        # get river names (drop ones without a name)
        self.rivers = self.rivers.dropna(subset=['name'])
        names = self.rivers.name.values
        # make name attribute more distinct to avoid conflict with geometry name attribute
        self.rivers['river_name'] = names
        # get unique names
        river_names = set(names)
        # integer id corresponding to each river name
        self.river_ids = {name: i + 1 for i, name in enumerate(river_names)}
        self.rivers['river_id'] = [self.river_ids[name] for name in names]
        print(f"Found river(s): {', '.join(river_names)}")
        # find river with greatest length (sum of all segments with same name)
        """
        print("\nRiver lengths:")
        river_lengths = {}
        for river_name in river_names:
            river_segments = self.rivers[self.rivers.river_name == river_name]
            river_length = river_segments.length.sum()
            print(f"\t{river_name}: {river_length:.4f}")
            river_lengths[river_name] = river_length
        longest_river = max(river_lengths, key=river_lengths.get)
        print(f"\nLongest river: {longest_river}\n")
        # only use longest river to make REM
        self.rivers = self.rivers[self.rivers.river_name == longest_river]
        """
        self.make_river_shp()
        return

    def make_river_shp(self):
        """Make list of OSM Way object geometries into a shapefile"""
        self.river_shp = f'{self.dem_name}_rivers.shp'
        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(self.river_shp)
        # create empty multiline geometry layer
        layer = ds.CreateLayer('', self.proj, ogr.wkbMultiLineString)
        # Add fields
        layer.CreateField(ogr.FieldDefn('name', ogr.OFTString))
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()
        # populate layer with a feature for each natural waterway geometrye
        for i, way in self.rivers.iterrows():
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            # set feature attributes
            feat.SetField('name', way.river_name)
            feat.SetField('id', way.river_id)
            # set feature geometry from Shapely object
            geom = ogr.CreateGeometryFromWkb(way.geometry.wkb)
            feat.SetGeometry(geom)
            # add feature to layer
            layer.CreateFeature(feat)
            feat = geom = None  # destroy these
        # Save and close everything
        ds = layer = feat = geom = None

    def get_dem_coords(self):
        """Get x, y coordinates of DEM raster pixels as array"""
        print("Getting coordinates of DEM pixels.")
        r = gdal.Open(self.dem, gdal.GA_ReadOnly)
        band = r.GetRasterBand(1)
        self.dem_array = band.ReadAsArray()
        # ensure that nodata values become np.nans in array
        self.nodata_val = band.GetNoDataValue()
        if self.nodata_val:
            self.dem_array = np.where(self.dem_array == self.nodata_val, np.nan, self.dem_array)
        rows, cols = np.shape(self.dem_array)
        # get dimensions of raster
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r.GetGeoTransform()
        self.cell_w, self.cell_h = x_size, y_size
        min_x, max_x = sorted([upper_left_x, upper_left_x + x_size * cols])
        min_y, max_y = sorted([upper_left_y, upper_left_y + y_size * rows])
        self.extent = (min_x, min_y, max_x, max_y)
        # make arrays with x, y indices
        x_indices = np.array([range(cols)] * rows)
        y_indices = np.array([[i] * cols for i in range(rows)])
        # map indices to coords
        self.xs_array = x_indices * x_size + upper_left_x + (x_size / 2)
        self.ys_array = y_indices * y_size + upper_left_y + (y_size / 2)
        return

    def get_river_elev(self):
        """Get DEM values along river centerline"""
        print("Getting river elevation at DEM pixels.")
        # gdal_rasterize centerline
        centerline_ras = f"{self.dem_name}_centerline.tif"
        extent = f"-te {' '.join(map(str, self.extent))}"
        res = f"-tr {self.cell_w} {self.cell_h}"
        cmd = f"gdal_rasterize -a id {extent} {res} {self.river_shp} {centerline_ras}"
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        # raster to numpy array same shape as DEM
        r = gdal.Open(centerline_ras, gdal.GA_ReadOnly)
        self.centerline_array = r.GetRasterBand(1).ReadAsArray()
        # remove cells where DEM is null
        self.centerline_array = np.where(np.isnan(self.dem_array), np.nan, self.centerline_array)
        # identify river with most active pixels in DEM domain (use that one to make REM)
        pixel_counts = {}
        for river_name, id in self.river_ids.items():
            pixel_count = len(self.centerline_array[self.centerline_array == id])
            pixel_counts[id] = pixel_count
        longest_river_id = max(pixel_counts, key=pixel_counts.get)
        print(f"\nLongest river in domain: {[name for name, id in self.river_ids.items() if id == longest_river_id][0]}")
        # redefine array with 1 for longest river, 0 otherwise
        self.centerline_array = np.where(self.centerline_array == longest_river_id, 1, np.nan)
        # use np.where to get DEM elevation and coordinates at active pixels
        self.river_x_coords = self.xs_array[np.where(self.centerline_array == 1)]
        self.river_y_coords = self.ys_array[np.where(self.centerline_array == 1)]
        self.river_wses = self.dem_array[np.where(self.centerline_array == 1)]
        return

    def guess_k(self):
        """Determine the number of k nearest neighbors to use for interpolation"""
        area = np.where(np.isnan(self.dem_array), 0, 1).sum()
        river_pixels = len(self.centerline_array[self.centerline_array == 1])
        # a crude estimte of sinuosity, seems to be preserved across resolutions
        area_ratio = river_pixels / np.sqrt(area)
        # use a percentage of total river pixels times area ratio
        k = int(river_pixels / 1e3 * area_ratio * 2)
        print(f"guessing k = {k}")
        # make k be a minimum of 5, maximum of 100
        k = min(100, max(5, k))
        return k


    def interp_river_elev(self):
        """Interpolate elevation at river centerline across DEM extent."""
        print("\nInterpolating river elevation across DEM extent.")
        k = self.guess_k()
        print(f"Using k = {k} nearest neighbors.")
        # coords of sampled pixels along centerline
        c_sampled = np.column_stack((self.river_x_coords, self.river_y_coords))
        # coords to interpolate over (don't interpolated where DEM is null or on centerline (where REM elevation is 0))
        interp_indices = np.where(~(np.isnan(self.dem_array) | (self.centerline_array == 1)))
        c_interpolate = np.column_stack((self.xs_array[interp_indices], self.ys_array[interp_indices]))
        # create 2D tree
        print("Constructing tree")
        tree = KDTree(c_sampled)
        # find k nearest neighbors
        print("Querying tree")
        try:
            distances, indices = tree.query(c_interpolate, k=k, eps=self.eps, n_jobs=self.workers)
            # interpolate (IDW with power = 1)
            print("Making interpolated WSE array")
            weights = 1 / distances  # weight river elevations by 1 / distance
            weights = weights / weights.sum(axis=1).reshape(-1, 1)  # normalize weights
            interpolated_values = (weights * self.river_wses[indices]).sum(axis=1)  # apply weights
        except MemoryError:
            print("WARNING: Large dataset. Chunking query...")
            chunk_size = 1e6
            # iterate over chunks
            chunk_count = c_interpolate.shape[0] // chunk_size
            interpolated_values = np.array([])
            for i, chunk in enumerate(np.array_split(c_interpolate, chunk_count)):
                print(f"{i / chunk_count * 100:.2f}%")
                distances, indices = tree.query(chunk, k=k, eps=self.eps, n_jobs=self.workers)
                weights = 1 / distances
                weights = weights / weights.sum(axis=1).reshape(-1, 1)
                interpolated_values = np.append(interpolated_values, (weights * self.river_wses[indices]).sum(axis=1))

        # create interpolated WSE array as elevations along centerline, nans everywhere else
        self.wse_interp_array = np.where(self.centerline_array == 1, self.dem_array, np.nan)
        # add the interpolated eleation values
        self.wse_interp_array[interp_indices] = interpolated_values
        return

    def detrend_dem(self):
        """Subtract interpolated river elevation from DEM elevation to get REM"""
        print("\nDetrending DEM.")
        self.rem_array = self.dem_array - self.wse_interp_array

        self.rem_ras = f"{self.dem_name}_REM.tif"
        # make copy of DEM raster
        r = gdal.Open(self.dem, gdal.GA_ReadOnly)
        driver = gdal.GetDriverByName("GTiff")
        rem = driver.CreateCopy(self.rem_ras, r, strict=0)
        # fill with REM array
        rem.GetRasterBand(1).WriteArray(self.rem_array)
        return self.rem_ras

    def make_image_blend(self):
        """Blend REM with DEM hillshade to make pretty finished product"""
        print("\nBlending REM with hillshade.")
        # make hillshade of original DEM
        dem_viz = RasterViz(self.dem, out_ext=".tif")
        dem_viz.make_hillshade(multidirectional=True, z=2)
        # make hillshdae color using hillshade from DEM and color-relief from REM
        rem_viz = RasterViz(self.rem_ras, out_ext=".tif", make_png=True, make_kmz=True, docker_run=False)
        rem_viz.hillshade_ras = dem_viz.hillshade_ras  # use hillshade of original DEM
        rem_viz.viz_srs = rem_viz.proj  # make png visualization using source projection
        rem_viz.make_hillshade_color(cmap='mako_r', log_scale=True, blend_percent=45)
        return

    def run(self):
        """Make pretty REM/hillshade blend from DEM"""
        self.get_dem_coords()
        self.get_river_centerline()
        self.get_river_elev()
        self.interp_river_elev()
        self.detrend_dem()
        self.make_image_blend()
        return


if __name__ == "__main__":
    dem = "./test_dems/smith_pc.tif"
    rem_maker = REMMaker(dem=dem, eps=0.1, workers=4)
    rem_maker.run()
    #rem_maker.rem_ras = f"{rem_maker.dem_name}_REM.tif"
    #rem_maker.make_image_blend()

    end = time.time()
    print(f'\nDone.\nRan in {end - start:.0f} s.')
