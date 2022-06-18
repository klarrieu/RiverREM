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


class REMViz(RasterViz):
    """An extension of RasterViz to view REMs"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _png_kmz_checker(func):
        """Used as a wrapper for making viz products, making png and kmz after tif if selected"""
        def wrapper(self, *args, **kwargs):
            try:
                # call decorated method
                self.proj = self.viz_srs  # don't do transform
                ras_path = func(self, *args, **kwargs)
                print(f"Saved {ras_path}.")
                # make png and kmz if applicable
                if self.make_png:
                    png_path = self.raster_to_png(ras_path)
                    print(f"Saved {png_path}.")
                if self.make_kmz:
                    kmz_path = self.raster_to_kmz(ras_path)
                    print(f"Saved {kmz_path}.")
            except Exception as e:
                raise Exception(e)
            finally:
                # clean up regardless of whether something failed
                self.clean_up()
            return
        return wrapper

    @_png_kmz_checker
    def make_hillshade_color(self, *args, **kwargs):
        """Make a pretty composite hillshade/color-relief image"""
        # make hillshade and/or color-relief if haven't already
        # temp variables to override make_png, make_kmz
        _make_png, _make_kmz = self.make_png, self.make_kmz
        self.make_png = self.make_kmz = False
        if not os.path.exists(self.hillshade_ras):
            self.make_hillshade(*args, **kwargs)
        if not os.path.exists(self.color_relief_ras):
            self.make_color_relief(*args, **kwargs)
        self.make_png, self.make_kmz = _make_png, _make_kmz

        print("\nMaking hillshade-color composite raster.")
        # blend images using GDAL and numpy to linearly interpolate RGB values
        temp_path = self.blend_images(blend_percent=45)
        out_path = f"{self.dem_name}_hillshade-color{self.ext}"
        self.tile_and_compress(temp_path, out_path)
        # set nodata value to 0 for color-relief
        r = gdal.Open(out_path, gdal.GA_Update)
        [r.GetRasterBand(i+1).SetNoDataValue(0) for i in range(3)]
        return out_path

    def make_color_relief(self, cmap='mako_r', *args, **kwargs):
        """
        Make color relief map from DEM (4 band RGBA raster)
        :param cmap: str, matplotlib colormap to use for making color relief map.
                     (see https://matplotlib.org/stable/gallery/color/colormap_reference.html)
        """
        print(f"\nMaking color relief map with cmap={cmap}.")
        temp_path = self.intermediate_rasters["color-relief"]
        out_path = self.color_relief_ras
        cmap_txt = self.get_cmap_txt(cmap)
        cmd = f"{self.drun}gdaldem color-relief {self.dp}{self.dem} {self.dp}{cmap_txt} {self.dp}{temp_path} " \
              f"-of {self.out_format}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        self.tile_and_compress(temp_path, out_path)
        # set nodata value to 0 for color-relief
        r = gdal.Open(out_path, gdal.GA_Update)
        [r.GetRasterBand(i+1).SetNoDataValue(0) for i in range(3)]
        return out_path

    def get_cmap_txt(self, cmap='mako_r'):
        min_elev, max_elev = self.get_elev_range()
        # sample 256 evenly log-spaced elevation values
        elevations = np.logspace(0, np.log10(0.5 * max_elev), 256) - 1
        # matplotlib cmap function maps [0-255] input to RGB
        cm_mpl = sn.color_palette(cmap, as_cmap=True)
        # convert output RGB colors on [0-255] range to [1-255] range used by gdaldem (reserving 0 for nodata)
        cm = lambda x: [val * 254 + 1 for val in cm_mpl(x)[:3]]
        # make cmap text file to be read by gdaldem color-relief
        cmap_txt = f'{self.dem_name}_cmap.txt'
        with open(cmap_txt, 'w') as f:
            lines = [f'{elev} {" ".join(map(str, cm(i)))}\n' for i, elev in enumerate(elevations)]
            lines.append("nv 0 0 0\n")
            f.writelines(lines)
        return cmap_txt

    def blend_images(self, blend_percent=60):
        """
        Blend hillshade and color-relief rasters by linearly interpolating RGB values
        :param blend_percent: Percent weight of hillshdae in blend, color-relief takes opposite weight [0-100]. Default 60.
        """
        b = blend_percent / 100
        # read in hillshade and color relief rasters as arrays
        hs = gdal.Open(self.hillshade_ras, gdal.GA_ReadOnly)
        cr = gdal.Open(self.color_relief_ras, gdal.GA_ReadOnly)
        hs_array = hs.ReadAsArray()   # singleband (m, n)
        cr_arrays = cr.ReadAsArray()  # RGBA       (4, m, n)
        # make a copy of color-relief raster
        driver = gdal.GetDriverByName(self.out_format)
        blend_ras_name = self.intermediate_rasters["hillshade-color"]
        blend_ras = driver.CreateCopy(blend_ras_name, cr, strict=0)
        # linearly interpolate between hillshade and color-relief RGB values (keeping alpha channel from color-relief)
        for i in range(3):
            blended_band = np.where((hs_array != 0) & (cr_arrays[i] != 0), b * hs_array + (1 - b) * cr_arrays[i], 0)
            blend_ras.GetRasterBand(i+1).WriteArray(blended_band)
        return blend_ras_name


class REMMaker(object):
    """
    An attempt to automatically make river REM from DEM

    TODO:
        - make into CLI util
        - try to set k nearest neighbors based on total pixels / centerline pixels or something similar
        - or possibly downsample the centerline interpolation data to cap the number of nearest neighbors?
        - make interpolation more efficient for querying rasters with large nodata areas by only interpolating valid DEM pixels
        - remove centerline pixels from sampling if they are nans (or fill holes?)
        - improve longest river identification by cropping geometries to DEM domain?
            - but even cropping to extent would not necessarily crop to valid data pixels...
        - error handling if rivers but no named rivers?
        - handling geographic coord system
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
        # TODO: account for rotation?
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
        area = np.where(self.dem_array > 0, 1, 0).sum()
        river_pixels = len(self.centerline_array[self.centerline_array == 1])
        # a crude estimte of sinuosity, seems to be preserved across resolutions
        area_ratio = river_pixels / np.sqrt(area)
        # use a percentage of total river pixels times area ratio
        k = int(river_pixels / 1e3 * area_ratio / 2)
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
        c_sampled = np.array([self.river_x_coords, self.river_y_coords]).T
        # coords to interpolate over
        c_interpolate = np.dstack(np.array([self.xs_array, self.ys_array])).reshape(-1, 2)
        # create 2D tree
        print("Constructing tree")
        tree = KDTree(c_sampled)
        # find k nearest neighbors
        print("Querying tree")
        try:
            distances, indices = tree.query(c_interpolate, k=k, eps=self.eps, n_jobs=self.workers)
            # interpolate (IDW with power = 1)
            print("Making interpolated WSE array")
            weights = 1 / distances
            weights = weights / weights.sum(axis=1).reshape(-1, 1)
            interpolated_values = (weights * self.river_wses[indices]).sum(axis=1)
            self.wse_interp_array = interpolated_values.reshape(np.shape(self.dem_array))
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
            self.wse_interp_array = interpolated_values.reshape(np.shape(self.dem_array))

        # interpolation scheme causes divide by zero at sampled pixels, overwrite the nans to elevation values there
        river_elev_array = np.where(self.centerline_array == 1, self.dem_array, np.nan)
        self.wse_interp_array = np.where(np.isnan(self.wse_interp_array), river_elev_array, self.wse_interp_array)
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
        rem_viz = REMViz(self.rem_ras, out_ext=".tif", make_png=True, make_kmz=True, docker_run=False)
        rem_viz.hillshade_ras = dem_viz.hillshade_ras
        rem_viz.make_hillshade_color()
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
    dem = "./test_dems/susitna_large.tin.tif"
    rem_maker = REMMaker(dem=dem, eps=0.1, workers=4)
    rem_maker.run()
    #rem_maker.rem_ras = f"{rem_maker.dem_name}_REM.tif"
    #rem_maker.make_image_blend()

    end = time.time()
    print(f'\nDone.\nRan in {end - start:.0f} s.')
