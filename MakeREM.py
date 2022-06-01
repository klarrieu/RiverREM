#!/usr/bin/env python
import os
import sys
import gdal
import osr
import ogr
import osmnx  # for querying OpenStreetMaps data to get river centerlines
import subprocess
import time


start = time.time()

"""An attempt to autoamtically make river REM from DEM"""


class REMMaker(object):
    def __init__(self, dem):
        self.dem = dem
        self.proj = self.get_projection()
        # bbox (n_lat, s_lat, e_lon, w_lon) of DEM
        self.bbox = self.get_bbox()
        self.run()

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
        return epsg_code

    def get_bbox(self):
        """Get extent for DEM raster."""
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
        """Find river centerline within DEM area using OSM Ways"""
        print("\nFinding river centerline.")
        # get OSM Ways within bbox of DEM (returns geopandas geodataframe)
        self.ways = osmnx.geometries_from_bbox(*self.bbox, tags={'waterway': ['river', 'stream', 'tidal channel']})
        self.ways = self.ways.dropna(subset=['name'])
        self.rivers = []
        for i in range(len(self.ways)):
            print(i)
            osmid = "W" + str(self.ways.index[i][1])
            river = osmnx.geocode_to_gdf(osmid, by_osmid=True).to_crs(epsg=self.proj)
            river.name = self.ways.iloc[i]['name']
            self.rivers.append(river)
        print(f"Found river(s): {', '.join(set(self.ways['name']))}")
        self.make_river_shp()

        return

    def make_river_shp(self):
        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource('rivers.shp')
        layer = ds.CreateLayer('', None, ogr.wkbMultiLineString)
        # Add one attribute
        layer.CreateField(ogr.FieldDefn('name', ogr.OFTString))
        defn = layer.GetLayerDefn()

        ## If there are multiple geometries, put the "for" loop here
        for way in self.rivers:
            print(f"Making layer for {way}")
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            feat.SetField('name', way.name)

            # Make a geometry, from Shapely object
            geom = ogr.CreateGeometryFromWkb(way['geometry'][0].wkb)
            feat.SetGeometry(geom)

            layer.CreateFeature(feat)
            feat = geom = None  # destroy these

        # Save and close everything
        ds = layer = feat = geom = None

    def interp_river_elev(self):
        """Interpolate elevation at river centerline across DEM extent."""
        print("\nInterpolating river elevation across DEM extent.")
        return

    def detrend_dem(self):
        """Subtract interpolated river elevation from DEM elevation to get REM"""
        print("\nDetrending DEM.")
        return

    def make_image_blend(self):
        """Blend REM with DEM hillshade to make pretty finished product"""
        print("\nBlending REM with hillshade.")
        return

    def run(self):
        """Make pretty REM/hillshade blend from DEM"""
        self.get_river_centerline()
        self.interp_river_elev()
        self.detrend_dem()
        self.make_image_blend()
        return


if __name__ == "__main__":
    dem = "./test_dems/sc_river.tin.tif"
    rem_maker = REMMaker(dem=dem)

    end = time.time()
    print(f'\nDone.\nRan in {end - start:.0f} s.')
