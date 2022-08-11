#!/usr/bin/env python
import os
import sys
sys.path.append("../")
import numpy as np
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from shapely.geometry import box  # for cropping centerlines to extent of DEM
from geopandas import clip, read_file
import osmnx  # for querying OpenStreetMaps data to get river centerlines
from scipy.spatial import cKDTree as KDTree  # for finding nearest neighbors/interpolating
from itertools import combinations
import time
from riverrem.RasterViz import RasterViz
import logging

level = logging.INFO
fmt = '[%(levelname)s] %(asctime)s - %(message)s'
logging.basicConfig(level=level, format=fmt)

start = time.time()

usage = """
Script to make river relative elevation model (REM) given a DEM raster as input.
This script can be called from Python using its class/methods or as a CLI utility.

CLI Usage:

"python REMMaker.py [-centerline_shp (default=None)] [-cmap (default=mako_r)] [-z (default=4)] 
                    [-blend_percent (default=25)] [-interp_pts (default=1000)] [-k (default=auto)] 
                    [-eps (default=0.1)] [-workers (default=4)] /path/to/dem"

Options:
    
    -centerline_shp: Path to user-provided river centerline shapefile. If used, overrides OpenStreetMap centerline.
    
    -cmap: Name of a matplotlib or seaborn colormap. Default "mako_r".
           (see https://matplotlib.org/stable/gallery/color/colormap_reference.html)
           
    - z: Vertical exaggeration scale factor for visualization. Default 4.
    
    - blend_percent: Percent of hillshade to blend in with color-relief REM. REM takes opposite weight. Default 25.
    
    -interp_pts: Max number of points to use for interpolation. Actual number of points is limited by number of
                 DEM pixels along centerline, so less points than this will be used for lower resolution DEMs. 
                 Default is 1,000.
    
    -k: Number of nearest neighbor pixels to use for interpolation of river centerline elevations across DEM.
        If no value is supplied, the value of k is automatically estiamted. Higher values make smoother looking REMs
        at the expense of increased computation time.
    
    -eps: Error tolerance in nearest neighbor matching for approximate KD tree query interpolation. 
          Higher values make the interpolation faster at the expense of accuracy.
          Approximate kth nearest neighbors are guaranteed to be no further than (1 + eps) times 
          the distance to the true kth nearest neighbor.
    
    -workers: Number of CPU threads to use when making KD tree query/interpolation. Default is 4. -1 uses all threads.
    
    /path/to/dem: The path to the DEM raster used to make a derived REM.
    
Notes: River centerlines used to create REMs are retrieved from OpenStreetMap (OSM). If a desired river segment is 
       not listed on OSM, a new river centerline can be created/edited at: https://www.openstreetmap.org/edit 
       (clear the ./.osm_cache folder after using the OSM editor to get the updated centerline).
       
       For large/high resolution DEMs, the interpolation can take a long time. Additionally, it may be necessary
       to increase the value of k if interpolation artefacts (discrete linear breaks in REM coloring) are present.
"""


def print_usage():
    print(usage)
    return

def clear_osm_cache():
    """Clear the OSM cache folder (./.osm_cache). This is useful if the OSM Editor has been used to update
    river centerlines."""
    print('Clearing OSM cache.')
    try:
        for dir, subdirs, files in os.walk('./.osm_cache'):
            for f in files:
                filepath = os.path.join(dir, f)
                os.remove(filepath)
    except Exception as e:
        raise Exception("Could not clear ./.osm_cache.", e)
    return

class REMMaker(object):
    """
    Handler to automatically make a river REM from an input DEM.

    :param dem: path to input DEM raster.
    :type dem: str
    :param centerline_shp: (optional) river centerline shapefile to use. If given, overrides OpenStreetMap centerline.
    :type centerline_shp: str
    :param out_dir: output file directory. Defaults to current working directory.
    :type out_dir: str
    :param interp_pts: maximum number of points to use for interpolation of river centerline elevation. Actual number
                       of points is limited by number of DEM pixels along centerline, so less points may be used for
                       lower resolution DEMs.
    :type interp_pts: int
    :param k: number of nearest neighbors to use for IDW interpolation. If None, an appropriate value is estimated.
              The estimation routine uses k between 5-100 points (0.5-10% of the river length) depending on the
              sinuosity of the river of interest. Greater values of k are used for more sinuous rivers.
    :type k: int
    :param eps: fractional error tolerance for finding nearest neighbors in KD tree query. Higher values allow faster
                interpolation at the expense of accuracy.
    :type eps: float
    :param workers: number of CPU threads to use for interpolation. -1 uses all threads.
    :type workers: int
    :param cache_dir: cache directory
    :type cache_dir: str

    """
    def __init__(self, dem, centerline_shp=None, out_dir='./',
                 interp_pts=1000, k=None, eps=0.1, workers=4, cache_dir='./.cache'):
        self.dem = dem
        self.dem_name = os.path.basename(dem).split('.')[0]
        self.centerline_shp = centerline_shp
        self.out_dir = out_dir
        self.cache_dir = cache_dir
        self.get_spatial_metadata()
        self.interp_pts = int(interp_pts)
        self.k = int(k) if k else None
        self.eps = float(eps)
        self.workers = int(workers)
        self.rem_ras = None

    @property
    def dem(self):
        return self._dem

    @dem.setter
    def dem(self, dem):
        if not os.path.exists(dem):
            raise FileNotFoundError(f"Cannot find input DEM: {dem}")
        self._dem = dem
        return

    @property
    def centerline_shp(self):
        return self._centerline_shp

    @centerline_shp.setter
    def centerline_shp(self, centerline_shp):
        if centerline_shp is not None:
            if not os.path.exists(centerline_shp):
                raise FileNotFoundError(f"Cannot find input river centerline shapefile: {centerline_shp}")
        self._centerline_shp = centerline_shp

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out_dir):
        if not os.path.exists(out_dir):
            raise IOError(f"The output directory does not exist: {out_dir}")
        self._out_dir = out_dir
        return

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self._cache_dir = cache_dir
        return

    def get_spatial_metadata(self):
        """
        Get various spatial metadata from DEM to use for processing: projection, EPSG code, DEM array,
        NoData Value, cell size, extent/bbox, function mapping array indices to x,y coordinates.
        """
        # get EPSG code for DEM raster projection
        logging.info("Getting DEM projection.")
        r = gdal.Open(self.dem, gdal.GA_ReadOnly)
        self.proj = osr.SpatialReference(wkt=r.GetProjection())
        self.epsg_code = self.proj.GetAttrValue('AUTHORITY', 1)
        self.h_unit = self.proj.GetAttrValue('UNIT')
        if self.epsg_code is None or self.h_unit is None:
            raise IOError("ERROR: CRS metadata is missing from the input DEM.")
        logging.info("Reading DEM as array.")
        band = r.GetRasterBand(1)
        self.dem_array = band.ReadAsArray()
        # ensure that nodata values become np.nans in array
        self.nodata_val = band.GetNoDataValue()
        if self.nodata_val:
            self.dem_array = np.where(self.dem_array == self.nodata_val, np.nan, self.dem_array)
        rows, cols = self.dem_array.shape
        # get extent of DEM (used to crop/set extent for centerline shapefile/raster)
        logging.info("Getting DEM bounds.")
        upper_left_x, x_size, x_rot, upper_left_y, y_rot, y_size = r.GetGeoTransform()
        self.cell_w, self.cell_h = x_size, y_size
        min_x, max_x = sorted([upper_left_x, upper_left_x + x_size * cols])
        min_y, max_y = sorted([upper_left_y, upper_left_y + y_size * rows])
        self.extent = (min_x, min_y, max_x, max_y)
        # Get lat/long bounding box for DEM raster (used to get OSM features within area)
        lower_right_x = upper_left_x + (r.RasterXSize * x_size)
        lower_right_y = upper_left_y + (r.RasterYSize * y_size)
        source_crs = r.GetSpatialRef()
        target_crs = osr.SpatialReference()
        target_crs.ImportFromEPSG(4326)  # WGS84 Geographic Coordinate System
        transform = osr.CoordinateTransformation(source_crs, target_crs)
        ul_lat, ul_long = transform.TransformPoint(upper_left_x, upper_left_y)[:2]
        lr_lat, lr_long = transform.TransformPoint(lower_right_x, lower_right_y)[:2]
        self.bbox = [ul_lat, lr_lat, lr_long, ul_long]
        # function for mapping indices to x, y coords
        logging.info("Mapping array indices to coordinates.")
        self.ix2coords = lambda t: np.column_stack(np.array([t[0] * x_size + upper_left_x + (x_size / 2),
                                                             t[1] * y_size + upper_left_y + (y_size / 2)]))
        return

    def get_river_centerline(self):
        """Find centerline of river(s) within DEM area using OSM Ways"""
        logging.info("Finding river centerline.")
        # get OSM Ways within bbox of DEM (returns geopandas geodataframe)
        osmnx.settings.cache_folder = './.osm_cache'
        self.rivers = osmnx.geometries_from_bbox(*self.bbox, tags={'waterway': ['river', 'stream', 'tidal channel']})
        if len(self.rivers) == 0:
            raise Exception("No rivers found within the DEM domain. Ensure the target river is on OpenStreetMap\n"
                            "and contains \"waterway\" and \"name\" tags: https://www.openstreetmap.org/edit")
        # read into geodataframe with same CRS as DEM
        self.rivers = self.rivers.to_crs(epsg=self.epsg_code)
        # crop to DEM extent
        self.rivers = clip(self.rivers, box(*self.extent))
        # get river names (drop ones without a name)
        self.rivers = self.rivers.dropna(subset=['name'])
        names = self.rivers.name.values
        # make name attribute more distinct to avoid conflict with geometry name attribute
        self.rivers['river_name'] = names
        # get unique names
        river_names = set(names)
        logging.info(f"Found river(s): {', '.join(river_names)}")
        # find river with greatest length (sum of all segments with same name)
        logging.info("\nRiver lengths:")
        river_lengths = {}
        for river_name in river_names:
            river_segments = self.rivers[self.rivers.river_name == river_name]
            river_length = river_segments.length.sum()
            logging.info(f"\t{river_name}: {river_length:.4f} {self.h_unit}")
            river_lengths[river_name] = river_length
        longest_river = max(river_lengths, key=river_lengths.get)
        self.river_length = river_lengths[longest_river]
        logging.info(f"\nLongest river in domain: {longest_river}\n")
        # if river length is shorter than geometric mean of DEM dimensions, print warning
        x_min, y_min, x_max, y_max = self.extent
        if self.river_length < np.sqrt((x_max - x_min) * (y_max - y_min)):
            print("WARNING: River length is shorter than DEM length. Ensure the target river is on OpenStreetMap\n"
                  "and contains \"waterway\" and \"name\" tags: https://www.openstreetmap.org/edit")
        # only keep longest river to make REM
        self.rivers = self.rivers[self.rivers.river_name == longest_river]
        # convert linestrings of river to points
        self.lines2pts()
        # make shapefile of points
        self.make_river_shp()
        return

    def lines2pts(self):
        """Convert river centerline segment linestrings to a set of points to sample for interpolation."""
        self.river_pts = []
        self.river_endpts = []
        for way, river_segment in self.rivers.iterrows():
            line_string = river_segment.geometry
            point_fraction = line_string.length / self.river_length
            point_num = int(point_fraction * self.interp_pts)
            distances = np.linspace(0, line_string.length, point_num)
            self.river_pts.extend([line_string.interpolate(d) for d in distances])
            self.river_endpts.extend([line_string.interpolate(0), line_string.interpolate(line_string.length)])
        return

    def make_river_shp(self):
        """Make points along river centerline into a shapefile"""
        # create points shapefile
        logging.info("Making river points shapefile.")
        self.river_shp = os.path.join(self.out_dir, f'{self.dem_name}_river_pts.shp')
        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(self.river_shp)
        # create empty multiline geometry layer
        layer = ds.CreateLayer('', self.proj, ogr.wkbPoint)
        # Add fields
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()
        # populate layer with a feature for each natural waterway geometrye
        for p in self.river_pts:
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            # set feature attributes
            feat.SetField('id', 1)
            # set feature geometry from Shapely object
            geom = ogr.CreateGeometryFromWkb(p.wkb)
            feat.SetGeometry(geom)
            # add feature to layer
            layer.CreateFeature(feat)
            feat = geom = None  # destroy these
        # Save and close everything
        ds = layer = feat = geom = None
        return

    def read_centerline_input(self):
        """Read user provided centerline shapefile instead of using OSM."""
        logging.info('Using input centerline shapefile.')
        self.rivers = read_file(self.centerline_shp).to_crs(epsg=self.epsg_code)
        self.rivers = clip(self.rivers, box(*self.extent))
        self.river_length = self.rivers.length.sum()
        self.lines2pts()
        self.make_river_shp()

    def get_river_elev(self):
        """Get DEM values along river centerline"""
        logging.info("Getting river elevation at DEM pixels.")
        # gdal_rasterize centerline
        self.centerline_ras = os.path.join(self.cache_dir, f"{self.dem_name}_centerline.tif")
        extent = f"-te {' '.join(map(str, self.extent))}"
        res = f"-tr {self.cell_w} {self.cell_h}"
        gdal.Rasterize(self.centerline_ras, self.river_shp, options=f"-a id {extent} {res}")
        # raster to numpy array same shape as DEM
        r = gdal.Open(self.centerline_ras, gdal.GA_ReadOnly)
        self.centerline_array = r.GetRasterBand(1).ReadAsArray()
        # remove cells where DEM is null
        self.centerline_array = np.where(np.isnan(self.dem_array), np.nan, self.centerline_array)
        # get coordinates and DEM elevation at river pixels
        self.river_indices = np.where(self.centerline_array == 1)
        self.river_coords = self.ix2coords(self.river_indices)
        self.river_wses = self.dem_array[self.river_indices]
        return

    def get_sinuosity(self):
        """Estimate sinuosity of the river using river centerline(s) length / distance between line endpoints"""
        straight_dist = max([p1.distance(p2) for p1, p2 in combinations(self.river_endpts, 2)])
        sinuosity = self.river_length / straight_dist
        sinuosity = max(1, sinuosity)  # ensure >= 1
        return sinuosity

    def estimate_k(self):
        """Determine the number of k nearest neighbors to use for interpolation"""
        logging.info("Estimating k.")
        # get total number of river pixels, sinuosity
        river_pixels = len(self.river_wses)
        sinuosity = self.get_sinuosity()
        # scale factor goes from 1 (at sinuosity = 1), asymptotes to 5 for very sinuous river
        scale_factor = 1 + 4 * np.tanh(sinuosity - 1)
        # Use 2% of sampled river pixels times scale factor
        k = int(2 * river_pixels / 1e2 * scale_factor)
        logging.info(f"Guessing k = {k}")
        # make k be a minimum of 5, maximum of 10% of self.interp_pts
        k = min(int(self.interp_pts / 10), max(5, k))
        return k

    def interp_river_elev(self):
        """
        Interpolate elevation at river centerline across DEM extent.
        Time for KDTree query scales with log(k).
        """
        logging.info("Interpolating river elevation across DEM extent.")
        if not self.k:
            self.k = self.estimate_k()
        logging.info(f"Using k = {self.k} nearest neighbors.")
        # coords to interpolate over (don't interpolate where DEM is null or on centerline where REM = 0)
        interp_indices = np.where(~(np.isnan(self.dem_array) | (self.centerline_array == 1)))
        logging.info("Getting coords of points to interpolate.")
        c_interpolate = self.ix2coords(interp_indices)
        # create 2D tree
        logging.info("Constructing tree.")
        tree = KDTree(self.river_coords)
        # find k nearest neighbors
        logging.info("Querying tree.")
        logging.info("Chunking query...")
        chunk_size = 1e6
        # iterate over chunks
        chunk_count = c_interpolate.shape[0] // chunk_size + 1
        interpolated_values = np.array([])
        for i, chunk in enumerate(np.array_split(c_interpolate, chunk_count)):
            logging.info(f"{i / chunk_count * 100:.2f}%")
            distances, indices = tree.query(chunk, k=self.k, eps=self.eps, workers=self.workers)
            # interpolate (IDW with power = 1)
            weights = 1 / distances  # weight river elevations by 1 / distance
            weights = weights / weights.sum(axis=1).reshape(-1, 1)  # normalize weights
            interpolated_values = np.append(interpolated_values, (weights * self.river_wses[indices]).sum(axis=1))
        # create interpolated WSE array as elevations along centerline, nans everywhere else
        logging.info("Created interpolated WSE array.")
        self.wse_interp_array = np.where(self.centerline_array == 1, self.dem_array, np.nan)
        # add the interpolated eleation values
        self.wse_interp_array[interp_indices] = interpolated_values
        return

    def detrend_dem(self):
        """Subtract interpolated river elevation from DEM elevation to get REM"""
        logging.info("\nDetrending DEM.")
        self.rem_array = self.dem_array - self.wse_interp_array
        self.rem_ras = os.path.join(self.out_dir, f"{self.dem_name}_REM.tif")
        # set nans back to nodata value
        self.rem_array = np.where(np.isnan(self.rem_array), self.nodata_val, self.rem_array)
        # make copy of DEM raster
        r = gdal.Open(self.dem, gdal.GA_ReadOnly)
        driver = gdal.GetDriverByName("GTiff")
        rem = driver.CreateCopy(self.rem_ras, r, strict=0)
        # fill with REM array
        rem.GetRasterBand(1).WriteArray(self.rem_array)
        return self.rem_ras

    def make_rem(self):
        """Make a relative elevation model (REM). Note that this method creates a raw REM raster and doesn't apply
        color-relief/shading for visualization.

        :returns: path to output REM raster.
        :rtype: str

        """
        if self.centerline_shp is None:
            self.get_river_centerline()
        else:
            self.read_centerline_input()
        self.get_river_elev()
        self.interp_river_elev()
        self.detrend_dem()
        self.clean_up()
        return self.rem_ras

    def make_rem_viz(self, cmap='mako_r', z=4, blend_percent=25, make_png=True, make_kmz=False, *args, **kwargs):
        """Create REM visualization by blending the REM color-relief with a DEM hillshade to make a pretty finished
        product.

        :param cmap: name of matplotlib/seaborn named colormap to use for REM coloring
                     (see https://matplotlib.org/stable/gallery/color/colormap_reference.html). Note the applied
                     colormap is logarithmically scaled in order to emphasize elevations differences close to the river
                     centerline.
        :type cmap: str
        :param z: z factor for exaggerating vertical scale differences of hillshade.
        :type z: float >1
        :param blend_percent: Percent weight of hillshdae in blended image, color-relief takes opposite weight.
        :type blend_percent: float [0-100]

        :returns: path to output raster
        :rtype: str
        """
        if not self.rem_ras:
            logging.info("No REM exists yet. Creating REM now.")
            self.make_rem()
        logging.info("\nBlending REM with hillshade.")
        # make hillshade of original DEM in cache dir
        dem_viz = RasterViz(self.dem, out_dir=self.cache_dir, out_ext=".tif")
        dem_viz.make_hillshade(multidirectional=True, z=z)
        # make color-relief of REM in cache dir
        rem_viz = RasterViz(self.rem_ras, out_dir=self.cache_dir, out_ext=".tif", make_png=make_png, make_kmz=make_kmz, *args, **kwargs)
        rem_viz.make_color_relief(cmap=cmap, log_scale=True, *args, **kwargs)
        # switch output location from cache to output directory for hillshade-color raster/png
        rem_viz.out_rasters["hillshade-color"] = os.path.join(self.out_dir, f"{self.dem_name}_hillshade-color.tif")
        rem_viz.hillshade_ras = dem_viz.hillshade_ras  # use hillshade of original DEM, color-relief of REM
        rem_viz.viz_srs = rem_viz.proj  # make png visualization using source projection
        viz_ras = rem_viz.make_hillshade_color(blend_percent=blend_percent, *args, **kwargs)
        self.clean_up()
        return viz_ras

    def clean_up(self):
        for dir, subdirs, files in os.walk(self.cache_dir):
            for f in files:
                try:
                    filepath = os.path.join(dir, f)
                    os.remove(filepath)
                except:
                    logging.warning(f"Cannot delete cache file {filepath}.")
        return


if __name__ == "__main__":
    # CLI call parsing
    argv = sys.argv
    if (len(argv) < 2) or (("-h" in argv) or ("--help" in argv)):
        print_usage()
    else:
        dem = argv[-1]
        maker_kwargs = {}
        viz_kwargs = {}
        type_dict = {'centerline_shp': str, 'interp_pts': int, 'k': int, 'eps': float, 'workers': int,
                     'cmap': str, 'z': float, 'blend_percent': float}
        for i, arg in enumerate(argv):
            if arg in ['-centerline_shp', '-interp_pts', '-k', '-eps', '-workers']:
                k = arg.replace('-', '')
                maker_kwargs[k] = type_dict[k](argv[i+1])
            if arg in ['-cmap', '-z', '-blend_percent']:
                k = arg.replace('-', '')
                viz_kwargs[k] = type_dict[k](argv[i+1])
        rem_maker = REMMaker(dem=dem, **maker_kwargs)
        rem_maker.make_rem()
        rem_maker.make_rem_viz(**viz_kwargs)

    end = time.time()
    logging.info(f'\nDone.\nRan in {end - start:.0f} s.')
