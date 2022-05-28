import sys
import gdal
import osr
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

import time
start = time.time()
usage = """
Python script for OpenTopography Raster Visualization products.
This script can be called using its class object/methods or as a CLI. 
See class definition for details of running via OOP.

CLI Usage:

"python3 RasterViz.py viz_type [-alt (default=45)] [-azim (default=315)] [-multidirectional] 
    [-cmap (default=terrain)] [-make_png] [-make_kmz] in_dem.tif"

Options:

    viz_type: string corresponding to method for producing the raster product, one of the following strings: 
        ["hillshade", "slope", "aspect", "roughness", "color-relief"]
        
    in_dem.tif: path to input DEM, currently assumed to be in GeoTIFF format.
    
    -alt: only if using "hillshade" viz_type, altitude of light source in degrees [0-90]. Default 45.
    
    -azim: only if using "hillshade" viz_type, azimuth of light source in degrees [0-360]. Default 315.
    
    -multidirectional: only if using "hillshade" viz_type. Makes multidirectional hillshade, overriding
     alt and azim args.
    
    -cmap: only if using "color-relief" viz_type, name of a matplotlib colormap. Default "terrain".
           (see https://matplotlib.org/stable/gallery/color/colormap_reference.html)
    
    -make_png: output a png version (EPSG:3857) of the viz_type in addition to the viz raster in source projection.
    
    -make_kmz: output a kmz version (EPSG:3857) of the viz_type in addition to the viz raster in source projection.

Output files use the DEM basename (before .) as a prefix with the name of the viz_type appended with an underscore,
and relevant extension. E.g. output.tin.tif --> output_hillshade.tif, output_hillshade.png, output_hillshade.kmz
"""


def print_usage():
    print(usage)
    return


class RasterViz(object):
    """
    Raster visualization object for handling DEM derivative product visualization processing routines.

    Usage: instantiate a RasterViz object with the DEM path as input, e.g.
        viz = RasterViz(dem_path, make_png=[True/False], make_kmz=[True/False])
    Then call the desired methods with optional arguments.
    Current methods corresponding to different output products:
        self.make_hillshade()
        self.make_slope()
        self.make_aspect()
        self.make_roughness()
        self.make_color_relief()

    See individual methods for further descriptions.

    TODO
        - test scaling of outputs with lat/long coord system GeoTIFFs
    """
    def __init__(self, dem, make_png=False, make_kmz=False, *args, **kwargs):
        # prefix to run commands in docker image
        pwd = "%cd%" if sys.platform == "win32" else "$(pwd)"
        # docker run string to call a command in the osgeo/gdal container
        self.drun = f"docker run -v {pwd}:/data osgeo/gdal "
        # docker path for mounted pwd volume
        self.dp = "/data/"
        # path to imagemagick
        self.magick_path = "magick"
        # the input DEM path
        self.dem = dem
        # used as prefix for all output filenames
        self.dem_name = os.path.basename(self.dem).split('.')[0]
        # projection of DEM
        self.proj = self.get_projection()
        self.extent = self.get_extent()
        # determine if we will make png/kmz files after making GeoTIFF
        self.make_png = make_png
        self.make_kmz = make_kmz
        # coordinate system to use for non-geodata visualization (i.e. PNG)
        self.viz_srs = "EPSG:3857"
        # dict mapping gdal format shortnames to file extension name
        self.extension_dict = {"GTiff": ".tif",
                               "PNG": ".png",
                               "KMLOVERLAY": ".kmz"}
        self.out_format = "GTiff"
        self.ext = self.extension_dict[self.out_format]
        # dict mapping CLI input viz_type strings to corresponding method
        self.viz_types = {"hillshade": self.make_hillshade,
                          "slope": self.make_slope,
                          "aspect": self.make_aspect,
                          "roughness": self.make_roughness,
                          "color-relief": self.make_color_relief}
        # for hillshade-color, keep track of hillshade and color-relief
        self.hillshade_ras = f"{self.dem_name}_hillshade{self.ext}"
        self.color_relief_ras = f"{self.dem_name}_color-relief{self.ext}"
        # names for intermediate rasters to make output GeoTIFFs
        self.intermediate_rasters = {viz: f"intermediate_{viz}.tif" for viz in self.viz_types.keys()}
        # names for intermediate rasters to make reproject/make output PNGs

    @property
    def dem(self):
        return self._dem

    @dem.setter
    def dem(self, dem):
        if not os.path.exists(dem):
            raise FileNotFoundError(f"Cannot find input DEM: {dem}")
        self._dem = dem
        return self._dem

    def _png_kmz_checker(func):
        """Used as a wrapper for making viz products, making png and kmz after tif if selected"""
        def wrapper(self, *args, **kwargs):
            try:
                # call decorated method
                ras_path = func(self, *args, **kwargs)
                # make png and kmz if applicable
                if self.make_png:
                    self.raster_to_png(ras_path)
                if self.make_kmz:
                    self.raster_to_kmz(ras_path)
            except Exception as e:
                raise Exception(e)
            finally:
                # clean up regardless of whether something failed
                self.clean_up()
            return
        return wrapper

    @_png_kmz_checker
    def make_hillshade(self, alt=45, azim=315, multidirectional=False, *args, **kwargs):
        """
        Make hillshade raster from DEM.
        :param alt: numeric, altitude of light source in degrees (default 45) [0-90]
        :param azim: numeric, azimuth for light source in degrees (default 315) [0-360]
        :param multidirectional: bool, makes multidirectional hillshade if True, overriding alt and azim.
        """
        if multidirectional:
            print("Making multidirectional hillshade raster.")
        else:
            print(f"\nMaking hillshade raster with alt={alt}, azim={azim}.")
        light_source = "-multidirectional" if multidirectional else f"-az {azim} -alt {alt}"
        temp_path = self.intermediate_rasters["hillshade"]
        out_path = self.hillshade_ras
        # create intermediate hillshade
        cmd = f"{self.drun}gdaldem hillshade {self.dp}{self.dem} {self.dp}{temp_path} " \
              f"{light_source} -of {self.out_format}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_slope(self, *args, **kwargs):
        """Make slope map from DEM, with slope at each pixel in degrees [0-90]"""
        print(f"\nMaking slope raster.")
        temp_path = self.intermediate_rasters["slope"]
        out_path = f"{self.dem_name}_slope{self.ext}"
        cmd = f"{self.drun}gdaldem slope {self.dp}{self.dem} {self.dp}{temp_path} " \
              f"-of {self.out_format}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_aspect(self, *args, **kwargs):
        """Make aspect map from DEM, with aspect at each pixel in degrees [0-360]"""
        print("\nMaking aspect raster.")
        temp_path = self.intermediate_rasters["aspect"]
        out_path = f"{self.dem_name}_aspect{self.ext}"
        cmd = f"{self.drun}gdaldem aspect {self.dp}{self.dem} {self.dp}{out_path} " \
              f"-of {self.out_format}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_roughness(self, *args, **kwargs):
        """Make roughness map from DEM."""
        print("\nMaking roughness raster.")
        temp_path = self.intermediate_rasters["roughness"]
        out_path = f"{self.dem_name}_roughness{self.ext}"
        cmd = f"{self.drun}gdaldem roughness {self.dp}{self.dem} {self.dp}{out_path} " \
              f"-of {self.out_format}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_color_relief(self, cmap='terrain', *args, **kwargs):
        """
        Make color relief map from DEM (3 band RGB raster)
        :param cmap: str, matplotlib colormap to use for making color relief map.
                     (see https://matplotlib.org/stable/gallery/color/colormap_reference.html)
        """
        print(f"\nMaking color relief map with cmap={cmap}.")
        temp_path = self.intermediate_rasters["color-relief"]
        out_path = self.color_relief_ras
        cmap_txt = self.get_cmap_txt(cmap)
        cmd = f"{self.drun}gdaldem color-relief -alpha {self.dp}{self.dem} {self.dp}{cmap_txt} {self.dp}{out_path} " \
                f"-of {self.out_format}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_hillshade_color(self, *args, **kwargs):
        """Make a pretty composite hillshade/color-relief image"""
        # make hillshade and/or color-relief if haven't already
        if not os.path.exists(self.hillshade_ras):
            self.make_hillshade(*args, **kwargs)
        if not os.path.exists(self.color_relief_ras):
            self.make_color_relief(*args, **kwargs)
        print("\nMaking hillshade-color composite raster.")
        out_path = f"{self.dem_name}_hillshade-color{self.ext}"
        # use image magick to blend tifs
        cmd = f"{self.magick_path} composite -blend 60 {self.hillshade_ras} {self.color_relief_ras} {out_path}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        # copy CRS, extent, and spatial reference metadata from DEM to non-georeferenced magick output
        r = gdal.Open(out_path, gdal.GA_Update)
        t = gdal.Open(self.dem)
        r.SetGeoTransform(t.GetGeoTransform())
        r.SetProjection(t.GetProjection())
        r.SetSpatialRef(t.GetSpatialRef())
        # for magick composite -blend, alpha layer is somewhere between 0-255 in nodata areas. Set it back to 0.
        alpha = r.ReadAsArray()[3]
        alpha = np.where(alpha < 255, 0, alpha)
        r.GetRasterBand(4).WriteArray(alpha)
        return out_path

    def get_cmap_txt(self, cmap='terrain'):
        """
        Make a matplotlib named colormap into a gdaldem
        colormap text file for color-relief mapping.
        Format is "elevation R G B" where RGB are in [0-255] range
        :param cmap: colormap to use for making color relief map.
        :return: .txt file containing colormap mapped to DEM
        """
        min_elev, max_elev = self.get_elev_range()
        # sample 256 evenly spaced elevation values
        elevations = np.linspace(min_elev, max_elev, 256)
        # function to convert elevations to [0-1] range (domain for cmap function)
        elev_to_x = lambda x: (x - min_elev) / (max_elev - min_elev)
        # matplotlib cmap function maps [0-1] input to RGBA, each color channel with [0-1] range
        cm_mpl = plt.get_cmap(cmap)
        # convert output RGBA colors on [0-1] range to [0-255] range used by gdaldem
        cm = lambda x: [val * 255 for val in cm_mpl(x)]
        # make cmap text file to be read by gdaldem color-relief
        cmap_txt = f'{self.dem_name}_cmap.txt'
        with open(cmap_txt, 'w') as f:
            lines = [f'{elev} {" ".join(map(str, cm(elev_to_x(elev))))}\n' for elev in elevations]
            lines.append("nv 0 0 0 0\n")
            f.writelines(lines)
        return cmap_txt

    def get_projection(self):
        """Get EPSG code for DEM raster projection."""
        gtif = gdal.Open(self.dem)
        proj = osr.SpatialReference(wkt=gtif.GetProjection())
        epsg_code = "EPSG:" + proj.GetAttrValue('AUTHORITY', 1)
        return epsg_code

    def get_extent(self):
        """Get extent for DEM raster."""
        gtif = gdal.Open(self.dem)
        geo_transform = gtif.GetGeoTransform()
        extent = " ".join([str(x) for x in geo_transform])
        return extent

    def get_elev_range(self):
        """Get range (min, max) of DEM elevation values."""
        gtif = gdal.Open(self.dem)
        elevband = gtif.GetRasterBand(1)
        elevband.ComputeStatistics(0)
        min_elev = elevband.GetMinimum()
        max_elev = elevband.GetMaximum()
        return min_elev, max_elev

    def tile_and_compress(self, in_path, out_path):
        """Used to turn intermediate raster viz products into final outputs."""
        print("Tiling and compressing raster.")
        cmd = f"{self.drun}gdal_translate {self.dp}{in_path} {self.dp}{out_path} " \
              f"-co \"COMPRESS=LZW\" -co \"TILED=YES\" " \
              f"-co \"blockxsize=256\" -co \"blockysize=256\" -co \"COPY_SRC_OVERVIEWS=YES\""
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        return out_path

    def raster_to_png(self, ras_path):
        """Convert raster to .png file. Coerce to EPSG:3857 consistent with existing OT service."""
        print("Generating .png file.")
        png_name = ras_path.replace(self.ext, ".png")
        # translate from DEM srs to EPSG 3857 (if DEM is not in this srs)
        #
        if self.proj != self.viz_srs:
            tmp_path = "tmp_3857.tif"
            cmd = f"{self.drun}gdalwarp " \
                  f"-s_srs {self.proj} -t_srs {self.viz_srs} {self.dp}{ras_path} {self.dp}{tmp_path}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            tmp_path = ras_path
        # convert to PNG
        cmd = f"{self.drun}gdal_translate -ot Byte -scale -of PNG " \
              f"{self.dp}{tmp_path} {self.dp}{png_name}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        return png_name

    def raster_to_kmz(self, ras_path):
        """Convert .tif raster to .kmz file"""
        print("Generating .kmz file.")
        kmz_name = ras_path.replace(self.ext, ".kmz")
        cmd = f"{self.drun}gdal_translate -ot Byte -scale -co format=png -of KMLSUPEROVERLAY " \
              f"{self.dp}{ras_path} {self.dp}{kmz_name}"
        subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        return kmz_name

    def clean_up(self):
        """Delete all intermediate files. Called by _png_kmz_checker decorator at end of function calls."""
        int_files = [*self.intermediate_rasters.values(),
                     "tmp_3857.tif",
                     f"{self.dem_name}_cmap.txt"]
        for f in int_files:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    # example run here
    dem = 'sc_river.tin.tif'
    viz = RasterViz(dem=dem, make_png=True, make_kmz=True)
    viz.make_hillshade(alt=42, azim=217, multidirectional=True)
    viz.make_slope()
    viz.make_aspect()
    viz.make_roughness()
    viz.make_color_relief(cmap='terrain')
    viz.make_hillshade_color(multidirectional=True)

    argv = sys.argv
    if (len(argv) < 2) or (argv[1] in ["-h", "--help"]):
        print_usage()
    else:
        # keep track of args/kwargs for the viz_type
        args = []
        kwargs = {}
        # mandatory args
        viz_type = argv[1]
        dem = argv[-1]
        # optional args for RasterViz init
        make_png = True if ("-make_png" in argv) else False
        make_kmz = True if ("-make_kmz" in argv) else False
        # instantiate RasterViz object
        viz = RasterViz(dem=dem, make_png=make_png, make_kmz=make_kmz)
        # handle args/kwargs for hillshade
        if viz_type == "hillshade":
            for i, arg in enumerate(argv):
                if arg in ["-alt", "-azim"]:
                    k = arg.replace('-', '')
                    kwargs[k] = float(argv[i+1])
                if arg == '-multidirectional':
                    k = arg.replace('-', '')
                    kwargs[k] = True
        # handle args/kwargs for color-relief
        if viz_type == "color-relief":
            for i, arg in enumerate(argv):
                if arg == "-cmap":
                    k = arg.replace('-', '')
                    kwargs[k] = argv[i+1]

        # call viz method
        viz.viz_types[viz_type](*args, **kwargs)

    end = time.time()
    print(f'Ran in {end - start:.0f} s.')
