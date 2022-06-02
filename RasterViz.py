#!/usr/bin/env python
import os
import sys
import gdal
import osr
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import time
start = time.time()  # track time for script to finish

usage = """
Python script for OpenTopography Raster Derivative/Visualization products.
This script can be called from Python using its class/methods or as a CLI utility.

CLI Usage:

"python RasterViz.py viz_type [-z (default=1)] [-alt (default=45)] [-azim (default=315)] [-multidirectional] 
    [-cmap (default=terrain)] [-make_png] [-make_kmz] [-docker] /path/to/dem.tif"

Options:
    
    viz_type: string corresponding to raster product to be produced, one of the following strings: 
        ["hillshade", "slope", "aspect", "roughness", "color-relief", "hillshade-color"]
    
    -z: only if using "hillshade" or "hillshade-color" viz_type,
        factor to scale/exaggerate vertical topographic differences. Default 1 (no rescale).
    
    -alt: only if using "hillshade" or "hillshade-color" viz_type, 
          altitude of light source in degrees [0-90]. Default 45.
    
    -azim: only if using "hillshade" or "hillshade-color" viz_type, 
           azimuth of light source in degrees [0-360]. Default 315.
    
    -multidirectional: only if using "hillshade" or "hillshade-color" viz_type. 
                       Makes multidirectional hillshade, overriding alt and azim args.
    
    -cmap: only if using "color-relief" viz_type, name of a matplotlib colormap. Default "terrain".
           (see https://matplotlib.org/stable/gallery/color/colormap_reference.html)
    
    -make_png: output a png version (EPSG:3857) of the viz_type in addition to the viz raster in source projection.
    
    -make_kmz: output a kmz version of the viz_type in addition to the viz raster in source projection.
    
    -docker: run GDAL commands from within the osgeo/gdal docker container. This makes it run slightly slower but will
             be needed to run all features if gdal install is on a version <2.2. If using docker, input path must be
             within the working directory path or a subdirectory.

    /path/to/dem.tif: path to input DEM, currently assumed to be in GeoTIFF format.

Notes: Output file naming convention uses the DEM basename as a prefix, then viz_type and file extension. 
       E.g. output.tin.tif --> output_hillshade.tif, output_hillshade.png, output_hillshade.kmz.
       Outputs are saved to the working directory.

Dependencies:
    - Python >=3.6
    - GDAL >=2.2 (or run with lower version in environment using -docker flag)
"""


def print_usage():
    print(usage)
    return


class RasterViz(object):
    """
    Raster visualization object for handling DEM derivative product visualization processing routines.

    Usage: instantiate a RasterViz object with the DEM path as input, e.g.
        viz = RasterViz(dem_path, make_png=[True/False], make_kmz=[True/False], docker_run=[True/False])
    Then call the desired methods with optional arguments.
    Current methods corresponding to different output products:
        self.make_hillshade()
        self.make_slope()
        self.make_aspect()
        self.make_roughness()
        self.make_color_relief()
        self.make_hillshade_color()

    See individual methods for further descriptions.

    TODO
        - test on global datasets, different projections
        - make thumbnail/downsampled .png
        - mockup GUI frontend
    """
    def __init__(self, dem, make_png=False, make_kmz=False, docker_run=False, *args, **kwargs):
        # ref working directory in windows or linux
        pwd = "%cd%" if sys.platform == "win32" else "$(pwd)"
        if docker_run:
            # docker run string to call a command in the osgeo/gdal container
            self.drun = f"docker run -v {pwd}:/data osgeo/gdal "
            # docker path for mounted pwd volume
            self.dp = "/data/"
        else:
            self.drun = ""
            self.dp = ""
        # the input DEM path
        self.dem = dem
        # used as prefix for all output filenames
        self.dem_name = os.path.basename(self.dem).split('.')[0]
        # get projection of DEM
        self.proj = self.get_projection()
        # scale horizontal units from lat/long --> meters for hillshade, slope in WGS84 coords (assumes near equator)
        self.scale = "-s 111120" if self.proj == "EPSG:4326" else ""
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
                          "color-relief": self.make_color_relief,
                          "hillshade-color": self.make_hillshade_color}
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
    def make_hillshade(self, z=1, alt=45, azim=315, multidirectional=False, *args, **kwargs):
        """
        Make hillshade raster from DEM.
        :param z: z factor for scaling/exaggerating vertical scale differences (default 1) [>0].
        :param alt: numeric, altitude of light source in degrees (default 45) [0-90]
        :param azim: numeric, azimuth for light source in degrees (default 315) [0-360]
        :param multidirectional: bool, makes multidirectional hillshade if True, overriding alt and azim.
        """
        if multidirectional:
            print("\nMaking multidirectional hillshade raster.")
        else:
            print(f"\nMaking hillshade raster with alt={alt}, azim={azim}.")
        light_source = "-multidirectional" if multidirectional else f"-az {azim} -alt {alt}"
        z_fact = f"-z {z}" if z != 1 else ""
        temp_path = self.intermediate_rasters["hillshade"]
        out_path = self.hillshade_ras
        # create intermediate hillshade
        cmd = f"{self.drun}gdaldem hillshade {self.dp}{self.dem} {self.dp}{temp_path} " \
              f"{z_fact} {self.scale} {light_source} -of {self.out_format}"
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
              f"{self.scale} -of {self.out_format}"
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
        Make color relief map from DEM (4 band RGBA raster)
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
        # blend images using GDAL and numpy to linearly interpolate RGB values
        temp_path = self.blend_images()
        out_path = f"{self.dem_name}_hillshade-color{self.ext}"
        self.tile_and_compress(temp_path, out_path)
        return out_path

    def blend_images(self, blend_percent=60):
        """
        Blend hillshade and color-relief rasters by linearly interpolating RGB values
        :param blend_percent: Percent weight of hillshdae in blend, color-relief takes opposite weight. Default 60.
        """
        # read in hillshade and color relief rasters
        hs = gdal.Open(self.hillshade_ras, gdal.GA_ReadOnly)
        cr = gdal.Open(self.color_relief_ras, gdal.GA_ReadOnly)
        # make a copy of color-relief raster
        driver = gdal.GetDriverByName(self.out_format)
        blend_ras_name = self.intermediate_rasters["hillshade-color"]
        blend_ras = driver.CreateCopy(blend_ras_name, cr, strict=0)
        hs_array = hs.ReadAsArray()
        # linearly interpolate RGB values between hillshade and color-relief
        for i in range(3):
            blended_band = blend_percent / 100 * hs_array + \
                           (1 - blend_percent / 100) * cr.GetRasterBand(i+1).ReadAsArray()
            blend_ras.GetRasterBand(i+1).WriteArray(blended_band)
        return blend_ras_name

    @staticmethod
    def copy_ras_metadata(src_ras, target_ras):
        """Copy GeoTIFF metadata (CRS, extent, spatial ref) from source to target raster."""
        s = gdal.Open(src_ras, gdal.GA_ReadOnly)
        t = gdal.Open(target_ras, gdal.GA_Update)
        t.SetGeoTransform(s.GetGeoTransform())
        t.SetProjection(s.GetProjection())
        t.SetSpatialRef(s.GetSpatialRef())
        return target_ras

    @staticmethod
    def bin_alpha(ras):
        """Set alpha channel of RGBA ras to 0 where it is <100%"""
        r = gdal.Open(ras, gdal.GA_Update)
        alpha = r.ReadAsArray()[3]
        alpha = np.where(alpha < 255, 0, alpha)
        r.GetRasterBand(4).WriteArray(alpha)
        return ras

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
        gtif = gdal.Open(self.dem, gdal.GA_ReadOnly)
        proj = osr.SpatialReference(wkt=gtif.GetProjection())
        epsg_code = "EPSG:" + proj.GetAttrValue('AUTHORITY', 1)
        return epsg_code

    def get_extent(self):
        """Get extent for DEM raster."""
        gtif = gdal.Open(self.dem, gdal.GA_ReadOnly)
        geo_transform = gtif.GetGeoTransform()
        extent = " ".join([str(x) for x in geo_transform])
        return extent

    def get_elev_range(self):
        """Get range (min, max) of DEM elevation values."""
        gtif = gdal.Open(self.dem, gdal.GA_ReadOnly)
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
        print("\nGenerating .png file.")
        png_name = ras_path.replace(self.ext, ".png")
        # translate from DEM srs to EPSG 3857 (if DEM is not in this srs)
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
        print("\nGenerating .kmz file.")
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
    # example Python here
    """
    dem = './test_dems/sc_river.tin.tif'
    viz = RasterViz(dem=dem, make_png=True, make_kmz=True, docker_run=False)
    viz.make_hillshade(alt=42, azim=217, multidirectional=True)
    viz.make_color_relief(cmap='terrain')
    viz.make_hillshade_color(multidirectional=True)
    viz.make_slope()
    viz.make_aspect()
    viz.make_roughness()
    """

    argv = sys.argv
    if (len(argv) < 2) or (("-h" in argv) or ("--help" in argv)):
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
        docker_run = True if ("-docker" in argv) else False
        # instantiate RasterViz object
        viz = RasterViz(dem=dem, make_png=make_png, make_kmz=make_kmz, docker_run=docker_run)
        # handle args/kwargs for hillshade
        if viz_type in ["hillshade", "hillshade-color"]:
            for i, arg in enumerate(argv):
                if arg in ["-z", "-alt", "-azim"]:
                    k = arg.replace('-', '')
                    kwargs[k] = float(argv[i+1])
                if arg == '-multidirectional':
                    k = arg.replace('-', '')
                    kwargs[k] = True
        # handle args/kwargs for color-relief
        if viz_type in ["color-relief", "hillshade-color"]:
            for i, arg in enumerate(argv):
                if arg == "-cmap":
                    k = arg.replace('-', '')
                    kwargs[k] = argv[i+1]

        # call viz method
        viz.viz_types[viz_type](*args, **kwargs)

        end = time.time()
        print(f'Done.\nRan in {end - start:.0f} s.')
