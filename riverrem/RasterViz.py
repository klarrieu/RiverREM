#!/usr/bin/env python
import os
import sys
from osgeo import gdal
from osgeo import osr
import subprocess
import seaborn as sn
import numpy as np
import time
start = time.time()  # track time for script to finish

usage = """
Python script for OpenTopography Raster Derivative/Visualization products.
This script can be called from Python using its class/methods or as a CLI utility.

CLI Usage:

"python RasterViz.py viz_type [-z (default=1)] [-alt (default=45)] [-azim (default=315)] [-multidirectional] 
    [-cmap (default=terrain)] [-out_ext tif | img (default=tif)] [-make_png] [-make_kmz] [-docker] [-shell] 
    /path/to/dem"

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
    
    -cmap: only if using "color-relief" viz_type, name of a matplotlib or seaborn colormap. Default "terrain".
           (see https://matplotlib.org/stable/gallery/color/colormap_reference.html)
    
    -out_ext: the extension/file format to use for geodata outputs (tif or img). Default "tif".
    
    -make_png: output a png version (EPSG:3857) of the viz_type in addition to the viz raster in source projection.
    
    -make_kmz: output a kmz version of the viz_type in addition to the viz raster in source projection.
    
    -docker: run GDAL commands from within the osgeo/gdal docker container. This makes it run slightly slower but will
             be needed to run all features if gdal install is on a version <2.2. If using docker, input path must be
             within the working directory path or a subdirectory.
             
    -shell: call GDAL functions (gdaldem, gdal_translate, gdalwarp) as shell commands instead of using Python bindings.
            This may be faster than using pure Python but requires additional environment configuration.

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
    """Handler to produce DEM derivatives/visualizations.

    :param dem: path to input DEM, either in GeoTIFF (.tif), ASCII (.asc), or IMG (.img) format.
    :type dem: str
    :param out_dir: output file directory. Defaults to current working directory.
    :type out_dir: str
    :param out_ext: extension for output georaster files.
    :type out_ext: str, '.tif' or '.img'
    :param make_png: output a png image of visualizations (EPSG:3857) in addition to a raster in source projection.
    :type make_png: bool
    :param make_kmz: output a kmz file (e.g. Google Earth) of visualizations in addition to a raster in source
                     projection.
    :type make_kmz: bool
    :param docker_run: only if shell=True as well, calls gdal utilities from shell with docker container. Must have the
                       osgeo/gdal docker container configured locally to use this option.
    :type docker_run: bool
    :param shell: call gdal utilities from a shell instead of using the Python bindings. May run faster for large files
                  but can be more difficult to configure GDAL environment outside conda.
    :type shell: bool
    :param cache_dir: cache directory
    :type cache_dir: str

    """
    def __init__(self, dem, out_dir='./', out_ext=".tif", make_png=False, make_kmz=False, docker_run=False, shell=False,
                 cache_dir='./.cache', *args, **kwargs):
        # set output and cache directories
        self.out_dir = out_dir
        self.cache_dir = cache_dir
        # call gdal from shell (faster) or Python bindings (easier install)
        self.shell = shell
        # ref working directory in windows or linux
        pwd = "%cd%" if sys.platform == "win32" else "$(pwd)"
        if docker_run and shell:
            # docker run string to call a command in the osgeo/gdal container
            self.drun = f"docker run -v {pwd}:/data osgeo/gdal "
            # docker path for mounted pwd volume
            self.dp = "/data/"
        else:
            self.drun = ""
            self.dp = ""
        # file extension to use for outputs
        self.ext = out_ext
        # dict mapping file extension name to gdal format shortname
        self.format_dict = {".tif": "GTiff",
                            ".img": "HFA",
                            ".asc": "AAIGrid",
                            ".png": "PNG",
                            ".kmz": "KMLOVERLAY"}
        # format to use for output files
        self.out_format = self.format_dict[self.ext]
        # the input DEM path
        self.dem = dem
        # used as prefix for all output filenames
        self.dem_name = os.path.basename(dem).split('.')[0]
        # get projection, horizontal units of DEM
        self.proj, self.h_unit = self.get_projection()
        # scale horizontal units from lat/long --> meters when calculating hillshade, slope (assumes near equator)
        self.scale = 111120 if self.h_unit == "degree" else 1
        # coordinate system to use for non-geodata visualization (i.e. PNG)
        self.viz_srs = "EPSG:3857"
        # determine if we will make png/kmz files after making GeoTIFF
        self.make_png = make_png
        self.make_kmz = make_kmz
        # dict mapping CLI input viz_type strings to corresponding method
        self.viz_types = {"hillshade": self.make_hillshade,
                          "slope": self.make_slope,
                          "aspect": self.make_aspect,
                          "roughness": self.make_roughness,
                          "color-relief": self.make_color_relief,
                          "hillshade-color": self.make_hillshade_color}
        # names for intermediate and output rasters
        self.intermediate_rasters = {viz: os.path.join(self.cache_dir, f"intermediate_{viz}{self.ext}")
                                     for viz in self.viz_types.keys()}
        self.out_rasters = {viz: os.path.join(self.out_dir, f"{self.dem_name}_{viz}{self.ext}")
                            for viz in self.viz_types.keys()}
        # for hillshade-color, keep track of hillshade and color-relief
        self.hillshade_ras = self.out_rasters["hillshade"]
        self.color_relief_ras = self.out_rasters["color-relief"]

    @property
    def dem(self):
        return self._dem

    @dem.setter
    def dem(self, dem):
        # make sure the given DEM exists
        if not os.path.exists(dem):
            raise FileNotFoundError(f"Cannot find input DEM: {dem}")
        self._dem = dem
        # if given DEM is in ASCII, convert to GeoTIFF
        if dem.lower().endswith('.asc'):
            print("Given DEM was in ASCII format, converting to GeoTIFF.")
            self._dem = self._asc_to_tif(dem)
        # if given DEM doesn't have NoData value, assume it is zero
        self._check_dem_nodata()
        return

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

    def _asc_to_tif(self, asc):
        """Convert ascii grid to geotiff"""
        tif_name = os.path.basename(asc).split('.')[0] + ".tif"
        if self.shell:
            cmd = f"{self.drun}gdal_translate -of GTiff {self.dp}{asc} {self.dp}{tif_name}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            gdal.Translate(f"{self.dp}{tif_name}", f"{self.dp}{asc}", format="GTiff")
        return tif_name

    def _check_dem_nodata(self):
        """Check that input DEM has a NoData value set. If not, set to zero."""
        r = gdal.Open(self._dem, gdal.GA_Update)
        band = r.GetRasterBand(1)
        if band.GetNoDataValue() is None:
            print("WARNING: NoData value not found for input DEM. Assuming NoData value is 0.")
            # make a copy of the DEM
            dem_copy_name = os.path.join(self.cache_dir, f"dem_copy{self.ext}")
            driver = gdal.GetDriverByName(self.out_format)
            dem_copy = driver.CreateCopy(dem_copy_name, r, strict=0)
            # assign nodata value for copy
            dem_copy.GetRasterBand(1).SetNoDataValue(0)
            # if there are np.nans in raster, set those values to our newly assigned nodata value (0)
            if np.any(np.isnan(band.ReadAsArray())):
                print("Writing nans to assigned nodata value...")
                arr = dem_copy.GetRasterBand(1).ReadAsArray()
                arr = np.where(np.isnan(arr), 0, arr)
                band.WriteArray(arr)
            # use copy as input DEM
            self._dem = dem_copy_name
        return

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
                self._clean_up()
            return ras_path
        return wrapper

    @_png_kmz_checker
    def make_hillshade(self, z=1, alt=45, azim=315, multidirectional=False, *args, **kwargs):
        """Make hillshade raster from the input DEM.

        :param z: z factor for exaggerating vertical scale differences (default 1).
        :type z: float >1
        :param alt: altitude of light source in degrees (default 45).
        :type alt: float [0-90]
        :param azim: azimuth for light source in degrees (default 315).
        :type azim: float [0-360]
        :param multidirectional: makes multidirectional hillshade if True, overriding alt and azim.
        :type multidirectional: bool

        :returns: path to output hillshade raster.
        :rtype: str
        """
        if multidirectional:
            print("\nMaking multidirectional hillshade raster.")
        else:
            print(f"\nMaking hillshade raster with alt={alt}, azim={azim}.")
        temp_path = self.intermediate_rasters["hillshade"]
        out_path = self.hillshade_ras
        # create intermediate hillshade
        if self.shell:
            z_fact = f"-z {z}" if z != 1 else ""
            scale = f"-s {self.scale}" if self.scale != 1 else ""
            light_source = "-multidirectional" if multidirectional else f"-az {azim} -alt {alt}"
            cmd = f"{self.drun}gdaldem hillshade {self.dp}{self.dem} {self.dp}{temp_path} " \
                  f"{z_fact} {scale} {light_source} -of {self.out_format}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            options = {"zFactor": z, "format": self.out_format}
            if self.scale != 1:
                options['scale'] = self.scale
            if multidirectional:
                options['multiDirectional'] = True
            else:
                options['altitude'] = alt
                options['azimuth'] = azim
            gdal.DEMProcessing(f"{self.dp}{temp_path}", f"{self.dp}{self.dem}", "hillshade", **options)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_slope(self, *args, **kwargs):
        """Make slope map from DEM, with slope at each pixel in degrees [0-90].

        :returns: path to output slope raster.
        :rtype: str
        """
        print(f"\nMaking slope raster.")
        temp_path = self.intermediate_rasters["slope"]
        out_path = self.out_rasters["slope"]
        if self.shell:
            scale = f"-s {self.scale}" if self.scale != 1 else ""
            cmd = f"{self.drun}gdaldem slope {self.dp}{self.dem} {self.dp}{temp_path} " \
                  f"{scale} -of {self.out_format}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            options = {"format": self.out_format}
            if self.scale != 1:
                options['scale'] = self.scale
            gdal.DEMProcessing(f"{self.dp}{temp_path}", f"{self.dp}{self.dem}", "slope", **options)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_aspect(self, *args, **kwargs):
        """Make aspect map from DEM, with aspect at each pixel in degrees [0-360].

        :returns: path to output aspect raster.
        :rtype: str
        """
        print("\nMaking aspect raster.")
        temp_path = self.intermediate_rasters["aspect"]
        out_path = self.out_rasters["aspect"]
        if self.shell:
            cmd = f"{self.drun}gdaldem aspect {self.dp}{self.dem} {self.dp}{temp_path} " \
                  f"-of {self.out_format}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            gdal.DEMProcessing(f"{self.dp}{temp_path}", f"{self.dp}{self.dem}", "aspect", format=self.out_format)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_roughness(self, *args, **kwargs):
        """Make roughness map from DEM.

        :returns: path to output roughness raster.
        :rtype: str
        """
        print("\nMaking roughness raster.")
        temp_path = self.intermediate_rasters["roughness"]
        out_path = self.out_rasters["roughness"]
        if self.shell:
            cmd = f"{self.drun}gdaldem roughness {self.dp}{self.dem} {self.dp}{temp_path} " \
                  f"-of {self.out_format}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            gdal.DEMProcessing(f"{self.dp}{temp_path}", f"{self.dp}{self.dem}", "Roughness", format=self.out_format)
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_color_relief(self, cmap='terrain', log_scale=False, *args, **kwargs):
        """Make color relief map from DEM (3 band RGB raster).

        :param cmap: matplotlib or seaborn named colormap to use for making color relief map.
                     (see https://matplotlib.org/stable/gallery/color/colormap_reference.html)
        :type cmap: str
        :param log_scale: bool, makes the colormap on a log scale from zero, so terrain closer to 0 elevation
                          has greater color variation. Intended to be used for REMs or coastal datasets.
        :type log_scale: bool

        :returns: path to output color-relief raster.
        :rtype: str
        """
        print(f"\nMaking color relief map with cmap={cmap}.")
        temp_path = self.intermediate_rasters["color-relief"]
        out_path = self.color_relief_ras
        cmap_txt = self.get_cmap_txt(cmap, log_scale=log_scale)
        if self.shell:
            cmd = f"{self.drun}gdaldem color-relief {self.dp}{self.dem} {self.dp}{cmap_txt} {self.dp}{temp_path} " \
                    f"-of {self.out_format}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            gdal.DEMProcessing(f"{self.dp}{temp_path}", f"{self.dp}{self.dem}", "color-relief", colorFilename=cmap_txt,
                               format=self.out_format)
        # set nodata value to 0 for color-relief
        r = gdal.Open(temp_path, gdal.GA_Update)
        [r.GetRasterBand(i+1).SetNoDataValue(0) for i in range(3)]
        r = None
        self.tile_and_compress(temp_path, out_path)
        return out_path

    @_png_kmz_checker
    def make_hillshade_color(self, blend_percent=60, *args, **kwargs):
        """Make a pretty composite hillshade/color-relief image.

        :param blend_percent: Percent weight of hillshdae in blend, color-relief takes opposite weight.
        :type blend_percent: float [0-100]

        This method also accepts all arguments of `make_hillshade` and `make_color_relief` if the respective rasters
        have not yet been created.

        :returns: path to output hillshade-color raster.
        :rtype: str
        """
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
        temp_path = self.blend_images(blend_percent=blend_percent)
        out_path = self.out_rasters["hillshade-color"]
        # set nodata value to 0 for color-relief
        r = gdal.Open(temp_path, gdal.GA_Update)
        [r.GetRasterBand(i+1).SetNoDataValue(0) for i in range(3)]
        r = None
        self.tile_and_compress(temp_path, out_path)
        return out_path

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
            blended_band = b * hs_array + (1 - b) * cr_arrays[i]
            blend_ras.GetRasterBand(i+1).WriteArray(blended_band)
        return blend_ras_name

    def get_cmap_txt(self, cmap='terrain', log_scale=False):
        """Make a matplotlib named colormap into a gdaldem colormap text file for color-relief mapping.
        Format is "elevation R G B" where RGB are in [0-255] range.

        :param cmap: colormap to use for making color relief map.
        :param log_scale: bool, logarithmically scale colormap, showing greatest color variation at zero elevation.
                          Intended to be used for REMs or coastal datasets.

        :return: .txt file containing colormap mapped to DEM
        """
        min_elev, max_elev = self.get_elev_range()
        if log_scale:
            # sample 255 log-spaced elevation values from 0 to max elevation / 2
            elevations = np.logspace(0, np.log10(0.5 * max_elev), 255) - 1
        else:
            # sample 255 linearly spaced elevation values
            elevations = np.linspace(min_elev, max_elev, 255)
        # matplotlib cmap function maps [0-255] input to RGB values [0-1]
        cm_mpl = sn.color_palette(cmap, as_cmap=True)
        # convert output RGB colors on [0-1] range to [1-255] range used by gdaldem (reserving 0 for nodata)
        cm = lambda x: [val * 254 + 1 for val in cm_mpl(x)[:3]]
        # make cmap text file to be read by gdaldem color-relief
        cmap_txt = os.path.join(self.cache_dir, f'{self.dem_name}_cmap.txt')
        with open(cmap_txt, 'w') as f:
            lines = [f'{elev} {" ".join(map(str, cm(i)))}\n' for i, elev in enumerate(elevations)]
            lines.append("nv 0 0 0\n")
            f.writelines(lines)
        return cmap_txt

    def get_projection(self):
        """Get EPSG code for DEM raster projection."""
        ras = gdal.Open(self.dem, gdal.GA_ReadOnly)
        proj = osr.SpatialReference(wkt=ras.GetProjection())
        epsg_code = proj.GetAttrValue('AUTHORITY', 1)
        epsg_code = "EPSG:" + epsg_code if epsg_code is not None else None
        h_unit = proj.GetAttrValue('UNIT')
        if epsg_code is None or h_unit is None:
            print("WARNING: CRS metadata is missing for input DEM.")
        return epsg_code, h_unit

    def get_elev_range(self):
        """Get range (min, max) of DEM elevation values."""
        ras = gdal.Open(self.dem, gdal.GA_ReadOnly)
        elevband = ras.GetRasterBand(1)
        elevband.ComputeStatistics(0)
        min_elev = elevband.GetMinimum()
        max_elev = elevband.GetMaximum()
        return min_elev, max_elev

    def tile_and_compress(self, in_path, out_path):
        """Used to turn intermediate raster viz products into final outputs."""
        print("Tiling and compressing raster.")
        if self.shell:
            cmd = f"{self.drun}gdal_translate {self.dp}{in_path} {self.dp}{out_path} " \
                  f"-co \"COMPRESS=LZW\" -co \"TILED=YES\" " \
                  f"-co \"blockxsize=256\" -co \"blockysize=256\" -co \"COPY_SRC_OVERVIEWS=YES\""
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            gdal.Translate(f"{self.dp}{out_path}", f"{self.dp}{in_path}",
                           options=f"-co \"COMPRESS=LZW\" -co \"TILED=YES\" "
                                   f"-co \"blockxsize=256\" -co \"blockysize=256\" -co \"COPY_SRC_OVERVIEWS=YES\"")
        return out_path

    def raster_to_png(self, ras_path):
        """Convert raster to .png file. Coerce to EPSG:3857 consistent with existing OT service."""
        print("\nGenerating .png file.")
        png_name = ras_path.replace(self.ext, ".png")
        # translate from DEM srs to EPSG 3857 (if DEM is not in this srs)
        if (self.proj is not None) and (self.proj != self.viz_srs):
            tmp_path = os.path.join(self.cache_dir, f"tmp_3857{self.ext}")
            if self.shell:
                cmd = f"{self.drun}gdalwarp " \
                      f"-t_srs {self.viz_srs} {self.dp}{ras_path} {self.dp}{tmp_path}"
                subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
            else:
                gdal.Warp(f"{self.dp}{tmp_path}", f"{self.dp}{ras_path}", dstSRS=self.viz_srs)
        else:
            tmp_path = ras_path
        # convert to PNG
        scale = self.get_scaling(tmp_path)
        if self.shell:
            cmd = f"{self.drun}gdal_translate -ot Byte{scale} -a_nodata 0 -of PNG " \
                  f"{self.dp}{tmp_path} {self.dp}{png_name}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            gdal.Translate(f"{self.dp}{png_name}", f"{self.dp}{tmp_path}",
                           options=f"-ot Byte{scale} -a_nodata 0 -of PNG")
        return png_name

    def raster_to_kmz(self, ras_path):
        """Convert .tif raster to .kmz file"""
        print("\nGenerating .kmz file.")
        kmz_name = ras_path.replace(self.ext, ".kmz")
        scale = self.get_scaling(ras_path)
        if self.shell:
            cmd = f"{self.drun}gdal_translate -ot Byte{scale} -a_nodata 0 -co format=png -of KMLSUPEROVERLAY " \
                  f"{self.dp}{ras_path} {self.dp}{kmz_name}"
            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        else:
            gdal.Translate(f"{self.dp}{kmz_name}", f"{self.dp}{ras_path}",
                           options=f"-ot Byte{scale} -a_nodata 0 -co format=png -of KMLSUPEROVERLAY")
        return kmz_name

    @staticmethod
    def get_scaling(ras_name):
        """Get scaling string for gdal_translate call when converting to Byte array for png/kmz outputs"""
        # color-relief and hillshade-color already have 0-255 color range, don't change
        if "color" in ras_name:
            scale = ""
        else:
            ras = gdal.Open(ras_name, gdal.GA_ReadOnly)
            band = ras.GetRasterBand(1)
            band.ComputeStatistics(0)
            min_val = band.GetMinimum()
            max_val = band.GetMaximum()
            # set output range to start at 1, so we don't erroneously set low values to nodata (0 for byte array)
            scale = f" -scale {min_val} {max_val} 1 255"
        return scale

    def _clean_up(self):
        """Delete all intermediate files. Called by _png_kmz_checker decorator at end of function calls."""
        int_files = [*self.intermediate_rasters.values(),
                     os.path.join(self.cache_dir, f"tmp_3857{self.ext}"),
                     os.path.join(self.cache_dir, f"{self.dem_name}_cmap.txt")]
        int_files += [f + ".aux.xml" for f in int_files]
        for f in int_files:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    # CLI call parsing
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
        for i, arg in enumerate(argv):
            if arg == "-out_ext":
                out_ext = ".img" if argv[i+1] == "img" else ".tif"
        make_png = True if ("-make_png" in argv) else False
        make_kmz = True if ("-make_kmz" in argv) else False
        docker_run = True if ("-docker" in argv) else False
        shell = True if ("-shell" in argv) else False
        # instantiate RasterViz object
        viz = RasterViz(dem=dem, out_ext=out_ext, make_png=make_png, make_kmz=make_kmz,
                        docker_run=docker_run, shell=shell)
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
