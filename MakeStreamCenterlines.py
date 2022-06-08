import os
import gdal
import numpy as np
import RasterToPolyline as r2p

"""
Script to make vector shapefile of stream centerlines from TauDEM output "D-Inf Contributing Area"
"""


class StreamCenterline(object):
    def __init__(self, ras):
        self.ras = ras
        self.mask_ras = self.make_mask_ras()
        r2p.RasterToPolyline(self.mask_ras)

    def make_mask_ras(self):
        """Make raster with value of one where streams are"""
        print('Making mask raster.')
        mask_ras_name = "mask_ras.tif"
        r = gdal.Open(self.ras, gdal.GA_ReadOnly)
        ras_array = r.ReadAsArray()
        # streams defined as locations where log10 of contributing area is within a threshold of maximum
        log_area = np.log10(ras_array)
        max_log_area = np.nanmax(log_area)
        print(f"Max log area: {max_log_area}")
        mask_arr = np.where(log_area >= max_log_area - 1.227, 1, 0)
        # convert mask_arr to raster
        driver = gdal.GetDriverByName("GTiff")
        mask_ras = driver.CreateCopy(mask_ras_name, r, strict=0)
        mask_ras.GetRasterBand(1).WriteArray(mask_arr)
        return mask_ras_name


if __name__ == "__main__":
    ras = "Dinfareasca.tif"
    StreamCenterline(ras)
