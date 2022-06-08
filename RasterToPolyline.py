import os
import gdal
import ogr
import numpy as np
import itertools

"""
Custom script to create a polyline shapefile from a raster input.

Input raster must have a value of 1 in all cells that will be part of lines
"""


class RasterToPolyline(object):
    def __init__(self, ras):
        self.ras = ras
        self.out_shp = 'ras2polyline.shp'
        self.ras_array = self.ras2array()
        self.get_dims()
        self.lines = []
        self.array2shp()

    def ras2array(self):
        """Read raster as numpy array"""
        print('Reading input raster.')
        r = gdal.Open(self.ras)
        array = r.ReadAsArray()
        # for test purposes
        arr = np.array([[1, 0, 1, 0],
                        [0, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        # coerce array to all ones or zeros
        array = np.where(array == 1, 1, 0)
        return array

    def get_dims(self):
        """Get dimensions of raster cells"""
        r = gdal.Open(self.ras)
        self.spatial_ref = r.GetSpatialRef()
        gt = r.GetGeoTransform()
        self.x_origin = gt[0]
        self.y_origin = gt[3]
        self.cell_w = gt[1]
        self.cell_h = gt[5]
        return

    def indices2coords(self, xi, yi):
        """Convert array indices to coordinates"""
        x = self.x_origin + self.cell_w * (xi + 1/2)  # adding 1/2 makes it align with raster cell centers
        y = self.y_origin + self.cell_h * (yi + 1/2)
        return x, y

    def make_lines(self, x, y, bytecode):
        """Make set of lines from an x, y coord to all active neighbors given bytecode"""
        lines = []
        # down
        if bytecode >= 128:
            self.lines.append(self.make_line(x, y, x, y + self.cell_h))
            bytecode -= 128
        # up
        if bytecode >= 64:
            self.lines.append(self.make_line(x, y, x, y - self.cell_h))
            bytecode -= 64
        # right
        if bytecode >= 32:
            self.lines.append(self.make_line(x, y, x + self.cell_w, y))
            bytecode -= 32
        # left
        if bytecode >= 16:
            self.lines.append(self.make_line(x, y, x - self.cell_w, y))
            bytecode -= 16
        # down and right
        if bytecode >= 8:
            self.lines.append(self.make_line(x, y, x + self.cell_w, y + self.cell_h))
            bytecode -= 8
        # down and left
        if bytecode >= 4:
            self.lines.append(self.make_line(x, y, x - self.cell_w, y + self.cell_h))
            bytecode -= 4
        # up and right
        if bytecode >= 2:
            self.lines.append(self.make_line(x, y, x + self.cell_w, y - self.cell_h))
            bytecode -= 2
        # up and left
        if bytecode >= 1:
            self.lines.append(self.make_line(x, y, x - self.cell_w, y - self.cell_h))
            bytecode -= 1
        assert bytecode == 0
        if len(lines) > 4:
            return lines[-4:]
        else:
            return lines

    def make_line(self, x1, y1, x2, y2):
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(x1, y1)
        line.AddPoint(x2, y2)
        return line

    def array2shp(self):
        print('Getting neighbors array.')
        # identify neighbors of each pixel
        neighbor_array = np.zeros_like(self.ras_array)
        # down
        neighbor_array[:-1] += self.ras_array[1:] * 128
        # up
        neighbor_array[1:] += self.ras_array[:-1] * 64
        # right
        neighbor_array[:, :-1] += self.ras_array[:, 1:] * 32
        # left
        neighbor_array[:, 1:] += self.ras_array[:, :-1] * 16
        # down and right
        neighbor_array[:-1, :-1] += self.ras_array[1:, 1:] * 8
        # down and left
        neighbor_array[:-1, 1:] += self.ras_array[1:, :-1] * 4
        # up and right
        neighbor_array[1:, :-1] += self.ras_array[:-1, 1:] * 2
        # up and left
        neighbor_array[1:, 1:] += self.ras_array[:-1, :-1] * 1
        # remove cells that have active neighbors but are not active themselves
        neighbor_array = np.where(self.ras_array == 1, neighbor_array, np.nan)
        # now we have an array of bytecodes that tell us which neighbors each active pixel shares

        # array indices for all valid points with neighbors
        yis, xis = np.where(~np.isnan(neighbor_array))
        tot = len(yis)
        prog = 0
        print('Making lines...')
        for i, (yi, xi) in enumerate(zip(yis, xis)):
                if i/tot * 100 > prog:
                    print(f"{prog}%")
                    prog += 10
                bytecode = neighbor_array[yi, xi]
                x, y = self.indices2coords(xi, yi)
                self.make_lines(x, y, bytecode)

        # join individual lines into multiline object
        print('Creating multiline geometry.')
        multiline = ogr.Geometry(ogr.wkbMultiLineString)
        for line in self.lines:
            multiline.AddGeometry(line)

        # save as shapefile
        print('Saving as shapefile.')
        shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(self.out_shp):
            shpDriver.DeleteDataSource(self.out_shp)

        outDataSource = shpDriver.CreateDataSource(self.out_shp)
        try:
            outLayer = outDataSource.CreateLayer(self.out_shp, self.spatial_ref, geom_type=ogr.wkbMultiLineString)
        except AttributeError:
            raise IOError(f"File {self.out_shp} is in use and cannot be deleted. Please delete and try again.")
        featureDefn = outLayer.GetLayerDefn()
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(multiline)
        outLayer.CreateFeature(outFeature)
        # how to avoid duplicate lines?
        return


if __name__ == "__main__":
    test_ras = 'log_dinfarea.tif'
    RasterToPolyline(test_ras)

