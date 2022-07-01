# RiverREM

RiverREM is a Python package for automatically generating river relative elevation model (REM) visualizations from nothing but an input digital elevation model (DEM). The package uses the OpenStreetMap API to retrieve river centerline geometries over the DEM extent. Interpolation of river elevations is automatically handled using a sampling scheme based on raster resolution and river sinuosity to create striking high-resolution visualizations without interpolation artefacts straight out of the box and without additional manual steps. See the [documentation](https://klarrieu.github.io/RiverREM/) pages for more details.

![yukon_crop](docs/pics/yukon_crop.png)



## Installation

Install via conda:

`conda install riverrem`

In order to handle dependencies such as GDAL and OSMnx, it is highly recommended to install with `conda` instead of `pip` for ease of use. 

## Usage

1. Get a DEM for the area of interest. Some sources for free topographic data:

   - [OpenTopography](https://opentopography.org/)
   - [USGS](https://apps.nationalmap.gov/downloader/)

2. Create an REM visualization with default arguments:

   ```python
   from riverrem.REMMaker import REMMaker
   # provide the DEM file path and desired output directory
   rem_maker = REMMaker(dem='/path/to/dem.tif', out_dir='/out/dir/')
   # create an REM
   rem_maker.make_rem()
   # create an REM visualization with the given colormap
   rem_maker.make_rem_viz(cmap='mako_r')
   ```

Options for adjusting colormaps, shading, interpolation parameters, and more are detailed in the [documentation](https://klarrieu.github.io/RiverREM/).

## Troubleshooting

- No river in DEM extent or inaccurate centerline: Use the [OSM editor](https://www.openstreetmap.org/edit) to create/modify the river centerline(s).

## Issues

Submitting issues, bugs, or suggested feature improvements are encouraged for this repository.

## References

This package was made possible and inspired by the following:

- The [beautiful REMs](https://www.dnr.wa.gov/publications/ger_presentations_dmt_2016_coe.pdf) popularized by [Daniel Coe](https://dancoecarto.com/creating-rems-in-qgis-the-idw-method)
- [DahnJ](https://github.com/DahnJ) for previous implementation of [REMs using xarray](https://github.com/DahnJ/REM-xarray)
- Geoff Boeing's [OSMnx](https://geoffboeing.com/publications/osmnx-complex-street-networks/) Python package leveraging the OSM Overpass API
- The team at [OpenTopography](https://opentopography.org/) for supporting this effort

![dirty_devil_REM_hillshade-color](docs/pics/dirty_devil_REM_hillshade-color.png)