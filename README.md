[![NSF-1948997](https://img.shields.io/badge/NSF-1948997-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1948997) [![NSF-1948994](https://img.shields.io/badge/NSF-1948994-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1948994) [![NSF-1948857](https://img.shields.io/badge/NSF-1948857-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1948857)

[![Conda](https://img.shields.io/conda/v/conda-forge/riverrem?color=success)](https://anaconda.org/conda-forge/riverrem) [![Conda](https://img.shields.io/conda/dn/conda-forge/riverrem?color=success)](https://anaconda.org/conda-forge/riverrem)

# RiverREM

RiverREM is a Python package for automatically generating river relative elevation model (REM) visualizations from nothing but an input digital elevation model (DEM). The package uses the OpenStreetMap API to retrieve river centerline geometries over the DEM extent. Interpolation of river elevations is automatically handled using a sampling scheme based on raster resolution and river sinuosity to create striking high-resolution visualizations without interpolation artefacts straight out of the box and without additional manual steps. The package also contains a helper class for creating DEM raster visualizations. See the [documentation](https://opentopography.github.io/RiverREM/) pages for more details.

For more information on REMs and this project see [this OpenTopography blog post](https://opentopography.org/blog/new-package-automates-river-relative-elevation-model-rem-generation).

![birch_creek_REM](docs/pics/birch_crop.png)

## Installation

Install via conda:

```bash
conda install -c conda-forge riverrem
```

or clone this repo and create a conda environment from the `environment.yml`:

```bash
git clone https://github.com/opentopography/RiverREM.git
cd RiverREM
conda env create -n riverrem --file environment.yml
```

In order to handle dependencies such as GDAL and OSMnx, it is highly recommended to install with `conda` instead of `pip` for ease of use. 

## Usage

1. Get a DEM for the area of interest. Some sources for free topographic data:

   - [OpenTopography](https://opentopography.org/)
   - [USGS](https://apps.nationalmap.gov/downloader/)
   - [Comprehensive list of DEM sources](https://github.com/DahnJ/Awesome-DEM)

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

Options for adjusting colormaps, shading, interpolation parameters, and more are detailed in the [documentation](https://opentopography.github.io/RiverREM/).

![yukon_flats_REM](docs/pics/yukon_crop.png)

## Troubleshooting

- No river in DEM extent or inaccurate centerline: Use the [OSM editor](https://www.openstreetmap.org/edit) to 
  create/modify river centerline(s). Alternatively, a user-provided centerline can be input to override the OSM centerline. See the [documentation](https://opentopography.github.io/RiverREM) for more details.

## Issues

Submitting [issues](https://github.com/OpenTopography/RiverREM/issues), bugs, or suggested feature improvements are highly encouraged for this repository.

## References

This is the OpenTopography fork of https://github.com/klarrieu/RiverREM by Kenneth Larrieu. This package was made possible and inspired by the following:

- The [beautiful REMs](https://www.dnr.wa.gov/publications/ger_presentations_dmt_2016_coe.pdf) popularized by [Daniel Coe](https://dancoecarto.com/creating-rems-in-qgis-the-idw-method)
- [DahnJ](https://github.com/DahnJ)'s implementation of [REMs using xarray](https://github.com/DahnJ/REM-xarray)
- Geoff Boeing's [OSMnx](https://geoffboeing.com/publications/osmnx-complex-street-networks/) Python package leveraging the OSM Overpass API
- The [UNAVCO](https://www.unavco.org/) Student Internship Program
- The team at [OpenTopography](https://opentopography.org/) for supporting this effort under the following U.S. National Science Foundation award numbers: 1948997, 1948994, 1948857.


![neches_REM](docs/pics/neches_REM_view.png)
