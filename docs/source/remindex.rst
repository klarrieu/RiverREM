.. _rem-module:

REM Module
**********

This module can be used to automatically produce visualizations of river relative elevation
models (REMs) from an input DEM.

River centerlines
-------------------

River centerlines used to create REMs are automatically retrieved from the OpenStreetMap (OSM) API. If a desired river segment is not listed on OSM, a new river centerline can be created/edited `on the OSM site <https://www.openstreetmap.org/edit>`_ (clear the OSM cache folder after using the OSM editor to get the updated centerline). Alternatively, if OSM is up to date but the input DEM has older river topography, a user-provided centerline shapefile can be input, overriding the OSM centerline.

Points sampled along the river centerline(s) are output as a shapefile along with each output REM raster. These points can be viewed to help ensure that the applied centerline is accurate.

Handling large datasets
---------------------------

For very large/high resolution DEMs, interpolation can take a long time. While this module is designed to work well using the default settings on a variety of datasets, interpolation can be sped up by changing the following parameters from default values:

- decreasing :code:`interp_pts` (fewer total centerline sample points)
- decreasing :code:`k` (fewer centerline samples per interpolation)
- increasing :code:`eps` (less accurate interpolation)
- increasing :code:`workers` (if more CPU threads are available)
- increasing :code:`chunk_size` (if more RAM is available)

REMMaker module
----------------------

.. automodule:: riverrem.REMMaker
    :members: clear_osm_cache

.. autoclass:: riverrem.REMMaker.REMMaker
    :members: make_rem, make_rem_viz
