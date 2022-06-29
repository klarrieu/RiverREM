REM Module
**********

This module can be used to automatically produce aesthetically pleasing visualizations of river relative elevation
models (REMs) from an input DEM.

River centerlines used to create REMs are automatically retrieved from the OpenStreetMap (OSM) API. If a desired river
segment is not listed on OSM, a new river centerline can be created/edited at: https://www.openstreetmap.org/edit
(clear the ./cache folder after using the OSM editor to get the updated centerline).

For very large/high resolution DEMs, interpolation can take a long time. While this module is designed to work well
using the default settings on a variety of datasets, interpolation can be sped up by changing the following parameters
from default values:

    - decreasing interp_pts
    - increasing k
    - increasing eps
    - increasing workers (if more CPU threads are available)

.. automodule:: riverrem.MakeREM

.. autoclass:: riverrem.MakeREM.REMMaker
    :members: run, make_image_blend
