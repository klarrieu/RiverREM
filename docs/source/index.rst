Welcome to RiverREM's documentation!
====================================

Source code: `https://github.com/OpenTopography/RiverREM <https://github.com/OpenTopography/RiverREM>`_

OT blog post:  `link <https://opentopography.org/blog/new-package-automates-river-relative-elevation-model-rem-generation>`_

.. image:: ../_images/neches_topo_crop.jpg



RiverREM is a Python package for automatically generating river relative elevation model (REM) visualizations from nothing but an input digital elevation model (DEM). The package uses the OpenStreetMap API to retrieve river centerline geometries over the DEM extent. Interpolation of river elevations is automatically handled using a sampling scheme based on raster resolution and river sinuosity to create striking high-resolution visualizations out of the box and without the need for additional manual steps. The package also contains a helper class for creating DEM raster visualizations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   remindex
   vizindex



Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
