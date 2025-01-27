Quickstart
**********

First, ensure you have gone through :ref:`Installation <installation>` of RiverREM.

Input DEM
------------

After installing RiverREM, you'll need a digital elevation model (DEM) for the area of interest. Some sources for free topographic data:

   - `OpenTopography <https://opentopography.org/>`_
   - `USGS <https://apps.nationalmap.gov/downloader/>`_
   - `Comprehensive list of DEM sources <https://github.com/DahnJ/Awesome-DEM>`_

To read the DEM into a :code:`REMMaker` object, we run the following code:

.. code-block:: python

   from riverrem.REMMaker import REMMaker
   # provide the DEM file path and desired output directory
   rem_maker = REMMaker(dem='/path/to/dem.tif')

The :code:`REMMaker` accepts other arguments to specify things like custom river centerlines and output locations. See the :ref:`REM Module documentation <rem-module>` for more.



Making an REM
-------------------

Next, we can create an REM from the input DEM. If a centerline shapefile was not provided, RiverREM will automatically identify the largest river in the DEM domain using OpenStreetMaps.

.. code-block:: python

   # create an REM
   rem_maker.make_rem()


Running this produces an output REM, containing the detrended elevation values (elevation heights relative to the river). The raw REM can be visualized using a GIS program such as QGIS and processed/analyzed manually if desired.


Making a colorful REM visualization
----------------------------------------

Lastly, we apply shading and color mapping to the REM to make a colorful visualization:

.. code-block:: python

   # create an REM visualization with the given colormap
   rem_maker.make_rem_viz(cmap='topo')


Visualization options
-----------------------------

By default, the colormap is applied to a logarithmically-transformed REM elevation values. This is used to emphasize elevation differences closer to the river and make the river geomorphology more prominent.

Any named colormap from `matplotlib <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_, `seaborn <https://seaborn.pydata.org/tutorial/color_palettes.html>`_, or `cmocean <https://matplotlib.org/cmocean/>`_ can be specified for the :code:`cmap` parameter. Alternatively, a custom colormap can be specified in the form of a function that takes numeric inputs in the 0-255 range and outputs (R, G, B) tuples in the 0-1 range.

The color relief REM is also blended with a hillshade raster to give a 3D shadow effect. Vertical exaggeration can be adjusted with the :code:`z` parameter. The amount of hillshade/color blending can be specified by the :code:`blend_percent` parameter.

For a fill list of options and descriptions of methods, see :ref:`REM Module documentation <rem-module>`.

