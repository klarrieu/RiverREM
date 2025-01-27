.. _installation:

Installation
***************

Installation method 1 (recommended): New environment with conda/mamba
--------------------------------------------------------------------------------------

Make a new Python environment with RiverREM installed:

.. code-block:: bash
 
   conda create -n rem_env riverrem


.. note::
	:code:`mamba` is recommended over :code:`conda` as it is able to solve environment dependencies quickly and robustly. If you are using :code:`mamba`, replace :code:`conda` with :code:`mamba` in the above command.


The environment can then be activated:

.. code-block:: bash

   conda activate rem_env

Installation method 2: Existing environment
-----------------------------------------------------

Install via conda/mamba

.. code-block:: bash

   conda install -c conda-forge riverrem


Installation method 3: Repository clone
-----------------------------------------------------

Clone the GitHub repository and create a conda environment from the :code:`environment.yml`


.. code-block:: bash

   git clone https://github.com/opentopography/RiverREM.git
   cd RiverREM
   conda env create -n rem_env --file environment.yml


Confirm installation works
--------------------------------

After installing, you should be able to open a Python interpreter and import without errors:

.. code-block:: python

   from riverrem.REMMaker import REMMaker


Continue to the Quickstart to start making REMs.

