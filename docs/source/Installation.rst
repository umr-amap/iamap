.. _installation-label : installation_1

Installation Guide
===================

Plug-in installation
---------------------

As of now, the plugin is not yet published in official QGIS repos, so you have to clone or copy this code into the python plugin directory of QGIS and manualy install.

this is where it probably is located : 

   ::

        # Windows
        %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins

        # Mac
        ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins

        # Linux
        ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins


Otherwise (for instance if you have several profiles), you can locate it by doing `Settings`>`User Profiles`>`Open active profile folder`.


Dependencies installation 
--------------------------

To work, the plugin requires QGIS >= 3.34 (LTR) and has been tested with python 3.11. At first usage, a pop up should appear if necessary dependencies are not detected, that gives the option to install them automatically via `pip`.

The main dependencies are ``torchgeo``, ``timm``, ``geopandas``, ``scikit-learn``. ``umap-learn`` can be used as well but as optional dependency.
The file ``requirement.txt`` in the plugin folder gives precise working versions of each dependencies.


However, if the automated installation does not work, here are detailled instructions to install them.

Linux
^^^^^^

If you want to control your python environment to avoid conflicts with other projects, you can work in a conda environment, where you can install QGIS.

We will provide a minimum functionning conda environment.

   ::

        conda env create -f environment.yml

.. this creates a separate qgis installation within the environment.

.. Alternatively, the main dependencies are installled via pip and available in the ``requirements_linux.txt`` file. 

Do keep in mind that there are possible conflict with geospatial dependencies (GDAL, PROJ etc).


Windows
^^^^^^^^

Go to OSGeo4W Shell and install the following dependencies.

   ::

       pip install torchgeo == 0.5.2
       pip install geopandas == 0.14.4
       pip install scikit-learn == 1.5.1
       # (optional) pip install umap-learn == 0.5.6

If you have any issue with windows, please provide your QGIS version, your python version an the dependencies currently installed in OSGeo4W.
You can do that by going to OSGeo4W Shell and run the following lines :

    ::

        python --version
        pip show

Then you can fill an issue with these informations on the github.


.. This guide provides instructions on how to install [Your Software].

.. Steps
.. -----

.. 1. Download [Your Software] from [website].
.. 2. Unzip the downloaded file.
.. 3. Open a terminal.
.. 4. Navigate to the directory where you unzipped [Your Software].
.. 5. Run the following command to install:

..    ::

..        python setup.py install

.. Conclusion
.. ----------
