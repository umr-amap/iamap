# Installation Guide


## Plug-in installation

As of now, the plugin is not yet published in official QGIS repos, so you have to clone or copy this code into the python plugin directory of QGIS and manualy install.

this is where it probably is located : 

```
# Windows
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins

# Mac
~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins

# Linux
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins
```

Otherwise (for instance if you have several profiles), you can locate it by doing `Settings`>`User Profiles`>`Open active profile folder`.


## Dependencies installation 

To work, the plugin requires QGIS >= 3.34 (LTR). At first usage, a pop up should appear if necessary dependencies are not detected, that gives the option to install them automatically via `pip`.

The main dependencies are `torch`, `rasterio`, `timm`, `geopandas`, `scikit-learn`.
The file `requirement.txt` in the plugin folder gives precise working versions of each dependencies.

However, if the automated installation does not work, here are detailled instructions to install them.

### Installation in separate environment

If you want to control your python environment to avoid conflicts with other projects, you can work in a conda environment, where you can install QGIS.

```
conda create -n iamap
conda activate iamap
conda install qgis
conda install --file requirements.txt
```

this creates a separate qgis installation within the environment.

Do keep in mind that there are possible conflict with geospatial dependencies (GDAL, PROJ etc).


### Manual installation on Windows

Go to OSGeo4W Shell and install the following dependencies.

```
pip install geopandas>=0.14.4
pip install scikit-learn>=1.5.1
pip install psutil>=5.0.0
pip install rasterio>=1.2
pip install rtree>=1
pip install einops>=0.3
pip install fiona>=1.8.19
pip install kornia>=0.6.9
pip install numpy>=1.19.3
pip install pyproj>=3.3
pip install shapely>=1.7.1
pip install timm>=0.4.12
```

### Install on GPU

Upon installation, the plugin should detect if you have a GPU available and install the correct version of pytorch.
If not, do check the [installation instructions of pytorch](https://pytorch.org/get-started/locally/).
