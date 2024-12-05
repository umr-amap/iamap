# Common issues

## Similarity or Random Forest causes QGIS to crash

This crash is due to a bug during `geopandas` reading of a shapefile, only when it has allready read a shapefile. It is probably linked to `fiona` as well. If you have any idea on how to solve this, please do participate in the [corresponding issue](https://github.com/umr-amap/iamap/issues/28).

Meanwhile, if QGIS crashes and your file were not saved, you can still find them in the temp files like `/tmp/iamap_features` for instance.



## QGIS crashes at the start of encoding

This issue should be solved but the solution has not been tested on all OSes. This is an issue with `rtree` during the creation of the dataset that will be used by the plugin.
Indeed, depending on the installation, `rtree` and QGIS may have conflicting `libspatialindex`. Currently, there is several solutions:

### 1. Uninstall `rtree` installed via pip and reinstall via package manager

in the QGIS python console:

(following code has not been tested, based on this [SE answer](https://gis.stackexchange.com/questions/418274/installing-python-module-using-pip-via-python-console-in-qgis-3-22))
```
import subprocess 
import sys

if sys.platform == 'win32':
    subprocess.call([sys.exec_prefix + '/python', "-m", 'pip', 'unininstall', 'rtree'])
else:
    subprocess.call([sys.executable, '-m', 'pip', 'uninstall', 'rtree']) 
```

then in a terminal (for a debian based linux)

```
sudo apt-get install python3-rtree
```

### 2. Uninstall `rtree` installed via pip and reinstall `rtree` at an older version (<1.0.0)

in the QGIS python console:

(following code has not been tested, based on this [SE answer](https://gis.stackexchange.com/questions/418274/installing-python-module-using-pip-via-python-console-in-qgis-3-22))
```
import subprocess 
import sys

if sys.platform == 'win32':
    subprocess.call([sys.exec_prefix + '/python', "-m", 'pip', 'unininstall', 'rtree'])
    subprocess.call([sys.exec_prefix + '/python', "-m", 'pip', 'install', 'rtree==0.9'])
else:
    subprocess.call([sys.executable, '-m', 'pip', 'uninstall', 'rtree']) 
    subprocess.call([sys.executable, "-m", 'pip', 'install', 'rtree==0.9'])
```

### 3. Use a conda environement.

This way, rtree and qgis will automatically share the same libspatialindex.

If you have any idea on how to solve this issue properly, do participate in the [corresponding issue](https://github.com/umr-amap/iamap/issues/13).


## UMAP crashes


If you're using Windows, the use of UMAP may cause the plug-in to crash QGIS when using it.
If this issue happens we recommend to go in the algo.py file of this plug in and comment the line :²

```
import umap
```

Then save the change, reload the plug-in and try again.
