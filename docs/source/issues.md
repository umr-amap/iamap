# Common issues

<!-- ### UMAP crashes -->

<!-- Because of these issues, UMAP is disabled by default for now. -->

<!-- If you're using Windows, the use of UMAP may cause the plug-in to crash QGIS when using it. -->
<!-- If this issue happens we recommend to go in the reduction.py file of this plug in and comment the line : -->

<!-- ``` -->
<!-- import umap -->
<!-- ``` -->

<!-- Then save the change, reload the plug-in and try again. -->

## Similarity or Random Forest causes QGIS to crash

This crash is due to a bug during `geopandas` reading of a shapefile, only when it has allready read a shapefile. It is probably linked to `fiona` as well. If you have any idea on how to solve this, please do participate in the [corresponding issue](https://github.com/umr-amap/iamap/issues/28).

In the meanwhile, if QGIS crashes and your file were not saved, you can still find them in the temp files like `/tmp/iamap_features` for instance.
