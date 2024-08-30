More
================


Commons Issue
---------------
If you're using Windows, the use of UMAP may cause the plug-in to crash QGIS when using it.
If this issue happens we recommend to go in the reduction.py file of this plug in and comment the line :
::
    import umap

Then save the change, reload the plug-in and try again.
Citation
---------




Future work
------------



Acknowledgements
-----------------

 This repo benefits from `Geo-SAM <https://github.com/coolzhao/Geo-SAM>`_ and  `TorchGeo <https://github.com/microsoft/torchgeo>`_. Thanks for their wonderful work.