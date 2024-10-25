More
================


Common Issues
---------------

UMAP crashes
^^^^^^^^^^^^^^^

Because of these issues, UMAP is disabled by default for now.

If you're using Windows, the use of UMAP may cause the plug-in to crash QGIS when using it.
If this issue happens we recommend to go in the reduction.py file of this plug in and comment the line :

::

    import umap

Then save the change, reload the plug-in and try again.

Similarity or Random Forest causes QGIS to crash
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This crash is due to a bug during `geopandas` reading of a shapefile, only when it has allready read a shapefile. It is probably linked to `fiona` as well. If you have any idea on how to solve this, please do participate in the [corresponding issue](https://github.com/umr-amap/iamap/issues/28).

In the meanwhile, if QGIS crashes and your file were not saved, you can still find them in the temp files like `/tmp/iamap_features` for instance.


FAQ
---------------

How does it handle more than three band images with pretrained models ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our models are created using the `timm` librairy, which is widely used in deep learning research. [Here](https://timm.fast.ai/models#How-is-timm-able-to-use-pretrained-weights-and-handle-images-that-are-not-3-channel-RGB-images?) is the doc explaining how they handle non-RGB images when loading pre-trained models.

How can I avoid tiling effects ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create an overlap by selecting a stride smaller than the sampling size of your raster. In the advanced options, you can change how the tiles will be merged afterwards.

How can I obtain a better resolution ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This plugin was developped with ViTs in mind as template models. These have spatialy explicit features and divide the image into patches of typially `16x16` or `14x14` pixels. By having a smaller sampling size, you will have better resolution but with less context for the model to work with.

Which model should I use ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We've selected some state of the art models that seem to work well on our usecases so far. If you are short in RAM, prefer using the ``ViT Tiny model``, that is almost ten times smaller than the others (but can provide a less nuanced map).


.. Citation
.. ---------

FAQ
---------------

How does it handle more than three band images with pretrained models ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our models are created using the `timm` librairy, which is widely used in deep learning research. [Here](https://timm.fast.ai/models#How-is-timm-able-to-use-pretrained-weights-and-handle-images-that-are-not-3-channel-RGB-images?) is the doc explaining how they handle non-RGB images when loading pre-trained models.

How can I avoid tiling effects ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create an overlap by selecting a stride smaller than the sampling size of your raster. In the advanced options, you can change how the tiles will be merged afterwards.

How can I obtain a better resolution ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This plugin was developped with ViTs in mind as template models. These have spatialy explicit features and divide the image into patches of typially `16x16` or `14x14` pixels. By having a smaller sampling size, you will have better resolution but with less context for the model to work with.

Which model should I use ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We've selected some state of the art models that seem to work well on our usecases so far. If you are short in RAM, prefer using the ``ViT Tiny model``, that is almost ten times smaller than the others (but can provide a less nuanced map).


.. Citation
.. ---------



Future work
------------

In the short term, we aim to extend the possible reduction and clustering algorithm available (for instance using HDBSCAN as a clustering algorithm).



Acknowledgements
-----------------

 This repo benefits from `Geo-SAM <https://github.com/coolzhao/Geo-SAM>`_ and  `TorchGeo <https://github.com/microsoft/torchgeo>`_. Thanks for their wonderful work.
