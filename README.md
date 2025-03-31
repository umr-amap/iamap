<p align="center">
<img src="./icons/favicon.svg" width=10% height=10%> 
</p>

# IAMAP

[Documentation](https://iamap.readthedocs.io/en/latest/) [Gitlab repo (mirror)](https://forge.ird.fr/amap/iamap)

## Rationale

Deep learning is a powerful tool for image analysis. However several limits exist to it's full democratization and it's extension to remote sensing. Most notably, training of deep learning models requires lots of labelised data and computational power. In a lot of cases, labelised data is not easy to acquire and machines with high computational power are expensive.

However, new foundation models trained with self-supervised methods (such as DINO, DINOv2, MAE, SAM) aim to be as generalist as possible and produce features of high quality, even before being trained on a specific downstream task.

With this plugin, we aim to provide an easy to use framework to use these models in an unsupervised way on raster images. The features produced by the models can often already be used to weed out a big part of analysis work using more conventional and lighter techniques than full deep learning. Therefore, one of our goals is that this plugin can be used without any GPU.

## Installation

### Plugin installation

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

### Dependencies

At first usage, a pop up should appear if necessary dependencies are not detected, that gives the option to install them automatically via `pip`.

You can find more detailled instructions in the documentation.

For now, if you want to use a GPU you should install torch manualy following the instructions on https://pytorch.org/get-started/locally/

Autommated GPU dependencies installation is in the works, you can try the `gpu-support` branch on this repo.


## Documentation

Documentation is available [here](https://iamap.readthedocs.io/en/latest/).

## FAQ

### How does it handle more than three band images with pretrained models ?

Our models are created using the `timm` librairy, which is widely used in deep learning research. [Here](https://timm.fast.ai/models#How-is-timm-able-to-use-pretrained-weights-and-handle-images-that-are-not-3-channel-RGB-images?) is the doc explaining how they handle non-RGB images when loading pre-trained models.

### How can I avoid tiling effects ?

You can create an overlap by selecting a stride smaller than the sampling size of your raster. In the advanced options, you can change how the tiles will be merged afterwards.

### How can I obtain a better resolution ?

This plugin was developped with ViTs in mind as template models. These have spatialy explicit features and divide the image into patches of typially `16x16` or `14x14` pixels. By having a smaller sampling size, you will have better resolution but with less context for the model to work with.

## Contributing

Feel free to fill an issue on GitHub or submit a PR. More detailled environment setup to come.


## Aknowledgments

The feature extraction algorithm was inspired by the [Geo-SAM](https://github.com/coolzhao/Geo-SAM) plugin. The dependencies installation popup was modified from code by [Deepness](https://github.com/PUTvision/qgis-plugin-deepness) plugin.

<!-- ## Citation -->

