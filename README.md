<p align="center">
<img src="./icons/favicon.svg" width=10% height=10%> 
</p>

# IAMAP

[![Documentation Status](https://app.readthedocs.org/projects/iamap/badge/)](https://iamap.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/umr-amap/iamap/actions/workflows/jobs.yml/badge.svg)](https://github.com/umr-amap/iamap/actions/workflows/jobs.yml)
[![GitLab Mirror](https://img.shields.io/badge/GitLab-Mirror-blue)](https://forge.ird.fr/amap/iamap)



<!-- [Documentation](https://iamap.readthedocs.io/en/latest/) [Gitlab repo (mirror)](https://forge.ird.fr/amap/iamap) -->
<!-- https://app.readthedocs.org/projects/iamap/badge/ -->

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

If this doesn't work, you can find [more detailled instructions in the documentation](https://iamap.readthedocs.io/en/latest/installation.html).

Otherwise, feel free to submit an issue.

## Documentation

Documentation is available [here](https://iamap.readthedocs.io/en/latest/).

## Roadmap

- [ ] Saving and using sklearn models in inference
- [ ] Implementation of [Pangaea benchmark models](https://github.com/VMarsocci/pangaea-bench/tree/main)
- [ ] Handling features of non-ViT-like models
- [ ] Publication on QGIS repo

## Contributing

Feel free to fill an issue on GitHub or submit a PR. More detailled environment setup to come.


## Aknowledgments

The feature extraction algorithm was inspired by the [Geo-SAM](https://github.com/coolzhao/Geo-SAM) plugin. The dependencies installation popup was modified from code by [Deepness](https://github.com/PUTvision/qgis-plugin-deepness) plugin.

<!-- ## Citation -->

