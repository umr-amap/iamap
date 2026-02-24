---
title: 'IAMAP: Unlocking Deep Learning in QGIS for non-coders and limited computing resources'
tags:
  - Python
  - GIS
  - Remote Sensing
  - Deep Learning
authors:
  - name: Paul Tresson
    orcid: 0000-0002-1275-4673
    corresponding: true
    affiliation: 1
  - name: Pierre Le Coz
    affiliation: "1,2"
  - name: Hadrien Tulet
    affiliation: 1
  - name: Anthony Malkassian
    orcid: 0000-0001-9603-4448
    affiliation: 3
  - name: Maxime Réjou-Méchain
    orcid: 0000-0003-2824-267X
    affiliation: "1,2"
affiliations:
 - name: AMAP, Univ. Montpellier, IRD, CNRS, CIRAD, INRAE, Montpellier, France
   index: 1
 - name: Forest Restoration Research Unit, Department of Biology, Faculty of Science, Chiang Mai University, Chiang Mai, Thailand
   index: 2
 - name: Université de la Réunion, UMR PVBMT, St. Pierre, La Réunion, France
   index: 3
date: 2025
bibliography: paper.bib

---

# Summary 

# Statement of need

The integration of remote sensing data with deep learning approaches is
currently revolutionizing Earth observation sciences, leading to
significant qualitative and quantitative improvements in large-scale
predictions [@zhu2017deep; @yuan2020deep; @yasir2023coupling]. However,
this revolution comes with a number of challenges. First, over the past
decade, most deep learning applications have been highly data-demanding,
requiring extensive manual labeling with typically more than one hundred
thousands labeled points [@safonova2023ten]. In most ecological and
environmental science studies, constructing such a large reference
dataset, through *e.g.*, ground observations or photo-interpretation,
remains a major barrier to the implementation of deep learning
approaches.

Recent developments in deep learning state of the art has seen an evolution
towards the use of large foundation models, trained on large scale datasets 
in an unsupervised way and capable of very good few-shot performances (*i.e.*
without expensive supervised training of the network).
[@ericsson2021well]. The main difference between a pre-trained
self-supervised learning (SSL) model and a pre-trained supervised model
lies in their training objectives: SSL models are not constrained by
predefined labels and are therefore free to explore and encode the
intrinsic structure and diversity of the data, often resulting in more
general and transferable representations. In contrast, supervised models
are explicitly optimized to perform a specific user-defined task, which
can lead to highly specialized representations that may overlook other
meaningful features in the data. As such, SSL foundation models can
perform well even in low-shot or zero-shot tasks, *i.e.* using the model
as is, with few or no training data. Consequently, SSL models are
considered particularly promising for remote sensing tasks, as
demonstrated by recent works and initiatives
[@jakubik2023prithvi; @cong2023satmae; @xiong2024neural; @marsocci2024pangaea].


# State of the field

With the democratization of deep learning, some developers have already
worked on the integration of deep learning models in geographic
information systems such as the open-source and widely used QGIS
software [@QGIS]. However, at the time of writing, these solutions
mostly focus on fine-tuning models or using a model in inference only
[*e.g.* see @aszkowski2023deepness; @zhao2023geosam]. Then, they are
only usable by users with access to high-end computing power, extensive
dataset, on interested in a task for which a specific model was already
trained.

# Software design

The plugin aims at allowing two main tasks: 
1) feeding a tiled raster through a pre-trained deep learning model,
    typically via the *timm* and *huggingface* library.
2) manipulating the produced features with common machine learning tools,
    provided by the *scikit-learn* library.

these tasks are separated into five modules depending on the type of models
and operations the user wants to perform: deep learning feature extraction(see.
Fig. \autoref{fig:backbones}), dimension reduction, clustering, similarity 
computation, or fitting a machine learning model in a supervised way (see.Fig.
 \autoref{fig:functionalities}).

![The five main modules of the IAMAP
plugin. \label{fig:functionalities}](figures/plugin.png)

![A sentinel 2 image of a forested landscape in Thailand (Khao
Banthat Wildlife Sanctuary; Lat 7.53°, Lon 99.82°) processed by
different backbones. The top row represents the first three feature
dimensions output by the models (which may not be the most informative).
The second row shows a 3D PCA of the features mapped to the red, green
and blue channel respectively. The third row shows a projection using a
3D T-SNE.\label{fig:backbones}](figures/backbones.png)

- **The deep learning feature extraction module** allows to feed a raster into 
a ViT-like backbone. The user can chose a raster loaded in QGIS, a pre-trained 
model that is downloaded from timm/huggingface or pass local pre-trained 
weights. The raster is then tiled according to user defined rules and fed into 
the model using a light fork of the *torchgeo library*. Resulting features are 
saved into a geotiff file and can then be used for further analysis.
- **The dimension reduction module** uses *scikit-learn* API to feed a raster 
into a model and collect the resulting raster. All algorithms available in 
the *scikit-learn* `decomposition` and `cluster` modules that have common APIs 
(namely, a `fit()`, a `transform()`, or a `fit_transform()` method) are 
available. The parameters can be fed following a json format.
- **The clustering module** works in a similar manner with algorithms 
available in the *scikit-learn* `cluster` module sharing common APIs (namely, 
a `fit()`, a `predict()`, or a `fit_predict()` method).
- **The similarity module** computes the similarity between a raster and user-
defined template pixels (via a vector layer).
- **The machine learning module** allows the user to fit a supervised machine 
learning model available in the *sciki-learn* `ensemble` and `neighbors` 
modules. Hyper-parameters (*e.g.* validation scheme, labelled data… can be 
defined by the user).

A core constraint of the development is for the plugin to be accessible for a 
non-coding user without a GPU and with limited internet access. As such, all 
design and development is done a CPU only laptop to assess usability with no 
GPU. Various optimizations are available, such as quantization, scheduled 
pauses and progress save on disk. We used code from the 
[deepness plugin](https://github.com/PUTvision/qgis-plugin-deepness/) to 
provide automated dependencies installation dialog.

# Research impact statement

The plugin is used internally and has been presented in several webinars with 
strong positive feedback from potential users. It has also be used for research
to be published soon by Malkassian et. al. (A preview of the results 
[is available in the documentation](https://iamap.readthedocs.io/en/latest/examples.html#mapping-land-cover-change-in-la-reunion-island)).

# AI usage disclosure

AI was only used to help generate small functions and "boilerplate" code on 
common tasks that could be immediatly be tested. 

# Availability

Development of the plugin is 
[open sourced on GitHub](https://github.com/umr-amap/iamap).
Documentation is available [on readthedocs](https://iamap.readthedocs.io/). 
The plugin is developed in continuous
integration. We plan to publish the plugin on official QGIS repository
to further ease the installation process.

# Acknowledgments

The authors would like to thank all people who have tested this software
during development and have provided meaningful feedback.

# Conflict of interest

The authors declare no conflict of interest.

# References
