# Examples


## Mapping Bamboo forests in Thailand


The goal in this example is to map bamboo forests in Thailand using UAV RGB data.
Here is the image with overlaping training points (the different colors correspond to different bamboo species):

```{image} ./_static/examples/drone_train.png
:alt: Image with training GT points
:class: centered-image
:width: 600px
:align: center
```

And here is the image with overlaping test points (train and test set are separated to avoid spatial auto-correlation):

```{image} ./_static/examples/drone_test.png
:alt: Image with test GT points
:class: centered-image
:width: 600px
:align: center
```

This image is fed through a ViT base DINO encoder (with default encoding parameters) before fitting a random forest (RF) classifier on the obtained features.
We achieve 71% accuracy on this dataset alone. If we fit a RF directly on RGB data, we achieve only 45% accuracy. This shows that the encoder has produced meaningfull features used by the classifier afterwards.



## Mapping Cameroon forest types with Sentinel 2 data


The goal in this example is to map different forest types using multispectral Sentinel 2 data.

We have a raster and points labelled by photo-interpretation:

```{image} ./_static/examples/original_points.png
:alt: Original image with GT points
:class: centered-image
:width: 600px
:align: center
```
One this image, the dark blue points correspond to open forests, green to swap forests and light blue to dense forests.
This dataset was randomly split into 80-20% between a train and a test set to learn a RF classifier.
The image was either fed into a deep learning (DL) backbone (ViT base DINO) or not.

Here is a recap of the random forest accuracy given different pre- and post-processing of the encoder's features.

| Using a DL encoder    | Pre-processing    | Post-processing   | Accuracy  |
| :-----------          | :-----------      | :--------------   |---------: |
| No                    | No                | No                |0.88       |
| Yes                   | No                | No                |0.75       |
| No                    | 3D PCA            | No                |**0.93**   |
| Yes                   | 3D PCA            | No                |0.86       |
| Yes                   | No                | 10D PCA           |0.86       |
| Yes                   | 3D PCA            | 10D PCA           |0.90       |


As we can see, the use of a ViT encoder does not necessarly improve the accuracy of the calssification.
Indeed, the backbone used here was trained on RGB images, working on multispectral data seems to be to far a step here.
Moreover, the different forest types have allready diffrente spectral responses.

By desing, ViT are better than other ML methods to interpret the structure of an image and the relation between patches.
Here, the information being mosty spectral leads to poorer performances after the use of a DL encoder, since the spectral information is drowned between unrelevant structural and tetural information interpreted by the DL backbone. 

Here is the resulting map obtained when infering the RF model without pre-processing:

```{image} ./_static/examples/rf_no_red.png
:alt: Random Forest inference without dimension reduction
:class: centered-image
:width: 600px
:align: center
```

And with pre-processing:

```{image} ./_static/examples/rf_red.png
:alt: Random Forest inference with dimension reduction
:class: centered-image
:width: 600px
:align: center
```


## Mapping Land Cover change in La Reunion island

The objective is to characterize land cover change in La Reunion island based on the analysis of aerial IGN ortho-photography from 1950 and 2022.

Data :
- The collection of historical Orthophotos (1950) is accessible [from the following website](https://geoservices.ign.fr/bdorthohisto)
The historical map is divided into a mosaic of 130 grayscale tiles, i.e. in a 1-band rasters, ranging from 0 to 255 to represent the light intensity.
The resolution of each image is 50cm/pixel.

- The collection of recent Orthophotos (2022) is accessible from the [following website](https://geoservices.ign.fr/bdortho)
The recent map is divided into a mosaic of 130 color tiles, i.e. in RGB (Red, Green, Blue) 3-bands rasters where each band uses pixel value ranging from 0 to 255 to represent the corresponding color intensity. The resolution of each image is 20cm/pixel.

- Preliminary step : A data transformation for the comparison 1950 / 2022 :
The recent images resolution  (2022) is transformed  from 20cm/pixel to 50cm/pixel using QGIS in order to obtain the same resolution as the historical images (1950) . In addition, a conversion from color (RGB or 3-bands) to grayscale (1-band) is applied using QGIS plugin OTB (OTB  > feature extraction > radiometric indices >set band 3 2 1 and choice Bi index (Brightness index) . Thanks to these two steps the comparison between both maps can be done pixel by pixel.

- Method : Preliminary test have been done to diffferenciate a variety of homogeneous patchs (such as forest, urban area, low vegetation) using the Haralick Texture metrics (ie 9 metrics from the R package GLCMTextures) with different sets of  parameters : the results were not satisfying.
  Promising result were obtained with the features obtained from the ViT base DINO encoder
Each image is then fed through a ViT base DINO encoder before fitting a random forest (RF) classifier on the obtained features.


- The Land Cover classes :
A photo-identification is achieved on 30 randomly selected tiles for each dataset. The variability inherent to each class is accounted for by the identification of 10 polygones per landcover class for the training set and another 10 polygones per landcover class for the validation step (The training and validation sets are spatially separated to avoid spatial auto-correlation).
The different landcover classes are the following :
- Agricultural
- Low vegetation
- Forest
- Shade
- Urban


Here is the figure showing the classification results for 1950

```{image} ./_static/examples/classif_1950.png
:alt: Image 1
:class: centered-image
:width: 600px
:align: center
```


Here is the figure showing the classification results for 2022

```{image} ./_static/examples/classif_2022.png
:alt: Image 2
:class: centered-image
:width: 600px
:align: center
```


Here is the figure showing the land cover changes between 1950 and 2022

```{image} ./_static/examples/landcover_change.png
:alt: Image 3
:class: centered-image
:width: 600px
:align: center
```
