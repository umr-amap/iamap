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


<!-- ## Mapping Land Cover in La Réunion -->
