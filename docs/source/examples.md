# Examples

## Mapping Cameroon forest types with Sentinel 2 data


The goal in this example is to map different forest types using multispectral Sentinel 2 data.

We have a raster and ... points that have been labelled by photo-interpretation.

```{image} ./_static/examples/original_points.png
:alt: Original image with GT points
:class: centered-image
:width: 600px
:align: center
```

Here is a recap of the random forest accuracy given different pre- and post-processing of the encoder's features.

| Using a DL encoder    | Pre-processing    | Post-processing   | Accuracy  |
| :-----------          | :-----------      | :--------------   |---------: |
| No                    | No                | No                |           |
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


## Mapping Bamboo forests in Thailand



## Mapping Land Cover in La Réunion
