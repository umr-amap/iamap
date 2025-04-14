# Tools

This section describes the different tools available in the plugin.

<!-- <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;"> -->
<!--     <iframe src="https://www.youtube.com/watch?v=dQw4w9WgXcQ" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe> -->
<!-- </div> -->

---------------------------------------
## Encoder

```{image} ../../icons/encoder.svg
:alt: Encoder icon
:class: centered-image
:width: 100px
:align: center
```

This tool enables the encoding of an image with a deep learning backbone.
Projecting an image through a deep learning backbone can indeed help bypass color, shadows or texture artefacts that make an object hard to detect otherwise.
We have coded this plugin with Vision Transformers (ViTs) in mind as backbones because they have become the state of the art in computer vision since 2021.
Modern deep learning backbones are pretrained in a Self-Supervised maner and can provide meaningfull descriptors of an image without further training.
Here, we rely on the [timm](https://huggingface.co/timm) library to download and use pre-trained models. This library is widely used in the deep learning community and regularly updated.
After being fed to the model, we save the outputed features produced by the model. Hopefully, these features can represent the image in a new feature space that is more informative and discriminant to help with further mapping.
Generally with ViTs, the resulting spatial resolution is coarser than the input resolution (_e.g._ for a ViT Base, 16x16 pixels become a single pixel).

### Encoding process

GIS raster images are usually too big to be fed in a deep learning backbone. Therefore, we tile the input image and each tile is fed to the backbone.
There is two major parameters you can set : `size` and `stride`. The original image will be sampled in tiles of `size x size` pixels with the given stride.
Then, if the stride is smaller than the size, there will be an overlap between tiles. If stride is equal to the size, it will be a perfect grid. If the stride is bigger than the size, if will probably not work properly !

Do keep in mind that before passsing through the model, the sampled tile will be resized according to the expected input size from the model.

During encoding, each tile is saved on disk and the tiles are merged regularly to save space. We chose to save on disk to be easier on ram an leave the option to stop the process and restart later.
By default, features are saved in a temporary directory but you can change the target location.

Encoding sessions are identified with a [md5 hash](https://en.wikipedia.org/wiki/MD5) so a set of input parameters is unique and can be recognised. This allows to easily start again where you left of.
The parameters corresponding to a sha are saved in the `parameters.csv` files in the target directory.

The backend for handling datasets and dataloader is a fork from [torchgeo](https://torchgeo.readthedocs.io/).

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
    <iframe src="./_static/encoder.webm" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>



<!-- ### Backbone choice -->

<!-- We have pre-selected a couple of backbones that are often used in computer vision. -->
<!-- However, if you input the name of another timm or huggingface backbone in the field below, this will be the used one. -->

<!-- ### Using a custom backbone -->

<!-- If you have a pretrained backbone, you can give a path to the weights in the corresponding field and this will be the backbone used for encoding. -->

### Parameters

- **Input raster layer or image file path:**
The raster you want to feed to a deep learning encoder. This can either be a layer loaded in QGIS or the path to a file.

- **Selected Bands:**
Bands that will be fed to the encoder. Selected none feeds all the bands as is into the encoder. If you select a number of bands different that the one of the backbone you are using, the pretrained model will be changed accordingly.

- **Processing extent:**
Defaults to the entire image. Otherwise, you can set a smaller processing extent, either by calculating from a layer, from the current map canvas or by drawing the extent on the map.

- **Sampling size:**
The input raster will be sampled in squares of this sampling size (in pixels). This size can differ from the input size of the chosen deep learning encoder, sampled tiles will be resized before entering the encoder.

- **Stride:**
Step size between two sampled tiles. 
If the stride is equall to the sampling size, the raster will be sampled allong a grid.
If the stride is smaller than the sampling size, there will be an overlap between neighboring tiles.
If the stride is larger, this will likely cause an error.

- **Use GPU if CUDA is available:**
If the plugin recognises a GPU, it will be used for computing.

- **Pre-selected backbones:**
A selection of backbones.

- **Enter an architecture name if you want to test another backbone**:
You can use other backbones available on [huggingface](https://huggingface.co/timm). These however may not work properly depending on their architecture.
Most ViT like backbones should work.

- **Batch size:**
How many tiles are fed into the network at once. This only takes effect if a GPU is available.

- **Output directory:**
Where resulting rasters will be saved. A subdirectory identified by a md5 hash will correspond to a given encoding session.


### Advanced parameters

- **Pretrained checkpoint:**
If you have a pretrained model available on disk, you can use this one rather than pre-trained weights available on the web.

- **CUDA Device ID:**
Enter CUDA device ID to choose on which GPU computations are done if you have several.

- **Remove temporary files after encoding:**
If selected, all temporary tiles will be removed at the end of encoding.

- **Merge method at the end of inference:**
Choose how tiles will be merged to reconstruct a full raster.
For more informations, see [rasterio documentation](https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html#rasterio.merge.merge).
Only the `average` method is custom and will average several overlapping tiles to obtain the values of the final pixels.

- **Frequency at which temporary files should be cleaned up:**
Every n batch, temporary tiles will be merged together and deleted.

- **Number of workers for dataloader:**
How many threads will be used by the dataloader to feed the tiles into the encoder. This defaults to all available workers.
You can chose less to ease the workload on your CPU.

- **Schedule pauses between batches:**
If a number is inputed, their will be a pause between each batch. This allows to pass the inference in background if other computations have to be made at the same time.

- **Target CRS:**
CRS into which the resulting raster should be projected.

- **Target resolution:**
target resolution in meters.

- **Compress final result to uint16 and JP2 to save space:**
If selected, the final features raster will be converted to uint16 rather than float32 (*i.e.* two times lighter) and compressed to JP2 rather than geotiff to save space.

- **Pass parameters as JSON file:**
In the output directory, the parameters corresponding to an encoding session, you can find a JSON file summarizing the input parameters used during encoding.
You can pass this JSON file here to overide all previous parameters. This can be usefull if you want to resume an encoding session.



---------------------------------------
## Dimension reduction 

```{image} ../../icons/proj.svg
:alt: Reduction icon
:class: centered-image
:width: 100px
:align: center
```

The features produced by a deep learning encoder are often of high dimensionality (_e.g._ 768 dimensions for a ViT base).
However, it can be cumbersome to deal with all these features and this high dimensionality feature space, especially when a majority are not really informative.
Therefore, it is possible to reduce the dimensions of a raster using a variety of algorithms.
We chose to rely on [scikit-learn](https://scikit-learn.org/) to provide the algorithms.
All algorighms available in the [decomposition](https://scikit-learn.org/stable/api/sklearn.decomposition.html), [manifold](https://scikit-learn.org/stable/api/sklearn.manifold.html) and the [cluster](https://scikit-learn.org/stable/api/sklearn.cluster.html) module that share a common API can be used.

Different algorithms have different arguments that can be passed. You can provide these as a json string in the corresponding field.

> Not all of the algorithms have been tested and some may be heavy on computing or need particular input types.

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
    <iframe src="./_static/proj.webm" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

---------------------------------------
## Clustering 

```{image} ../../icons/cluster.svg
:alt: Clustering icon
:class: centered-image
:width: 100px
:align: center
```

Features or reduced features can be clustered (_i.e._ unsupervised classification) using algorithms form the scikit-learn [cluster](https://scikit-learn.org/stable/api/sklearn.cluster.html) module that share a common API.

Different algorithms have different arguments that can be passed. You can provide these as a json string in the corresponding field.

> Not all of the algorithms have been tested and some may be heavy on computing or need particular input types.

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
    <iframe src="./_static/cluster.webm" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

---------------------------------------
## Similarity

```{image} ../../icons/sim.svg
:alt: Similarity icon
:class: centered-image
:width: 100px
:align: center
```

A good way to compare two points in a high-dimension setting is through [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
This measure will be equall to one for vectors having the same coordinates and 0 for orthogonal vectors. Thus, the closer to one the cosine similarity is, the more similar two points should be.

Here, additionnaly to an input raster, you have to provide a shapefile (or any format that will be read by geopandas) that will serve to prodive reference point(s).

> If the geometry of your input is not points, it will automatically be sampled as points. You can check the sampling rate in the options.

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
    <iframe src="./_static/sim.webm" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

---------------------------------------
## Machine Learning Algorithms

```{image} ../../icons/forest.svg
:alt: Machine learning icon
:class: centered-image
:width: 100px
:align: center
```

If the features you have seem informative, you can fit a Machine Learning model on it by providing ground truth points.
Thus, you have to provide an input shapfile (or any format that will be read by geopandas) and the column corresponding to the ground truth values.
Based on the algorithm you choose, these values will be interpreted as integers (classification) or floats (regression).
All models provided by scikit-learn [ensemble](https://scikit-learn.org/stable/api/sklearn.ensemble.html) (_e.g._ Random Forests, Gradient Boosting) and [neighbors](https://scikit-learn.org/stable/api/sklearn.neighbors.html)(_e.g._ KNN) module  that share a common API are available.

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
    <iframe src="./_static/ml.webm" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

### Training and testing

It is good practice to train and test a ML model on separate datasets. If you do not provide a test dataset, you have the option to perform a (cross-validation)[https://en.wikipedia.org/wiki/Cross-validation_(statistics)].
Then, you can either define the column that defined the separation between the different folds of your dataset or go for an automatic split.
If you choose the automatic split, ye perform a random sampling and each points are randomly attributed to a fold. Be carefull that this may not be the best way to validate a model in your case !

> If the geometry of your input is not points, it will automatically be sampled as points. You can check the sampling rate in the options.

> Not all of the algorithms have been tested and some may be heavy on computing or need particular input types.
