# FAQ

## How does it handle more than three band images with pretrained models ?

Our models are created using the `timm` librairy, which is widely used in deep learning research. [Here](https://timm.fast.ai/models#How-is-timm-able-to-use-pretrained-weights-and-handle-images-that-are-not-3-channel-RGB-images?) is the doc explaining how they handle non-RGB images when loading pre-trained models.

## How can I avoid tiling effects ?

You can create an overlap by selecting a stride smaller than the sampling size of your raster. In the advanced options, you can change how the tiles will be merged afterwards.

## How can I obtain a better resolution ?

This plugin was developped with ViTs in mind as template models. These have spatialy explicit features and divide the image into patches of typially `16x16` or `14x14` pixels. By having a smaller sampling size, you will have better resolution but with less context for the model to work with.
Using a model with smaller patch size will also in the end lead to a better resolution but these models are often heavier.

## Which model should I use ?

We've selected some state of the art models that seem to work well on our usecases so far. If you are short in RAM, prefer using the ``ViT Tiny model``, that is almost ten times smaller than the others (but can provide a less nuanced map).

