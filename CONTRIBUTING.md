# Contributing

Thanks for taking the time to contribute to IAMAP! Here are some ways you can
help with development if you want to :

## Reporting Bugs

Before creating bug reports, please check existing issues to avoid 
duplicates. 

Please include steps and informations needed to reproduce the behavior
(OS, QGIS version) and if possible a file to have a minimal reproductible 
example.

Usual suspects are conflicts between QGIS and rasterio, consider passing 
the output of

```
import rasterio
rasterio.show_versions()
```

in a python console to have a summary of various GIS dependencies.


## Suggesting Features

Feature requests are welcome! You can check existing issues to see if that
is not something already considered. Otherwise, feel free to open an issue!

### Model support

To keep code and dependencies as light as possible, we prefer models that are
available on well supported hubs like `timm`, `huggingface` of `torch.hub`.
please prefer these implementations if possible. Otherwise, please provide a 
link to implementation code and a way to reliably download pre-trained weights.


## Improving Documentation

Documentation improvements are always welcome! This includes:
- Fixing typos
- Adding examples
- Clarifying confusing sections


## Submitting Code

Feel free to open an issue to discute changes you would like to implement.
Here are a few points to consider:

### Dependencies and environment setup

The dependencies of this plugin are already quite heavy, when implementing
a method, please check that it is not something allready supported by
`rasterio`, `timm` or `sklearn`.

For development, we advise starting in a fresh conda environment, installing
QGIS via conda and starting qgis from the command line to be sure of the
versions and dependencies used by the plugin. For example:

```
conda create -n iamap-dev
conda install qgis
```

Afterwards, the dependencies installer should pop up as with a normal 
installation. Otherwise, you can handle dependencies inside you conda 
environment without messing your existing install.

### Hardware limitations

We aim for the plugin to be usable on a regular laptop, avoid hardcoding
`.cuda()` and prefer `.to(device)` in pytorch for instance.

### Testing and documentation

Please include a way to test the feature you implement with `pytest` 
(see `tests/` files for examples). 
If relevant,update the documentation as well.

---

Thank you for contributing!
