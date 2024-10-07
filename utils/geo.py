from typing import Callable, Union
import sys
import rasterio
import geopandas as gpd
import numpy as np
import warnings
from rasterio.io import MemoryFile
from rasterio.merge import merge

def replace_nan_with_zero(array):
    array[array != array] = 0  # Replace NaN values with zero
    return array

def custom_method_avg(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """Returns the average value pixel.
    cf. https://amanbagrecha.github.io/posts/2022-07-31-merge-rasters-the-modern-way-using-python/index.html
    """
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.nanmean([merged_data, new_data], axis=0, out=merged_data, where=mask)
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")

def merge_tiles(
        tiles:list, 
        dst_path,
        dtype:str = 'float32',
        nodata=None,
        #method:str | Callable ='first',
        method: Union[str, Callable] = 'first',
        ):
    """
    cf. https://amanbagrecha.github.io/posts/2022-07-31-merge-rasters-the-modern-way-using-python/index.html
    """

    file_handler = [rasterio.open(ds) for ds in tiles]
    extents = [ds.bounds for ds in file_handler]
    # Extract individual bounds
    lefts, bottoms, rights, tops = zip(*extents)
    union_extent = (
        min(lefts),     # Left
        min(bottoms),   # Bottom
        max(rights),    # Right
        max(tops)       # Top
    )

    if method == 'average':
        method = custom_method_avg

    # memfile = MemoryFile()
    try:
        merge(sources=file_handler, # list of dataset objects opened in 'r' mode
            bounds=union_extent, # tuple
            nodata=nodata, # float
            dtype=dtype, # dtype
            # resampling=Resampling.nearest,
            method=method, # strategy to combine overlapping rasters
            # dst_path=memfile.name, # str or PathLike to save raster
            dst_path=dst_path,
            # dst_kwds={'blockysize':512, 'blockxsize':512} # Dictionary
          )
    except TypeError:
        merge(datasets=file_handler, # list of dataset objects opened in 'r' mode
            bounds=union_extent, # tuple
            nodata=nodata, # float
            dtype=dtype, # dtype
            # resampling=Resampling.nearest,
            method=method, # strategy to combine overlapping rasters
            # dst_path=memfile.name, # str or PathLike to save raster
            dst_path=dst_path,
            # dst_kwds={'blockysize':512, 'blockxsize':512} # Dictionary
          )

def get_mean_sd_by_band(path, force_compute=True, ignore_zeros=True, subset=1_000):
    '''
    Reads metadata or computes mean and sd of each band of a geotiff.
    If the metadata is not available, mean and standard deviation can be computed via numpy.

    Parameters
    ----------
    path : str
        path to a geotiff file
    ignore_zeros : boolean
        ignore zeros when computing mean and sd via numpy

    Returns
    -------
    means : list
        list of mean values per band
    sds : list
        list of standard deviation values per band
    '''

    np.random.seed(42)
    src = rasterio.open(path)
    print('pouet\n\n')
    means = []
    sds = []
    for band in range(1, src.count+1):
        try:
            tags = src.tags(band)
            if 'STATISTICS_MEAN' in tags and 'STATISTICS_STDDEV' in tags:
                mean = float(tags['STATISTICS_MEAN'])
                sd = float(tags['STATISTICS_STDDEV'])
                means.append(mean)
                sds.append(sd)
            else:
                raise KeyError("Statistics metadata not found.")

        except KeyError:

            arr = src.read(band)
            arr = replace_nan_with_zero(arr)
            ## let subset by default for now
            if subset:
                arr = np.random.choice(arr.flatten(), size=subset) 
            if ignore_zeros:
                mean = np.ma.masked_equal(arr, 0).mean()
                sd = np.ma.masked_equal(arr, 0).std()
            else:
                mean = np.mean(arr)
                sd = np.std(arr)
            means.append(float(mean))
            sds.append(float(sd))

        except Exception as e:
            print(f"Error processing band {band}: {e}")


    src.close()
    return means, sds

