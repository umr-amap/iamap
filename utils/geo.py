import os
from typing import Callable, Union
import rasterio
import rasterio.errors
import geopandas as gpd
import numpy as np
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

def get_extents(raster_files):
    extents = []
    for file in raster_files:
        with rasterio.open(file, 'r') as src:
            bounds = src.bounds
            extents.append(bounds)
    return extents

def merge_tiles(
    tiles: list,
    dst_path,
    dtype: str = "float32",
    nodata=None,
    # method:str | Callable ='first',
    method: Union[str, Callable] = "first",
):
    """
    cf. https://amanbagrecha.github.io/posts/2022-07-31-merge-rasters-the-modern-way-using-python/index.html
    """

    file_handler = [rasterio.open(ds) for ds in tiles]
    extents = [ds.bounds for ds in file_handler]
    # Extract individual bounds
    lefts, bottoms, rights, tops = zip(*extents)
    union_extent = (
        min(lefts),  # Left
        min(bottoms),  # Bottom
        max(rights),  # Right
        max(tops),  # Top
    )

    if method == "average":
        method = custom_method_avg

    try:
        merge(
            sources=file_handler,  # list of dataset objects opened in 'r' mode
            bounds=union_extent,  # tuple
            nodata=nodata,  # float
            dtype=dtype,  # dtype
            method=method,  # strategy to combine overlapping rasters
            dst_path=dst_path,
        )
    ## different rasterio versions take different keyword args
    except TypeError:
        merge(
            datasets=file_handler,  # list of dataset objects opened in 'r' mode
            bounds=union_extent,  # tuple
            nodata=nodata,  # float
            dtype=dtype,  # dtype
            method=method,  # strategy to combine overlapping rasters
            dst_path=dst_path,
        )

    # close datasets
    for ds in file_handler:
        ds.close()


def merge_two_tiles(raster1, raster2, temp_dst_path, nodata,dtype, method):
    with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2:
        extents = [src1.bounds, src2.bounds]
        lefts, bottoms, rights, tops = zip(*extents)
        union_extent = (
            min(lefts),  # Left
            min(bottoms),  # Bottom
            max(rights),  # Right
            max(tops),  # Top
        )

        if method == "average":
            method = custom_method_avg

        try:
            merge(
                sources=[src1, src2],
                bounds=union_extent,
                nodata=nodata,
                dtype=dtype,
                method=method,
                dst_path=temp_dst_path,
            )
    ## different rasterio versions take different keyword args
    except TypeError:
            merge(
                datasets=[src1, src2],
                bounds=union_extent,
                nodata=nodata,
                dtype=dtype,
                method=method,
                dst_path=temp_dst_path,
            )
    return temp_dst_path



def get_mean_sd_by_band(path, force_compute=True, ignore_zeros=True, subset=1_000):
    """
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
    """

    np.random.seed(42)
    src = rasterio.open(path)
    means = []
    sds = []
    for band in range(1, src.count + 1):
        try:
            tags = src.tags(band)
            if "STATISTICS_MEAN" in tags and "STATISTICS_STDDEV" in tags:
                mean = float(tags["STATISTICS_MEAN"])
                sd = float(tags["STATISTICS_STDDEV"])
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


def get_random_samples_in_gdf(gdf, num_samples, seed=42):
    ## if input is not point based, we take random samples in it
    if not all(gdf.geometry.geom_type == "Point"):
        ## Calculate the area of each polygon
        ## to determine the number of samples for each category
        gdf["iamap_area"] = gdf.geometry.area
        total_area = gdf["iamap_area"].sum()
        gdf["iamap_sample_size"] = (
            gdf["iamap_area"] / total_area * num_samples
        ).astype(int)

        series = []
        # Sample polygons proportional to their size
        ## see https://geopandas.org/en/stable/docs/user_guide/sampling.html#Variable-number-of-points
        for idx, row in gdf.iterrows():
            sampled_points = (
                gdf.loc[gdf.index == idx]
                .sample_points(size=row["iamap_sample_size"], rng=seed)
                .explode(ignore_index=True)
            )

            for point in sampled_points:
                new_row = row.copy()
                new_row.geometry = point
                series.append(new_row)

        point_gdf = gpd.GeoDataFrame(series, crs=gdf.crs)
        point_gdf.index = [i for i in range(len(point_gdf))]
        del point_gdf["iamap_area"]
        del point_gdf["iamap_sample_size"]

        return point_gdf

    return gdf


def get_unique_col_name(gdf, base_name="fold"):
    column_name = base_name
    counter = 1

    # Check if the column already exists, if yes, keep updating the name
    while column_name in gdf.columns:
        column_name = f"{base_name}{counter}"
        counter += 1

    return column_name


def validate_geotiff(output_file, expected_output_size=4428850, expected_wh=(60,24)):
    """
    tests geotiff validity by opening with rasterio,
    checking if the file weights as expected and has the correct width and height.
    Additionaly, it is checked if there is more than one value in the raster.
    """

    expected_size_min = .8*expected_output_size
    expected_size_max = 1.2*expected_output_size
    # 1. Check if the output file is a valid GeoTIFF
    try:
        with rasterio.open(output_file) as src:
            assert src.meta['driver'] == 'GTiff', "File is not a valid GeoTIFF."
            width = src.width
            height = src.height
            # 2. Read the data and check width/height
            assert width == expected_wh[0], f"Expected width {expected_wh[0]}, got {width}."
            assert height == expected_wh[1], f"Expected height {expected_wh[1]}, got {height}."
            # 3. Read the data and check for unique values
            data = src.read(1)  # Read the first band
            unique_values = np.unique(data)

            assert len(unique_values) > 1, "The GeoTIFF contains only one unique value."

    except rasterio.errors.RasterioIOError:
        print("The file could not be opened as a GeoTIFF, indicating it is invalid.")
        assert False

    # 4. Check if the file size is within the expected range
    file_size = os.path.getsize(output_file)
    assert expected_size_min <= file_size <= expected_size_max, (
        f"File size {file_size} is outside the expected range."
    )
    return

if __name__ == "__main__":
    gdf = gpd.read_file("assets/ml_poly.shp")
    print(gdf)
    gdf = get_random_samples_in_gdf(gdf, 100)
    print(gdf)
    print(len(gdf))
