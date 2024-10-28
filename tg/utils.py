# modified from torchgeo code

"""Common dataset utilities."""

# https://github.com/sphinx-doc/sphinx/issues/11327
from __future__ import annotations


import math

import collections
import sys

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Union, overload

import numpy as np
import rasterio
import torch
from torch import Tensor
from torchvision.datasets.utils import check_integrity, download_url


__all__ = (
    "check_integrity",
    "download_url",
    "BoundingBox",
    "disambiguate_timestamp",
    "concat_samples",
    "merge_samples",
)


@dataclass(frozen=True)
class BoundingBox:
    """Data class for indexing spatiotemporal data."""

    #: western boundary
    minx: float
    #: eastern boundary
    maxx: float
    #: southern boundary
    miny: float
    #: northern boundary
    maxy: float
    #: earliest boundary
    mint: float
    #: latest boundary
    maxt: float

    def __post_init__(self) -> None:
        """Validate the arguments passed to :meth:`__init__`.

        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)

        .. versionadded:: 0.2
        """
        if self.minx > self.maxx:
            raise ValueError(
                f"Bounding box is invalid: 'minx={self.minx}' > 'maxx={self.maxx}'"
            )
        if self.miny > self.maxy:
            raise ValueError(
                f"Bounding box is invalid: 'miny={self.miny}' > 'maxy={self.maxy}'"
            )
        if self.mint > self.maxt:
            raise ValueError(
                f"Bounding box is invalid: 'mint={self.mint}' > 'maxt={self.maxt}'"
            )

    # https://github.com/PyCQA/pydocstyle/issues/525
    @overload
    def __getitem__(self, key: int) -> float:  # noqa: D105
        pass

    @overload
    def __getitem__(self, key: slice) -> list[float]:  # noqa: D105
        pass

    def __getitem__(self, key: int | slice) -> float | list[float]:
        """Index the (minx, maxx, miny, maxy, mint, maxt) tuple.

        Args:
            key: integer or slice object

        Returns:
            the value(s) at that index

        Raises:
            IndexError: if key is out of bounds
        """
        return [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt][key]

    def __iter__(self) -> Iterator[float]:
        """Container iterator.

        Returns:
            iterator object that iterates over all objects in the container
        """
        yield from [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt]

    def __contains__(self, other: BoundingBox) -> bool:
        """Whether or not other is within the bounds of this bounding box.

        Args:
            other: another bounding box

        Returns:
            True if other is within this bounding box, else False

        .. versionadded:: 0.2
        """
        return (
            (self.minx <= other.minx <= self.maxx)
            and (self.minx <= other.maxx <= self.maxx)
            and (self.miny <= other.miny <= self.maxy)
            and (self.miny <= other.maxy <= self.maxy)
            and (self.mint <= other.mint <= self.maxt)
            and (self.mint <= other.maxt <= self.maxt)
        )

    def __or__(self, other: BoundingBox) -> BoundingBox:
        """The union operator.

        Args:
            other: another bounding box

        Returns:
            the minimum bounding box that contains both self and other

        .. versionadded:: 0.2
        """
        return BoundingBox(
            min(self.minx, other.minx),
            max(self.maxx, other.maxx),
            min(self.miny, other.miny),
            max(self.maxy, other.maxy),
            min(self.mint, other.mint),
            max(self.maxt, other.maxt),
        )

    def __and__(self, other: BoundingBox) -> BoundingBox:
        """The intersection operator.

        Args:
            other: another bounding box

        Returns:
            the intersection of self and other

        Raises:
            ValueError: if self and other do not intersect

        .. versionadded:: 0.2
        """
        try:
            return BoundingBox(
                max(self.minx, other.minx),
                min(self.maxx, other.maxx),
                max(self.miny, other.miny),
                min(self.maxy, other.maxy),
                max(self.mint, other.mint),
                min(self.maxt, other.maxt),
            )
        except ValueError:
            raise ValueError(f"Bounding boxes {self} and {other} do not overlap")

    @property
    def area(self) -> float:
        """Area of bounding box.

        Area is defined as spatial area.

        Returns:
            area

        .. versionadded:: 0.3
        """
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    @property
    def volume(self) -> float:
        """Volume of bounding box.

        Volume is defined as spatial area times temporal range.

        Returns:
            volume

        .. versionadded:: 0.3
        """
        return self.area * (self.maxt - self.mint)

    def intersects(self, other: BoundingBox) -> bool:
        """Whether or not two bounding boxes intersect.

        Args:
            other: another bounding box

        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
            and self.mint <= other.maxt
            and self.maxt >= other.mint
        )

    def split(
        self, proportion: float, horizontal: bool = True
    ) -> tuple[BoundingBox, BoundingBox]:
        """Split BoundingBox in two.

        Args:
            proportion: split proportion in range (0,1)
            horizontal: whether the split is horizontal or vertical

        Returns:
            A tuple with the resulting BoundingBoxes

        .. versionadded:: 0.5
        """
        if not (0.0 < proportion < 1.0):
            raise ValueError("Input proportion must be between 0 and 1.")

        if horizontal:
            w = self.maxx - self.minx
            splitx = self.minx + w * proportion
            bbox1 = BoundingBox(
                self.minx, splitx, self.miny, self.maxy, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                splitx, self.maxx, self.miny, self.maxy, self.mint, self.maxt
            )
        else:
            h = self.maxy - self.miny
            splity = self.miny + h * proportion
            bbox1 = BoundingBox(
                self.minx, self.maxx, self.miny, splity, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                self.minx, self.maxx, splity, self.maxy, self.mint, self.maxt
            )

        return bbox1, bbox2


def disambiguate_timestamp(date_str: str, format: str) -> tuple[float, float]:
    """Disambiguate partial timestamps.

    TorchGeo stores the timestamp of each file in a spatiotemporal R-tree. If the full
    timestamp isn't known, a file could represent a range of time. For example, in the
    CDL dataset, each mask spans an entire year. This method returns the maximum
    possible range of timestamps that ``date_str`` could belong to. It does this by
    parsing ``format`` to determine the level of precision of ``date_str``.

    Args:
        date_str: string representing date and time of a data point
        format: format codes accepted by :meth:`datetime.datetime.strptime`

    Returns:
        (mint, maxt) tuple for indexing
    """
    mint = datetime.strptime(date_str, format)

    # TODO: This doesn't correctly handle literal `%%` characters in format
    # TODO: May have issues with time zones, UTC vs. local time, and DST
    # TODO: This is really tedious, is there a better way to do this?

    if not any([f"%{c}" in format for c in "yYcxG"]):
        # No temporal info
        return 0, sys.maxsize
    elif not any([f"%{c}" in format for c in "bBmjUWcxV"]):
        # Year resolution
        maxt = datetime(mint.year + 1, 1, 1)
    elif not any([f"%{c}" in format for c in "aAwdjcxV"]):
        # Month resolution
        if mint.month == 12:
            maxt = datetime(mint.year + 1, 1, 1)
        else:
            maxt = datetime(mint.year, mint.month + 1, 1)
    elif not any([f"%{c}" in format for c in "HIcX"]):
        # Day resolution
        maxt = mint + timedelta(days=1)
    elif not any([f"%{c}" in format for c in "McX"]):
        # Hour resolution
        maxt = mint + timedelta(hours=1)
    elif not any([f"%{c}" in format for c in "ScX"]):
        # Minute resolution
        maxt = mint + timedelta(minutes=1)
    elif not any([f"%{c}" in format for c in "f"]):
        # Second resolution
        maxt = mint + timedelta(seconds=1)
    else:
        # Microsecond resolution
        maxt = mint + timedelta(microseconds=1)

    maxt -= timedelta(microseconds=1)

    return mint.timestamp(), maxt.timestamp()


def _list_dict_to_dict_list(samples: Iterable[dict[Any, Any]]) -> dict[Any, list[Any]]:
    """Convert a list of dictionaries to a dictionary of lists.

    Args:
        samples: a list of dictionaries

    Returns:
        a dictionary of lists

    .. versionadded:: 0.2
    """
    collated = collections.defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            collated[key].append(value)
    return collated


def _dict_list_to_list_dict(sample: dict[Any, Sequence[Any]]) -> list[dict[Any, Any]]:
    """Convert a dictionary of lists to a list of dictionaries.

    Args:
        sample: a dictionary of lists

    Returns:
        a list of dictionaries

    .. versionadded:: 0.2
    """
    uncollated: list[dict[Any, Any]] = [
        {} for _ in range(max(map(len, sample.values())))
    ]
    for key, values in sample.items():
        for i, value in enumerate(values):
            uncollated[i][key] = value
    return uncollated


def stack_samples(samples: Iterable[dict[Any, Any]]) -> dict[Any, Any]:
    """Stack a list of samples along a new axis.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: dict[Any, Any] = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.stack(value)
    return collated


def concat_samples(samples: Iterable[dict[Any, Any]]) -> dict[Any, Any]:
    """Concatenate a list of samples along an existing axis.

    Useful for joining samples in a :class:`torchgeo.datasets.IntersectionDataset`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: dict[Any, Any] = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.cat(value)
        else:
            collated[key] = value[0]
    return collated


def merge_samples(samples: Iterable[dict[Any, Any]]) -> dict[Any, Any]:
    """Merge a list of samples.

    Useful for joining samples in a :class:`torchgeo.datasets.UnionDataset`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: dict[Any, Any] = {}
    for sample in samples:
        for key, value in sample.items():
            if key in collated and isinstance(value, Tensor):
                # Take the maximum so that nodata values (zeros) get replaced
                # by data values whenever possible
                collated[key] = torch.maximum(collated[key], value)
            else:
                collated[key] = value
    return collated


def rasterio_loader(path: str) -> np.typing.NDArray[np.int_]:
    """Load an image file using rasterio.

    Args:
        path: path to the image to be loaded

    Returns:
        the image
    """
    with rasterio.open(path) as f:
        array: np.typing.NDArray[np.int_] = f.read().astype(np.int32)
        # NonGeoClassificationDataset expects images returned with channels last (HWC)
        array = array.transpose(1, 2, 0)
    return array


def path_is_vsi(path: str) -> bool:
    """Checks if the given path is pointing to a Virtual File System.

    .. note::
       Does not check if the path exists, or if it is a dir or file.

    VSI can for instance be Cloud Storage Blobs or zip-archives.
    They will start with a prefix indicating this.
    For examples of these, see references for the two accepted syntaxes.

    * https://gdal.org/user/virtual_file_systems.html
    * https://rasterio.readthedocs.io/en/latest/topics/datasets.html

    Args:
        path: string representing a directory or file

    Returns:
        True if path is on a virtual file system, else False

    .. versionadded:: 0.6
    """
    return "://" in path or path.startswith("/vsi")


"""Common sampler utilities."""


@overload
def _to_tuple(value: Union[tuple[int, int], int]) -> tuple[int, int]: ...


@overload
def _to_tuple(value: Union[tuple[float, float], float]) -> tuple[float, float]: ...


def _to_tuple(value: Union[tuple[float, float], float]) -> tuple[float, float]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (float, int)):
        return (value, value)
    else:
        return value


def get_random_bounding_box(
    bounds: BoundingBox, size: Union[tuple[float, float], float], res: float
) -> BoundingBox:
    """Returns a random bounding box within a given bounding box.

    The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size = _to_tuple(size)

    width = (bounds.maxx - bounds.minx - t_size[1]) // res
    height = (bounds.maxy - bounds.miny - t_size[0]) // res

    minx = bounds.minx
    miny = bounds.miny

    # random.randrange crashes for inputs <= 0
    if width > 0:
        minx += torch.rand(1).item() * width * res
    if height > 0:
        miny += torch.rand(1).item() * height * res

    maxx = minx + t_size[1]
    maxy = miny + t_size[0]

    mint = bounds.mint
    maxt = bounds.maxt

    query = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    return query


def tile_to_chips(
    bounds: BoundingBox,
    size: tuple[float, float],
    stride: Optional[tuple[float, float]] = None,
) -> tuple[int, int]:
    r"""Compute number of :term:`chips <chip>` that can be sampled from a :term:`tile`.

    Let :math:`i` be the size of the input tile. Let :math:`k` be the requested size of
    the output patch. Let :math:`s` be the requested stride. Let :math:`o` be the number
    of output chips sampled from each tile. :math:`o` can then be computed as:

    .. math::

       o = \left\lceil \frac{i - k}{s} \right\rceil + 1

    This is almost identical to relationship 5 in
    https://doi.org/10.48550/arXiv.1603.07285. However, we use ceiling instead of floor
    because we want to include the final remaining chip in each row/column when bounds
    is not an integer multiple of stride.

    Args:
        bounds: bounding box of tile
        size: size of output patch
        stride: stride with which to sample (defaults to ``size``)

    Returns:
        the number of rows/columns that can be sampled

    .. versionadded:: 0.4
    """
    if stride is None:
        stride = size

    assert stride[0] > 0
    assert stride[1] > 0

    rows = math.ceil((bounds.maxy - bounds.miny - size[0]) / stride[0]) + 1
    cols = math.ceil((bounds.maxx - bounds.minx - size[1]) / stride[1]) + 1

    return rows, cols
