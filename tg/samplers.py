import abc
import sys
from collections.abc import Iterator
from typing import Optional, Union

import torch
from rtree.index import Index, Property
from torch.utils.data import Sampler

from .datasets import GeoDataset
from enum import Enum, auto
from .utils import BoundingBox
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips
from qgis.core import QgsSpatialIndex, QgsRectangle, QgsPointXY, QgsFeature, QgsGeometry
from qgis.core import QgsRectangle, QgsFeatureRequest

# def compute_bounds(spatial_index: QgsSpatialIndex, dataset: GeoDataset) -> BoundingBox:
#     """Compute the bounding box (extent) for all features in the spatial index."""
#     minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')

#     # Use QgsFeatureRequest to fetch the features from the dataset using their IDs
#     for feature_id in spatial_index.intersects(QgsRectangle(float('-inf'), float('-inf'), float('inf'), float('inf'))):
#         # Use QgsFeatureRequest to fetch the feature by ID
#         request = QgsFeatureRequest().setFilterFid(feature_id)
#         feature = next(dataset.features(request), None)  # Fetch the feature

#         if feature is not None:
#             geometry = feature.geometry().boundingBox()

#             # Update bounds
#             minx = min(minx, geometry.xMinimum())
#             miny = min(miny, geometry.yMinimum())
#             maxx = max(maxx, geometry.xMaximum())
#             maxy = max(maxy, geometry.yMaximum())

#     return BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=0, maxt=sys.maxsize)

def compute_bounds(spatial_index: QgsSpatialIndex, dataset: GeoDataset) -> BoundingBox:
    """Compute the bounding box (extent) for all features in the spatial index."""
    minx, miny, maxx, maxy = float('-inf'), float('-inf'), float('inf'), float('inf')

    # Use QgsFeatureRequest to fetch the features from the dataset using their IDs
    for feature_id in spatial_index.intersects(QgsRectangle(float('-inf'), float('-inf'), float('inf'), float('inf'))):
        # Access the feature directly from the dataset.features dictionary using feature_id
        feature = dataset.features.get(feature_id, None)

        if feature is not None:
            geometry = feature.geometry().boundingBox()

            # Update bounds
            minx = min(minx, geometry.xMinimum())
            miny = min(miny, geometry.yMinimum())
            maxx = max(maxx, geometry.xMaximum())
            maxy = max(maxy, geometry.yMaximum())

    return BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=0, maxt=sys.maxsize)



# def compute_bounds(spatial_index: QgsSpatialIndex, dataset: GeoDataset) -> BoundingBox:
#     """Compute the bounding box (extent) for all features in the spatial index."""
#     minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')

#     # Iterate over all features in the dataset
#     for feature_id in spatial_index.intersects(QgsRectangle(float('-inf'), float('-inf'), float('inf'), float('inf'))):
#         feature = dataset.features[feature_id]
#         geometry = feature.geometry().boundingBox()

#         # Update bounds
#         minx = min(minx, geometry.xMinimum())
#         miny = min(miny, geometry.yMinimum())
#         maxx = max(maxx, geometry.xMaximum())
#         maxy = max(maxy, geometry.yMaximum())

#     return BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=0, maxt=sys.maxsize)


class Units(Enum):
    """Enumeration defining units of ``size`` parameter.

    Used by :class:`~torchgeo.samplers.GeoSampler` and
    :class:`~torchgeo.samplers.BatchGeoSampler`.
    """

    #: Units in number of pixels
    PIXELS = auto()

    #: Units of :term:`coordinate reference system (CRS)`
    CRS = auto()


# class GeoSampler(Sampler[BoundingBox], abc.ABC):
#     """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

#     Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
#     returns enough geospatial information to uniquely index any
#     :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
#     longitude, height, width, projection, coordinate system, and time.
#     """

#     def __init__(self, dataset: GeoDataset, roi: Optional[BoundingBox] = None) -> None:
#         """Initialize a new Sampler instance.

#         Args:
#             dataset: dataset to index from
#             roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
#                 (defaults to the bounds of ``dataset.index``)
#         """
#         if roi is None:
#             self.index = dataset.index
#             roi = BoundingBox(*self.index.bounds)
#         else:
#             self.index = Index(interleaved=False, properties=Property(dimension=3))
#             hits = dataset.index.intersection(tuple(roi), objects=True)
#             for hit in hits:
#                 bbox = BoundingBox(*hit.bounds) & roi
#                 self.index.insert(hit.id, tuple(bbox), hit.object)

#         self.res = dataset.res
#         self.roi = roi

#     @abc.abstractmethod
#     def __iter__(self) -> Iterator[BoundingBox]:
#         """Return the index of a dataset.

#         Returns:
#             (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
#         """

class GeoSampler(Sampler[BoundingBox], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: Optional[BoundingBox] = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        self.dataset = dataset
        self.res = dataset.res
        
        # If ROI is not provided, use the dataset bounds
        if roi is None:
            self.index = dataset.index
            roi = BoundingBox(*self.index.boundingBox())
        else:
            # Create a new spatial index and insert only items in the region of interest (ROI)
            self.index = QgsSpatialIndex()
            self.temporal_data = {}  # To store the temporal components (mint, maxt)

            # Find dataset features intersecting with the ROI
            hits = dataset.index.intersects(QgsRectangle(roi.minx, roi.miny, roi.maxx, roi.maxy))
            print(hits)
            print(dataset.features)
            for fid in hits:
                feature = dataset.features[fid]
                print(feature)
                # self.temporal_data[fid] = (bbox.mint, bbox.maxt)
                # print(dataset.features)
                ## Not handling temporal values for now
                bbox = BoundingBox(*feature.geometry().boundingBox().toRectF().getCoords(), 0,0) & roi
                self.index.insertFeature(feature)

        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        length: Optional[int],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        # for hit in self.index.intersection(tuple(self.roi), objects=True):
        #     bounds = BoundingBox(*hit.bounds)
        #     if (
        #         bounds.maxx - bounds.minx >= self.size[1]
        #         and bounds.maxy - bounds.miny >= self.size[0]
        #     ):
        #         self.hits.append(hit)
        for hit in self.index.intersects(QgsRectangle(self.roi.minx, self.roi.miny, self.roi.maxx, self.roi.maxy)):
            
            feature = self.dataset.features[hit]
            # print(feature)
            ## Not handling temporal values for now
            bounds = BoundingBox(*feature.geometry().boundingBox().toRectF().getCoords(), 0,0)
            self.index.insertFeature(feature)
            # hit.bounds = bounds
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                self.hits.append(bounds)

        self.length = 0
        for hit in self.hits:
            # bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(hit, self.size, self.stride)
            self.length += rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


class NoBordersGridGeoSampler(GridGeoSampler):

    def __iter__(self) -> Iterator[BoundingBox]:
        """
        Modification of original Torchgeo sampler to avoid overlapping borders of a dataset.
        Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for bounds in self.hits:
            # bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > bounds.maxy:
                    maxy = bounds.maxy
                    miny = bounds.maxy - self.size[0]
                    if miny < bounds.miny:
                        miny = bounds.miny

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > bounds.maxx:
                        maxx = bounds.maxx
                        minx = bounds.maxx - self.size[1]
                        if minx < bounds.minx:
                            minx = bounds.minx

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)
