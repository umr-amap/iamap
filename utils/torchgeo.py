from torchgeo.samplers import GridGeoSampler
from collections.abc import Iterator

from torchgeo.datasets import BoundingBox
from torchgeo.samplers.utils import tile_to_chips

class NoBordersGridGeoSampler(GridGeoSampler):

    def __iter__(self) -> Iterator[BoundingBox]:
        """
        Modification of original Torchgeo sampler to avoid overlapping borders of a dataset.
        Return the index of a dataset.

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
