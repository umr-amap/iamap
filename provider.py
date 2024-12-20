from qgis.core import QgsProcessingProvider

from .encoder import EncoderAlgorithm
from .reduction import ReductionAlgorithm
from .clustering import ClusterAlgorithm
from .similarity import SimilarityAlgorithm
from .ml import MLAlgorithm
from .icons import QIcon_EncoderTool


class IAMapProvider(QgsProcessingProvider):
    def loadAlgorithms(self, *args, **kwargs):
        self.addAlgorithm(EncoderAlgorithm())
        self.addAlgorithm(ReductionAlgorithm())
        self.addAlgorithm(ClusterAlgorithm())
        self.addAlgorithm(SimilarityAlgorithm())
        self.addAlgorithm(MLAlgorithm())
        # add additional algorithms here
        # self.addAlgorithm(MyOtherAlgorithm())

    def id(self, *args, **kwargs):
        """The ID of your plugin, used for identifying the provider.

        This string should be a unique, short, character only string,
        eg "qgis" or "gdal". This string should not be localised.
        """
        return "iamap"

    def name(self, *args, **kwargs):
        """The human friendly name of your plugin in Processing.

        This string should be as short as possible (e.g. "Lastools", not
        "Lastools version 1.0.1 64-bit") and localised.
        """
        return self.tr("IAMap")

    def icon(self):
        """Should return a QIcon which is used for your provider inside
        the Processing toolbox.
        """
        return QIcon_EncoderTool

    def longName(self) -> str:
        return self.name()
