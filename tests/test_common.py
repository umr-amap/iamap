import os
import pytest
from pathlib import Path
import tempfile
import unittest
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from ..ml import MLAlgorithm
from ..similarity import SimilarityAlgorithm
from ..clustering import ClusterAlgorithm
from ..reduction import ReductionAlgorithm
from ..utils.misc import get_file_md5_hash, remove_files_with_extensions
from ..utils.geo import validate_geotiff


INPUT = os.path.join(Path(__file__).parent.parent.absolute(), "assets", "test.tif")
OUTPUT = os.path.join(tempfile.gettempdir(), "iamap_test")
EXTENSIONS_TO_RM = [".tif", ".pkl", ".json", ".shp", ".shx", ".prj", ".dbf", ".cpg"]
TEMPLATE = os.path.join(
    Path(__file__).parent.parent.absolute(), "assets", "template.shp"
)
TEMPLATE_RF = os.path.join(
    Path(__file__).parent.parent.absolute(), "assets", "ml_poly.shp"
)
GT_COL = "Type"


class TestReductionAlgorithm(unittest.TestCase):
    """
    Base test class, other will inherit from this
    """

    algorithm = ReductionAlgorithm()
    default_parameters = {"INPUT": INPUT, "OUTPUT": OUTPUT}
    possible_hashes = [
        "d7a32c6b7a4cee1af9c73607561d7b25",
        "e04f8c86d9aad81dd9c625b9cd8f9824",
    ]
    output_size = 4405122
    output_wh = (968,379)
    out_name = "proj.tif"

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()

    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        _ = self.algorithm.processAlgorithm(
            self.default_parameters, self.context, self.feedback
        )
        expected_result_path = os.path.join(self.algorithm.output_dir, self.out_name)
        @pytest.mark.parametrize("output_file", expected_result_path, "expected_output_size", self.output_size, "expected_wh", self.output_wh)
        def test_geotiff_validity(output_file):
            validate_geotiff(output_file)
        remove_files_with_extensions(self.algorithm.output_dir, EXTENSIONS_TO_RM)


class TestClusteringAlgorithm(TestReductionAlgorithm):
    algorithm = ClusterAlgorithm()
    # possible_hashes = ["0c47b0c4b4c13902db5da3ee6e5d4aef"]
    out_name = "cluster.tif"
    output_size = 4405122


class TestSimAlgorithm(TestReductionAlgorithm):
    algorithm = SimilarityAlgorithm()
    default_parameters = {"INPUT": INPUT, "OUTPUT": OUTPUT, "TEMPLATE": TEMPLATE}
    # possible_hashes = ["f76eb1f0469725b49fe0252cfe86829a"]
    out_name = "similarity.tif"
    output_size = 1468988


class TestMLAlgorithm(TestReductionAlgorithm):
    algorithm = MLAlgorithm()
    default_parameters = {
        "INPUT": INPUT,
        "OUTPUT": OUTPUT,
        "TEMPLATE": TEMPLATE_RF,
        "GT_COL": GT_COL,
    }
    # possible_hashes = ["bd22d66180347e043fca58d494876184"]
    out_name = "ml.tif"
    output_size = 367520


if __name__ == "__main__":
    for algo in [
        TestReductionAlgorithm(),
        TestClusteringAlgorithm(),
        TestSimAlgorithm(),
        TestMLAlgorithm(),
    ]:
        algo.setUp()
        print(algo.algorithm)
        algo.test_valid_parameters()
