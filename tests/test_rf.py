import os
import hashlib
import unittest
from qgis.core import QgsProcessingContext, QgsProcessingFeedback

from ..random_forest import RFAlgorithm
from ..similarity import SimilarityAlgorithm
from ..utils.misc import get_file_md5_hash


class TestRFAlgorithm(unittest.TestCase):

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = RFAlgorithm()
        self.sim = SimilarityAlgorithm()

    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_dir,'random_forest.tif')
        result_file_hash = get_file_md5_hash(expected_result_path)
        assert result_file_hash == '80b7dd5b5ad5a4c0ad1d637fa20cf8b0'
        os.remove(expected_result_path)

    def test_rf_then_sim(self):
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        self.sim.initAlgorithm()
        parameters = {}
        result = self.sim.processAlgorithm(parameters, self.context, self.feedback)
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)


if __name__ == "__main__":

    test_algo = TestRFAlgorithm()
    test_algo.setUp()
    test_algo.test_valid_parameters()
    test_algo.test_rf_then_sim()
