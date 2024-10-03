import os
import hashlib
import unittest
from qgis.core import QgsProcessingContext, QgsProcessingFeedback

from ..similarity import SimilarityAlgorithm
from ..utils.misc import get_file_md5_hash


class TestSimilarityAlgorithm(unittest.TestCase):

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = SimilarityAlgorithm()

    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_dir,'similarity.tif')
        result_file_hash = get_file_md5_hash(expected_result_path)
        assert result_file_hash == 'f76eb1f0469725b49fe0252cfe86829a'
        os.remove(expected_result_path)


if __name__ == "__main__":

    test_algo = TestSimilarityAlgorithm()
    test_algo.setUp()
    test_algo.test_valid_parameters()
