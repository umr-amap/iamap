import os
import hashlib
import unittest
from qgis.core import QgsProcessingContext, QgsProcessingFeedback

from ..clustering import ClusterAlgorithm
from ..utils.misc import get_file_md5_hash

class TestClusteringAlgorithm(unittest.TestCase):

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = ClusterAlgorithm()

    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_dir,'cluster.tif')
        result_file_hash = get_file_md5_hash(expected_result_path)
        assert result_file_hash == 'ecb1e17173b1601866ae7055694739e8'
        os.remove(expected_result_path)


if __name__ == "__main__":

    test_algo = TestClusteringAlgorithm()
    test_algo.setUp()
    test_algo.test_valid_parameters()
