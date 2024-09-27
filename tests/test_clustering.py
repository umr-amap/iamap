import os
import hashlib
import unittest
from qgis.core import QgsProcessingContext, QgsProcessingFeedback

from ..clustering import ClusterAlgorithm

## for hashing without using to much memory
BUF_SIZE = 65536


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
        md5 = hashlib.md5()
        with open(expected_result_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
        result_file_hash = md5.hexdigest()
        assert result_file_hash == '835bcb7ab7d7e97d2de26e81415a0d19'
        os.remove(expected_result_path)


if __name__ == "__main__":

    test_algo = TestClusteringAlgorithm()
    test_algo.setUp()
    test_algo.test_valid_parameters()
