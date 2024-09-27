import os
import hashlib
import unittest
from qgis.core import QgsProcessingContext, QgsProcessingFeedback

from ..random_forest import RFAlgorithm

## for hashing without using to much memory
BUF_SIZE = 65536


class TestRFAlgorithm(unittest.TestCase):

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = RFAlgorithm()

    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_dir,'random_forest.tif')
        md5 = hashlib.md5()
        with open(expected_result_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
        result_file_hash = md5.hexdigest()
        assert result_file_hash == '80b7dd5b5ad5a4c0ad1d637fa20cf8b0'


if __name__ == "__main__":

    test_algo = TestRFAlgorithm()
    test_algo.setUp()
    test_algo.test_valid_parameters()




