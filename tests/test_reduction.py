import os
import hashlib
import unittest
from qgis.core import QgsProcessingContext, QgsProcessingFeedback

from ..reduction import ReductionAlgorithm

## for hashing without using to much memory
BUF_SIZE = 65536


class TestReductionAlgorithm(unittest.TestCase):

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = ReductionAlgorithm()

    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_dir,'proj.tif')
        md5 = hashlib.md5()
        with open(expected_result_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
        result_file_hash = md5.hexdigest()
        assert result_file_hash == 'e47ca7b39f252506526a1ebfa1f33f6c'
        os.remove(expected_result_path)


if __name__ == "__main__":

    test_algo = TestReductionAlgorithm()
    test_algo.setUp()
    test_algo.test_valid_parameters()

