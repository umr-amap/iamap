import os
import hashlib
import unittest
from qgis.core import (
        QgsProcessingContext, 
        QgsProcessingFeedback,
        )

from ..encoder import EncoderAlgorithm

## for hashing without using to much memory
BUF_SIZE = 65536


class TestEncoderAlgorithm(unittest.TestCase):

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = EncoderAlgorithm()

    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        parameters = {}
        result = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_subdir,'merged.tif')
        md5 = hashlib.md5()
        with open(expected_result_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
        result_file_hash = md5.hexdigest()
        assert result_file_hash == '018b6fc5d88014a7e515824d95ca8686'
        os.remove(expected_result_path)

if __name__ == "__main__":

    test_encoder = TestEncoderAlgorithm()
    test_encoder.setUp()
    test_encoder.test_valid_parameters()
