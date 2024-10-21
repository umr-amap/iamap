import os
import hashlib
import unittest
from qgis.core import QgsProcessingContext, QgsProcessingFeedback

from ..reduction import ReductionAlgorithm
from ..utils.misc import get_file_md5_hash


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
        result_file_hash = get_file_md5_hash(expected_result_path)
        ## different rasterio versions lead to different hashes ? 
        possible_hashes = [
                # '5eef4ea313d45b12beba8a7b9e2500ba', 
                # '743465d291bd2ada6ea9807752c6e7fe',
                'd7a32c6b7a4cee1af9c73607561d7b25',
                           ]
        assert result_file_hash in possible_hashes
        # assert result_file_hash == '5eef4ea313d45b12beba8a7b9e2500ba'
        os.remove(expected_result_path)


if __name__ == "__main__":

    test_algo = TestReductionAlgorithm()
    test_algo.setUp()
    test_algo.test_valid_parameters()

