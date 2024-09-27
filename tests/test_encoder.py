import os
import hashlib
import unittest
from qgis.core import (
        QgsProcessingContext, 
        QgsProcessingFeedback,
        )

import timm
import torch
from torchgeo.transforms import AugmentationSequential

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


    def test_timm_create_model(self):

        archs = [
                'vit_base_patch16_224.dino',
                'vit_tiny_patch16_224.augreg_in21k',
                'vit_base_patch16_224.mae',
                # 'samvit_base_patch16.sa1b',
                ]
        expected_output_size = [
                torch.Size([1,197,768]),
                torch.Size([1,197,192]),
                torch.Size([1,197,768]),
                # torch.Size([1, 256, 64, 64]),
                ]

        for arch, exp_feat_size in zip(archs, expected_output_size):

            model = timm.create_model(
                arch,
                pretrained=True,
                in_chans=6,
                num_classes=0,
                )
            model = model.eval()

            data_config = timm.data.resolve_model_data_config(model)
            _, h, w, = data_config['input_size']
            output = model.forward_features(torch.randn(1,6,h,w))

            assert output.shape == exp_feat_size




if __name__ == "__main__":

    test_encoder = TestEncoderAlgorithm()
    test_encoder.setUp()
    test_encoder.test_valid_parameters()
    test_encoder.test_timm_create_model()
