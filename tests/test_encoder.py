import os
import tempfile
from pathlib import Path
import unittest
import pytest
from qgis.core import (
        QgsProcessingContext, 
        QgsProcessingFeedback,
        )

import timm
import torch
# from torchgeo.datasets import RasterDataset
from..tg.datasets import RasterDataset

from ..encoder import EncoderAlgorithm
from ..utils.misc import get_file_md5_hash


INPUT = os.path.join(Path(__file__).parent.parent.absolute(), 'assets', 'test.tif')
OUTPUT = os.path.join(tempfile.gettempdir(), "iamap_test")

class TestEncoderAlgorithm(unittest.TestCase):

    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = EncoderAlgorithm()
        self.default_parameters = {
                'BACKBONE_CHOICE': '', 
                'BACKBONE_OPT': 0, 
                'BANDS': None, 
                'BATCH_SIZE': 1, 
                'CKPT': 'NULL', 
                'CRS': None, 
                'CUDA': True, 
                'CUDA_ID': 0, 
                'EXTENT': None, 
                'FEAT_OPTION': True, 
                'INPUT': INPUT,
                'MERGE_METHOD': 0, 
                'OUTPUT': OUTPUT,
                'PAUSES': 0, 
                'QUANT': True, 
                'REMOVE_TEMP_FILES': True, 
                'RESOLUTION': None, 
                'SIZE': 224, 
                'STRIDE': 224, 
                'TEMP_FILES_CLEANUP_FREQ': 1000, 
                'WORKERS': 0,
                'JSON_PARAM': 'NULL', 
                'OUT_DTYPE': 0, 
                      }

    def test_valid_parameters(self):

        self.algorithm.initAlgorithm()
        _ = self.algorithm.processAlgorithm(self.default_parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_subdir,'merged.tif')
        result_file_hash = get_file_md5_hash(expected_result_path)

        ## different rasterio versions lead to different hashes ? 
        ## GPU and quantization as well
        possible_hashes = [
                '0fb32cc57a0dd427d9f0165ec6d5418f',
                '48c3a78773dbc2c4c7bb7885409284ab',
                '431e034b842129679b99a067f2bd3ba4',
                '60153535214eaa44458db4e297af72b9',
                'f1394d1950f91e4f8277a8667ae77e85',
                'a23837caa3aca54aaca2974d546c5123',
                           ]
        assert result_file_hash in possible_hashes
        os.remove(expected_result_path)

    @pytest.mark.slow
    def test_data_types(self):

        self.algorithm.initAlgorithm()
        parameters = self.default_parameters
        parameters['OUT_DTYPE'] = 1
        _ = self.algorithm.processAlgorithm(parameters, self.context, self.feedback)
        expected_result_path = os.path.join(self.algorithm.output_subdir,'merged.tif')
        result_file_hash = get_file_md5_hash(expected_result_path)

        ## different rasterio versions lead to different hashes ? 
        possible_hashes = [
                'ef0c4b0d57f575c1cd10c0578c7114c0',
                'ebfad32752de71c5555bda2b40c19b2e',
                'd3705c256320b7190dd4f92ad2087247',
                '65fa46916d6d0d08ad9656d7d7fabd01',
                           ]
        assert result_file_hash in possible_hashes
        os.remove(expected_result_path)


    def test_timm_create_model(self):

        archs = [
                'vit_base_patch16_224.dino',
                'vit_tiny_patch16_224.augreg_in21k',
                'vit_base_patch16_224.mae',
                'samvit_base_patch16.sa1b',
                ]
        expected_output_size = [
                torch.Size([1,197,768]),
                torch.Size([1,197,192]),
                torch.Size([1,197,768]),
                torch.Size([1, 256, 64, 64]),
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


    def test_RasterDataset(self):

        self.algorithm.initAlgorithm()
        parameters = {}
        self.algorithm.process_options(parameters, self.context, self.feedback)
        RasterDataset.filename_glob = self.algorithm.rlayer_name
        RasterDataset.all_bands = [
            self.algorithm.rlayer.bandName(i_band) for i_band in range(1, self.algorithm.rlayer.bandCount()+1)
        ]
        # currently only support rgb bands
        input_bands = [self.algorithm.rlayer.bandName(i_band)
                       for i_band in self.algorithm.selected_bands]

        if self.algorithm.crs == self.algorithm.rlayer.crs():
            dataset = RasterDataset(
                paths=self.algorithm.rlayer_dir, crs=None, res=self.algorithm.res, bands=input_bands, cache=False)
        else:
            dataset = RasterDataset(
                paths=self.algorithm.rlayer_dir, crs=self.algorithm.crs.toWkt(), res=self.algorithm.res, bands=input_bands, cache=False)
        del dataset

    def test_cuda(self):
        if torch.cuda.is_available():
            assert True





if __name__ == "__main__":

    test_encoder = TestEncoderAlgorithm()
    test_encoder.setUp()
    test_encoder.test_timm_create_model()
    test_encoder.test_RasterDataset()
    test_encoder.test_valid_parameters()
    test_encoder.test_data_types()
    test_encoder.test_cuda()
