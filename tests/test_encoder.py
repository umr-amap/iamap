import os
import logging
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
import huggingface_hub

# from torchgeo.datasets import RasterDataset
from ..tg.datasets import RasterDataset

from ..encoder import EncoderAlgorithm
from ..utils.misc import get_file_md5_hash
from ..utils.geo import validate_geotiff


INPUT = os.path.join(Path(__file__).parent.parent.absolute(), "assets", "test.tif")
OUTPUT = os.path.join(tempfile.gettempdir(), "iamap_test")


class TestEncoderAlgorithm(unittest.TestCase):
    def setUp(self):
        self.context = QgsProcessingContext()
        self.feedback = QgsProcessingFeedback()
        self.algorithm = EncoderAlgorithm()
        self.default_parameters = {
            "BACKBONE_CHOICE": "",
            "BACKBONE_OPT": 0,
            "BANDS": None,
            "BATCH_SIZE": 1,
            "CKPT": "",
            "CRS": None,
            "CUDA": True,
            "CUDA_ID": 0,
            "EXTENT": None,
            "FEAT_OPTION": True,
            "INPUT": INPUT,
            "MERGE_METHOD": 0,
            "OUTPUT": OUTPUT,
            "PAUSES": 0,
            "QUANT": False,
            "REMOVE_TEMP_FILES": True,
            "RESOLUTION": None,
            "SIZE": 224,
            "STRIDE": 224,
            "TEMP_FILES_CLEANUP_FREQ": 1000,
            "WORKERS": 0,
            "JSON_PARAM": "NULL",
            "OUT_DTYPE": 0,
        }
        logging_level = logging.INFO
        ignore_rasterio_logs = True
        self.algorithm.logger = self.algorithm.redirect_logger(
                self.feedback, 
                level=logging_level, 
                ignore_rasterio=ignore_rasterio_logs
                )

    @pytest.mark.xfail(raises=huggingface_hub.errors.LocalEntryNotFoundError)
    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        _ = self.algorithm.processAlgorithm(
            self.default_parameters, self.context, self.feedback
        )
        expected_result_path = os.path.join(self.algorithm.output_subdir, "test-merged.tif")
        @pytest.mark.parametrize("output_file", expected_result_path)
        def test_geotiff_validity(output_file):
            validate_geotiff(output_file)
        os.remove(expected_result_path)


    @pytest.mark.xfail(raises=huggingface_hub.errors.LocalEntryNotFoundError)
    def test_timm_create_model(self):
        archs = [
            "vit_base_patch16_224.dino",
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "samvit_base_patch16.sa1b",
        ]
        expected_output_size = [
            torch.Size([1, 197, 768]),
            torch.Size([1, 197, 192]),
            torch.Size([1, 197, 768]),
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
            (
                _,
                h,
                w,
            ) = data_config["input_size"]
            output = model.forward_features(torch.randn(1, 6, h, w))

            assert output.shape == exp_feat_size

    @pytest.mark.xfail(raises=huggingface_hub.errors.LocalEntryNotFoundError)
    def test_init_model(self):
        self.algorithm.cwd = Path(__file__).parent.parent.absolute()
        self.algorithm.ckpt_path = ''
        self.algorithm.quantization = True
        self.algorithm.device = 'cpu'
        archs = [
            Path(os.path.join(self.algorithm.cwd,'pangaea','configs','encoder','ssl4eo_moco.yaml')),
            Path(os.path.join(self.algorithm.cwd,'pangaea','configs','encoder','dofa.yaml')),
            "vit_small_patch8_224.dino",
            "vit_base_patch16_224.dino",
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "samvit_base_patch16.sa1b",
        ]
        expected_output_size = [
            torch.Size([1, 197, 768]),
            torch.Size([1, 197, 768]),
            torch.Size([1, 197, 768]),
            torch.Size([1, 197, 192]),
            torch.Size([1, 197, 768]),
            torch.Size([1, 197, 768]),
            # torch.Size([1, 256, 64, 64]),
            # torch.Size([1, 256, 64, 64]),
        ]
        nbands = [[1],[1,2,3,4,5,6,7,8,9,10,11,12]]

        for nband in nbands:
            for arch, exp_feat_size in zip(archs, expected_output_size):
                self.algorithm.backbone_name = arch
                self.algorithm.input_bands = nband 
                self.algorithm.init_model()
                images = torch.rand(1,len(nband),self.algorithm.w, self.algorithm.h)
                if '.yaml' in str(self.algorithm.backbone_name):
                    input={}
                    input['optical'] = images
                    features = self.algorithm.model(input)

                else:
                    features = self.algorithm.model.forward_features(images)

            assert features.shape == exp_feat_size

    def test_RasterDataset(self):
        self.algorithm.initAlgorithm()
        parameters = {
            "INPUT": INPUT,
                }
        self.algorithm.logger = self.algorithm.redirect_logger(feedback=self.feedback)
        self.algorithm.process_options(parameters, self.context)
        RasterDataset.filename_glob = self.algorithm.rlayer_name
        RasterDataset.all_bands = [
            self.algorithm.rlayer.bandName(i_band)
            for i_band in range(1, self.algorithm.rlayer.bandCount() + 1)
        ]
        # currently only support rgb bands
        input_bands = [
            self.algorithm.rlayer.bandName(i_band)
            for i_band in self.algorithm.selected_bands
        ]

        if self.algorithm.crs == self.algorithm.rlayer.crs():
            dataset = RasterDataset(
                paths=self.algorithm.rlayer_dir,
                crs=None,
                res=self.algorithm.res,
                bands=input_bands,
                cache=False,
            )
        else:
            dataset = RasterDataset(
                paths=self.algorithm.rlayer_dir,
                crs=self.algorithm.crs.toWkt(),
                res=self.algorithm.res,
                bands=input_bands,
                cache=False,
            )
        del dataset

    def test_cuda(self):
        if torch.cuda.is_available():
            assert True


if __name__ == "__main__":
    test_encoder = TestEncoderAlgorithm()
    test_encoder.setUp()
    # test_encoder.test_timm_create_model()
    test_encoder.test_init_model()
    test_encoder.test_RasterDataset()
    test_encoder.test_valid_parameters()
    test_encoder.test_cuda()
