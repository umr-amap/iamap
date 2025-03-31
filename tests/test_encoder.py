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
            "CKPT": "NULL",
            "CRS": None,
            "CUDA": True,
            "CUDA_ID": 0,
            "EXTENT": None,
            "FEAT_OPTION": True,
            "INPUT": INPUT,
            "MERGE_METHOD": 0,
            "OUTPUT": OUTPUT,
            "PAUSES": 0,
            "QUANT": True,
            "REMOVE_TEMP_FILES": True,
            "RESOLUTION": None,
            "SIZE": 224,
            "STRIDE": 224,
            "TEMP_FILES_CLEANUP_FREQ": 1000,
            "WORKERS": 0,
            "JSON_PARAM": "NULL",
            "OUT_DTYPE": 0,
        }

    @pytest.mark.xfail(raises=huggingface_hub.errors.LocalEntryNotFoundError)
    def test_valid_parameters(self):
        self.algorithm.initAlgorithm()
        _ = self.algorithm.processAlgorithm(
            self.default_parameters, self.context, self.feedback
        )
        expected_result_path = os.path.join(self.algorithm.output_subdir, "merged.tif")
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

    def test_RasterDataset(self):
        self.algorithm.initAlgorithm()
        parameters = {
            "INPUT": INPUT,
                }
        self.algorithm.process_options(parameters, self.context, self.feedback)
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
    test_encoder.test_timm_create_model()
    test_encoder.test_RasterDataset()
    test_encoder.test_valid_parameters()
    test_encoder.test_cuda()
