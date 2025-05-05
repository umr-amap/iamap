import os
import sys
import logging
import time
import tempfile
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

import rasterio
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBand,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterCrs,
    QgsProcessingParameterDefinition,
)

import torch
import torch.quantization
from torch.utils.data import DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
import timm

from .pangaea.encoders.base import Encoder
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .utils.geo import get_mean_sd_by_band
from .utils.geo import merge_tiles, merge_two_tiles
from .utils.misc import (
    get_dir_size,
    get_model_size,
    remove_files,
    check_disk_space,
    get_unique_filename,
    save_parameters_to_json,
    compute_md5_hash,
    log_parameters_to_csv,
)
from .utils.trch import modify_first_conv2d, quantize_model
from .utils.algo import IAMAPAlgorithm

from .tg.datasets import RasterDataset
from .tg.utils import stack_samples, BoundingBox
from .tg.samplers import NoBordersGridGeoSampler, Units
from .tg.transforms import AugmentationSequential

from .icons import QIcon_EncoderTool


class EncoderAlgorithm(IAMAPAlgorithm):
    """ """

    FEAT_OPTION = "FEAT_OPTION"
    INPUT = "INPUT"
    CKPT = "CKPT"
    BANDS = "BANDS"
    STRIDE = "STRIDE"
    SIZE = "SIZE"
    EXTENT = "EXTENT"
    QUANT = "QUANT"
    OUTPUT = "OUTPUT"
    RESOLUTION = "RESOLUTION"
    CRS = "CRS"
    CUDA = "CUDA"
    BATCH_SIZE = "BATCH_SIZE"
    CUDA_ID = "CUDA_ID"
    BACKBONE_CHOICE = "BACKBONE_CHOICE"
    BACKBONE_OPT = "BACKBONE_OPT"
    MERGE_METHOD = "MERGE_METHOD"
    WORKERS = "WORKERS"
    PAUSES = "PAUSES"
    REMOVE_TEMP_FILES = "REMOVE_TEMP_FILES"
    TEMP_FILES_CLEANUP_FREQ = "TEMP_FILES_CLEANUP_FREQ"
    JSON_PARAM = "JSON_PARAM"
    COMPRESS = "COMPRESS"

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.cwd = Path(__file__).parent.absolute()
        tmp_wd = os.path.join(tempfile.gettempdir(), "iamap_features")

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr("Input raster layer or image file path"),
                # defaultValue=os.path.join(self.cwd, "assets", "test.tif"),
            ),
        )

        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BANDS,
                description=self.tr("Selected Bands (defaults to all bands selected)"),
                defaultValue=None,
                parentLayerParameterName=self.INPUT,
                optional=True,
                allowMultiple=True,
            )
        )
        compress_param = QgsProcessingParameterBoolean(
            name=self.COMPRESS,
            description=self.tr(
                "Compress final result to uint16 and JP2 to save space"
            ),
            defaultValue=False,
            optional=True,
        )

        crs_param = QgsProcessingParameterCrs(
            name=self.CRS,
            description=self.tr("Target CRS (default to original CRS)"),
            optional=True,
        )

        res_param = QgsProcessingParameterNumber(
            name=self.RESOLUTION,
            description=self.tr(
                "Target resolution in meters (default to native resolution)"
            ),
            type=QgsProcessingParameterNumber.Double,
            optional=True,
            minValue=0,
            maxValue=100000,
        )

        cuda_id_param = QgsProcessingParameterNumber(
            name=self.CUDA_ID,
            description=self.tr(
                "CUDA Device ID (choose which GPU to use, default to device 0)"
            ),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=9,
        )
        nworkers_param = QgsProcessingParameterNumber(
            name=self.WORKERS,
            description=self.tr("Number of CPU workers for dataloader (0 selects all)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=10,
        )
        pauses_param = QgsProcessingParameterNumber(
            name=self.PAUSES,
            description=self.tr(
                "Schedule pauses between batches to ease CPU usage (in seconds)."
            ),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=10000,
        )

        tmp_files_cleanup_frq = QgsProcessingParameterNumber(
            name=self.TEMP_FILES_CLEANUP_FREQ,
            description=self.tr(
                "Frequency at which temporary files should be cleaned up (zero means no cleanup)."
            ),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=500,
            minValue=1,
            maxValue=10000,
        )

        remove_tmp_files = QgsProcessingParameterBoolean(
            name=self.REMOVE_TEMP_FILES,
            description=self.tr(
                "Remove temporary files after encoding. If you want to test different merging options, it may be better to keep the tiles."
            ),
            defaultValue=True,
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr("Processing extent (default to the entire image)"),
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.SIZE,
                description=self.tr(
                    "Sampling size (the raster will be sampled in a square with a side of that many pixel)"
                ),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=224,
                minValue=1,
                maxValue=1024,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.STRIDE,
                description=self.tr(
                    "Stride (If smaller than the sampling size, tiles will overlap. If larger, it may cause errors.)"
                ),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=224,
                minValue=1,
                maxValue=1024,
            )
        )

        chkpt_param = QgsProcessingParameterFile(
            name=self.CKPT,
            description=self.tr("Pretrained checkpoint"),
            # extension='pth',
            fileFilter="Checkpoint Files (*.pth *.pkl);; All Files (*.*)",
            optional=True,
            defaultValue=None,
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr(
                    "Output directory (choose the location that the image features will be saved)"
                ),
                defaultValue=tmp_wd,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CUDA, self.tr("Use GPU if CUDA is available."), defaultValue=True
            )
        )
        self.backbone_opt = [
            "SSL4EO MOCO",
            "ViT small DINO patch 8",
            "ViT base DINO",
            "ViT tiny Imagenet (smallest)",
            "ViT base MAE",
            "SAM",
            "--Empty--",
        ]
        self.timm_backbone_opt = [
            Path(os.path.join(self.cwd,'pangaea','configs','encoder','ssl4eo_moco.yaml')),
            "vit_small_patch8_224.dino",
            "vit_base_patch16_224.dino",
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "samvit_base_patch16.sa1b",
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                name=self.BACKBONE_OPT,
                description=self.tr(
                    "Pre-selected backbones if you don't know what to pick"
                ),
                defaultValue=0,
                options=self.backbone_opt,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                name=self.BACKBONE_CHOICE,
                description=self.tr(
                    "Enter an architecture name if you want to test another backbone (see huggingface.co/timm/)"
                ),
                defaultValue=None,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BATCH_SIZE,
                # large images will be sampled into patches in a grid-like fashion
                description=self.tr(
                    "Batch size (take effect if choose to use GPU and CUDA is available)"
                ),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
                maxValue=1024,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.QUANT,
                self.tr("Quantization of the model to reduce space"),
                defaultValue=True,
            )
        )

        self.merge_options = ["first", "average", "min", "max", "sum", "count", "last"]
        merge_param = QgsProcessingParameterEnum(
            name=self.MERGE_METHOD,
            description=self.tr("Merge method at the end of inference."),
            options=self.merge_options,
            defaultValue=0,
        )

        json_param = QgsProcessingParameterFile(
            name=self.JSON_PARAM,
            description=self.tr("Pass parameters as json file"),
            fileFilter="JSON Files (*.json)",
            optional=True,
            defaultValue=None,
        )

        for param in (
            chkpt_param,
            cuda_id_param,
            remove_tmp_files,
            merge_param,
            tmp_files_cleanup_frq,
            nworkers_param,
            pauses_param,
            crs_param,
            res_param,
            compress_param,
            json_param,
        ):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced
            )
            self.addParameter(param)


    @torch.no_grad()
    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        logging_level = logging.INFO
        ignore_rasterio_logs = True
        self.logger = self.redirect_logger(
                feedback, 
                level=logging_level, 
                ignore_rasterio=ignore_rasterio_logs
                )
        parameters = self.load_parameters_as_json(parameters)
        self.process_options(parameters, context)
        self.create_output_dir(parameters)

        self.logger.info("Creating dataset...")
        self.create_dataset()
        self.logger.info("Dataset done")
        if feedback.isCanceled():
            feedback.pushWarning(self.tr("\n !!!Processing is canceled by user!!! \n"))
            return

        self.logger.info("Starting model loading ...")
        self.logger.info("This may take some time if it is the first time downloading model")
        # self.logger.info("You can change verbosity for more feedback")
        self.init_model()
        self.logger.info("Model done")
        if feedback.isCanceled():
            feedback.pushWarning(self.tr("\n !!!Processing is canceled by user!!! \n"))
            return

        self.create_dataloader(feedback)
        self.logger.debug(f"Total batch num: {len(self.dataloader)}")

        self.logger.info(f'\n\n{"-"*16}\nBegining inference \n{"-"*16}\n\n')
        self.inference(feedback)

        ## merging all temp tiles
        self.logger.info(f"\n\n{'-'*8}\n Merging tiles \n{'-'*8}\n")
        all_tiles = [
            os.path.join(self.output_subdir, f)
            for f in os.listdir(self.output_subdir)
            if f.endswith("_tmp.tif")
        ]
        rlayer_name, ext = os.path.splitext(self.rlayer_name)

        if not self.all_encoding_done:
            dst_path = Path(os.path.join(self.output_subdir, "merged_tmp.tif"))
            layer_name = f"{rlayer_name} features tmp"
        else:
            dst_path, layer_name = get_unique_filename(self.output_subdir, "merged.tif", f"{rlayer_name} features")
            dst_path = Path(dst_path)

        merge_tiles(
            tiles=all_tiles,
            dst_path=dst_path,
            method=self.merge_method,
        )

        if self.remove_tmp_files:
            self.remove_temp_files()

        parameters["OUTPUT_RASTER"] = dst_path

        if self.compress:
            self.dst_path = self.tiff_to_jp2(parameters, feedback)

        return {
            "Output feature path": self.output_subdir,
            "Patch samples saved": self.iPatch,
            "OUTPUT_RASTER": dst_path,
            "OUTPUT_LAYER_NAME": layer_name,
        }

    def inference(self, feedback):

        last_batch_done = self.get_last_batch_done()
        if last_batch_done >= 0:
            self.logger.info(
                f"\n\n {'-'*8} \n Resuming at batch number {last_batch_done}\n {'-'*8} \n\n"
            )

        bboxes = []  # keep track of bboxes to have coordinates at the end
        elapsed_time_list = []
        nbatch_total = len(self.dataloader)
        total = 100 / (nbatch_total+1) if nbatch_total else 0

        ## will update if process is canceled by the user
        self.all_encoding_done = True

        for current, sample in enumerate(self.dataloader):
            if current <= last_batch_done:
                continue

            start_time = time.time()

            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                self.load_feature = False
                feedback.pushWarning(self.tr("\n !!!Processing is canceled by user!!! \n"))
                self.all_encoding_done = False
                break

            images = sample["image"].to(self.device)
            images = images.squeeze(1)## kornia creates an empty dimension between batch and channels

            feedback.setProgressText(f"Processing batch {current}/{nbatch_total}")


            if '.yaml' in str(self.backbone_name):
                input={}
                input['optical'] = images
                features = self.model(input)

            else:
                features = self.model.forward_features(images)

            if features.shape[1] % 2 == 1: 
                features = features[:, 1:, :]  # take only patch tokens

            if current <= last_batch_done + 1:
                n_patches = int(np.sqrt(features.shape[1]))

            features = features.view(
                features.shape[0], n_patches, n_patches, features.shape[-1]
            )

            features = features.detach().cpu().numpy()
            self.logger.debug(f"Features shape {features.shape}")

            self.save_features(features, sample["bbox"], current)
            self.logger.debug("Features saved")

            if current <= last_batch_done + 1:
                total_space, total_used_space, free_space = check_disk_space(self.output_subdir)
                used_outputsubdir = get_dir_size(str(self.output_subdir))
                to_use = ((nbatch_total / (current + 1)) - 1) * used_outputsubdir
                if to_use >= free_space:
                    feedback.pushWarning(self.tr(f"\n !!! only {free_space} GB disk space remaining, canceling !!! \n"))
                    break

            bboxes.extend(sample["bbox"])

            if self.pauses != 0:
                time.sleep(self.pauses)

            end_time = time.time()
            # get the execution time of encoder, ms
            elapsed_time = end_time - start_time
            elapsed_time_list.append(elapsed_time)
            time_spent = sum(elapsed_time_list)
            time_remain = (time_spent / (current + 1)) * (nbatch_total - current - 1)

            # TODO: show gpu usage info
            # if torch.cuda.is_available() and self.use_gpu:
            #     gpu_mem_used = torch.cuda.max_memory_reserved(self.sam_model.device) / (1024 ** 3)
            #     # gpu_mem_free = torch.cuda.mem_get_info(self.sam_model.device)[0] / (1024 ** 3)
            #     gpu_mem_total = torch.cuda.mem_get_info(self.sam_model.device)[1] / (1024 ** 3)
            #     self.logger.info(
            #         f'GPU memory usage: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB')
            #     self.logger.info(str(torch.cuda.memory_summary(self.sam_model.device)))

            self.logger.debug(f"Encoder executed with {elapsed_time:.3f}s")
            self.logger.debug(f"Time spent: {time_spent:.3f}s")

            if time_remain <= 60:
                self.logger.info(f"Estimated time remaining: {time_remain:.3f}s \n {'-'*8}")
            else:
                time_remain_m, time_remain_s = divmod(int(time_remain), 60)
                time_remain_h, time_remain_m = divmod(time_remain_m, 60)
                self.logger.info(f"Estimated time remaining: {time_remain_h:d}h:{time_remain_m:02d}m:{time_remain_s:02d}s \n")

            if ((current + 1) % self.cleanup_frq == 0) and self.remove_tmp_files:
                ## not the cleanest way to do for now
                ## but avoids to refactor all
                self.all_encoding_done = False
                self.logger.info("Cleaning temporary files...")
                all_tiles = [
                    os.path.join(self.output_subdir, f)
                    for f in os.listdir(self.output_subdir)
                    if f.endswith("_tmp.tif")
                ]
                all_tiles = [f for f in all_tiles if not f.startswith("merged")]

                dst_path = Path(os.path.join(self.output_subdir, f"merged_tmp.tif"))

                try:
                    merge_tiles(
                        tiles=all_tiles,
                        dst_path=dst_path,
                        method=self.merge_method,
                    )
                    self.remove_temp_files()
                    
                # overwritting merged_tmp.tif may be impossible in windows (e.g. if an antivirus is analysing the newly created data)
                # then, merging and cleaning is impossible
                except Exception as e :
                    feedback.pushWarning(f"Unable to clean temporary files ! Try to delete them latter at {self.output_subdir}\n{e}")

                self.all_encoding_done = True

            # Update the progress bar
            feedback.setProgress(int((current + 1) * total))


    def create_dataloader(self, feedback):
        transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32),  # change dtype for normalize to be possible
            K.Normalize(self.means, self.sds),  # normalize occurs only on raster, not mask
            K.Resize((self.h, self.w)),  # resize to model expected input size, regardless of sampling size
            data_keys=["image"],
        )
        self.dataset.transforms = transform

        sampler = NoBordersGridGeoSampler(
            self.dataset,
            size=self.size,
            stride=self.stride,
            roi=self.extent_bbox,
            units=Units.PIXELS,
        )

        if len(sampler) == 0:
            self.load_feature = False
            feedback.pushWarning("\n !!!No available patch sample inside the chosen extent!!! \n")


        self.logger.debug(f"Batch size: {self.batch_size}")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=stack_samples,
            num_workers=self.nworkers,
        )


    def create_dataset(self):

        RasterDataset.filename_glob = self.rlayer_name
        RasterDataset.all_bands = [
            self.rlayer.bandName(i_band)
            for i_band in range(1, self.rlayer.bandCount() + 1)
        ]
        # currently only support rgb bands
        self.input_bands = [self.rlayer.bandName(i_band) for i_band in self.selected_bands]

        if self.crs == self.rlayer.crs():
            self.dataset = RasterDataset(
                paths=self.rlayer_dir,
                crs=None,
                res=self.res,
                bands=self.input_bands,
                cache=False,
            )
        else:
            self.dataset = RasterDataset(
                paths=self.rlayer_dir,
                crs=self.crs.toWkt(),
                res=self.res,
                bands=self.input_bands,
                cache=False,
            )
        self.extent_bbox = BoundingBox(
            minx=self.extent.xMinimum(),
            maxx=self.extent.xMaximum(),
            miny=self.extent.yMinimum(),
            maxy=self.extent.yMaximum(),
            mint=self.dataset.index.bounds[4],
            maxt=self.dataset.index.bounds[5],
        )
        return



    def create_output_dir(self,parameters):
        """
        compute parameters hash to have a unique identifier for the run
        some parameters do not change the encoding part of the algorithm
        """

        keys_to_remove = ["MERGE_METHOD", "WORKERS", "PAUSES"]
        subdir_hash = compute_md5_hash(parameters, keys_to_remove=keys_to_remove)

        self.output_subdir = Path(os.path.join(self.output_dir, subdir_hash))
        self.output_subdir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"output_subdir: {self.output_subdir}")

        self.logger.debug("saving parameters to json file")
        save_parameters_to_json(parameters, self.output_subdir)
        self.logger.debug("logging parameters to csv")
        log_parameters_to_csv(parameters, self.output_dir)
        return


    def merge_tiles_iteratively(self, feedback, dtype: str = "float32", nodata=None,):

        # tiles = sorted([
        #     os.path.join(self.output_subdir, f)
        #     for f in os.listdir(self.output_subdir)
        #     # if (f.endswith("_tmp.tif") or f.endswith("_wip.tif"))
        #     if f.endswith("_tmp.tif")
        # ])
        last_batch_done = self.get_last_batch_done()
        tiles = [
            os.path.join(self.output_subdir, f)
            for f in os.listdir(self.output_subdir)
            if f.endswith("_tmp.tif") and not f.startswith(str(last_batch_done))
        ]
        tiles = sorted([
            f for f in tiles if not f.endswith("merged_tmp.tif")
        ])

        rlayer_name, ext = os.path.splitext(self.rlayer_name)

        if not self.all_encoding_done:
            self.dst_path = Path(os.path.join(self.output_subdir, "merged_tmp.tif"))
            self.layer_name = f"{rlayer_name} features tmp"
        else:
            ## update filename if a merged.tif file allready exists
            dst_path, self.layer_name = get_unique_filename(self.output_subdir, "merged.tif", f"{rlayer_name} features")
            self.dst_path = Path(dst_path)

        print(tiles)
        # Iteratively merge the remaining rasters
        for i, tile in enumerate(tiles):
            self.logger.info(f"Merging raster {i+1}/{len(tiles)}")
            merge_tiles(
                    tiles[i:i+1],
                    self.dst_path,
                    dtype, 
                    nodata,
                    self.merge_method
                    )

            if feedback.isCanceled():
                feedback.pushWarning(self.tr("\n !!!Processing is canceled by user!!! \n"))
                return

        return


    def init_model(self):


        if '.yaml' in str(self.backbone_name):
            model, self.h, self.w = self.init_model_pangaea()
        else :
            model, self.h, self.w = self.init_model_timm()

        if not (self.ckpt_path == '' or self.ckpt_path == 'NULL') : 
            model.load_state_dict(torch.load(self.ckpt_path, weights_only=True))

        if self.quantization:
            try:
                self.logger.debug(f"Model size before quantization : {get_model_size(model)}")
                self.model = quantize_model(model, self.device)
                self.logger.debug(f"Model size after quantization : {get_model_size(model)}")

            except Exception:
                self.logger.debug("Quantization impossible, using original model.")
                self.model = model

        self.model.to(device=self.device)
        return


    def init_model_timm(self):
        model = timm.create_model(
            self.backbone_name,
            pretrained=True,
            in_chans=len(self.input_bands),
            num_classes=0,
        )
        self.logger.info("Model loaded succesfully !")

        data_config = timm.data.resolve_model_data_config(model)
        (_, h, w) = data_config["input_size"]
        return model, h, w

    def init_model_pangaea(self):

        cfg = OmegaConf.load(self.backbone_name)
        ## add cwd to path, otherwise hydra cannot find encoder classes
        ## cf. https://github.com/facebookresearch/hydra/issues/922
        ## and https://stackoverflow.com/a/53311583
        cfg.encoder_weights = os.path.join(self.cwd, str(cfg.encoder_weights))
        sys.path.append(str(self.cwd))
        model: Encoder = instantiate(cfg)
        model.load_encoder_weights(self.logger)
        model = modify_first_conv2d(model, in_chans=len(self.input_bands))
        return model, model.input_size, model.input_size

    def load_parameters_as_json(self, parameters):
        parameters["JSON_PARAM"] = str(parameters["JSON_PARAM"])
        json_param = parameters["JSON_PARAM"]
        print(json_param)
        if json_param != "NULL":
            with open(json_param) as json_file:
                parameters = json.load(json_file)
            self.logger.info(f"Loading previous parameters from {json_param}")
            parameters.pop("JSON_PARAM", None)
        else:
            parameters.pop("JSON_PARAM", None)

        return parameters

    def remove_temp_files(self):
        """
        cleaning up temp tiles
        keep last tiles and merged tiles in case of resume
        """

        last_batch_done = self.get_last_batch_done()
        if not self.all_encoding_done:
            tiles_to_remove = [
                os.path.join(self.output_subdir, f)
                for f in os.listdir(self.output_subdir)
                if f.endswith("_tmp.tif") and not f.startswith(str(last_batch_done))
            ]
            tiles_to_remove = [
                f for f in tiles_to_remove if not f.endswith("merged_tmp.tif")
            ]

        ## else cleanup all temp files
        else:
            tiles_to_remove = [
                os.path.join(self.output_subdir, f)
                for f in os.listdir(self.output_subdir)
                if f.endswith("_tmp.tif")
            ]

        remove_files(tiles_to_remove)

        return

    def get_last_batch_done(self):
        ## get largest batch_number achieved
        ## files are saved with the pattern '{batch_number}_{image_id_within_batch}_tmp.tif'
        # Regular expression pattern to extract numbers
        # pattern = re.compile(r'^(\d+)_\d+\.tif$')
        pattern = re.compile(r"^(\d+)_\d+_tmp\.tif$")

        # Initialize a set to store unique first numbers
        batch_numbers = set()

        # Iterate over all files in the directory
        for filename in os.listdir(self.output_subdir):
            # Match the filename pattern
            match = pattern.match(filename)
            if match:
                # Extract the batch number
                batch_number = int(match.group(1))
                # Add to the set of batch numbers
                batch_numbers.add(batch_number)

        # Find the maximum value of the batch numbers
        if batch_numbers:
            return max(batch_numbers)
        else:
            return -1

    def save_features(
        self,
        feature: np.ndarray,
        bboxes: BoundingBox,
        nbatch: int,
        dtype: str = "float32",
    ):
        if dtype == "int8":
            feature = (feature * 127).astype(np.int8)
        # iterate over batch_size dimension
        for idx in range(feature.shape[0]):
            _, height, width, channels = feature.shape
            bbox = bboxes[idx]
            rio_transform = rasterio.transform.from_bounds(
                bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height
            )  # west, south, east, north, width, height
            feature_path = os.path.join(self.output_subdir, f"{nbatch}_{idx}_tmp.tif")
            with rasterio.open(
                feature_path,
                mode="w",
                driver="GTiff",
                height=height,
                width=width,
                count=channels,
                dtype=dtype,
                crs=self.crs.toWkt(),
                transform=rio_transform,
            ) as ds:
                ds.write(np.transpose(feature[idx, ...], (2, 0, 1)))
                tags = {
                    "model_type": self.backbone_name,
                }
                ds.update_tags(**tags)

            self.iPatch += 1

        return

    def process_options(self, parameters, context):

        self.iPatch = 0
        self.feature_dir = ""
        self.logger.debug(f"PARAMETERS :\n{parameters}")

        self.process_geo_parameters(parameters, context)

        self.ckpt_path = self.parameterAsFile(parameters, self.CKPT, context)  # noqa: F841

        ## Use the given backbone name is any, use preselected models otherwise.
        input_name = self.parameterAsString(parameters, self.BACKBONE_CHOICE, context)

        if input_name:
            self.backbone_name = input_name
        else:
            backbone_idx = self.parameterAsEnum(parameters, self.BACKBONE_OPT, context)
            self.backbone_name = self.timm_backbone_opt[backbone_idx]
            self.logger.debug(f"self.backbone_name:{self.backbone_name}")

        self.compress = self.parameterAsBoolean(parameters, self.COMPRESS, context)
        self.stride = self.parameterAsInt(parameters, self.STRIDE, context)
        self.size = self.parameterAsInt(parameters, self.SIZE, context)
        self.quantization = self.parameterAsBoolean(parameters, self.QUANT, context)
        self.use_gpu = self.parameterAsBoolean(parameters, self.CUDA, context)
        self.batch_size = self.parameterAsInt(parameters, self.BATCH_SIZE, context)
        self.output_dir = self.parameterAsString(parameters, self.OUTPUT, context)
        self.cuda_id = self.parameterAsInt(parameters, self.CUDA_ID, context)
        self.pauses = self.parameterAsInt(parameters, self.PAUSES, context)
        self.cleanup_frq = self.parameterAsInt(
            parameters, self.TEMP_FILES_CLEANUP_FREQ, context
        )
        self.nworkers = self.parameterAsInt(parameters, self.WORKERS, context)
        merge_method_idx = self.parameterAsEnum(parameters, self.MERGE_METHOD, context)
        self.merge_method = self.merge_options[merge_method_idx]
        self.remove_tmp_files = self.parameterAsBoolean(
            parameters, self.REMOVE_TEMP_FILES, context
        )

        # get mean and sd of dataset from raster metadata
        self.logger.debug("Computing means and sds for normalization")
        means, sds = get_mean_sd_by_band(self.rlayer_path)
        # subset with selected_bands
        self.logger.debug(f"Selected bands: {self.selected_bands}")
        self.means = [means[i - 1] for i in self.selected_bands]
        self.sds = [sds[i - 1] for i in self.selected_bands]
        self.logger.debug(f"Means for normalization: {self.means}")
        self.logger.debug(f"Std. dev. for normalization: {self.sds}")

        if torch.cuda.is_available() and self.use_gpu:
            if self.cuda_id + 1 > torch.cuda.device_count():
                self.cuda_id = torch.cuda.device_count() - 1
            cuda_device = f"cuda:{self.cuda_id}"  # noqa: F841
            self.device = f"cuda:{self.cuda_id}"
        else:
            self.batch_size = 1
            self.device = "cpu"

        self.logger.debug(f"Device id: {self.device}")



    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        return {}

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return EncoderAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "encoder"

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("Image Encoder")

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr("")

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return ""

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Generate image features using a deep learning backbone.")

    def icon(self):
        return QIcon_EncoderTool
