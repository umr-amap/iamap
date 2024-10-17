import os
import logging
import sys
import time
import tempfile
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

import rasterio
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (Qgis,
                       QgsGeometry,
                       QgsCoordinateTransform,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
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
import torch.nn as nn
from torch import Tensor
import torch.quantization
from torch.utils.data import DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
import timm

# from torchgeo.datasets import RasterDataset, BoundingBox,stack_samples
# from torchgeo.samplers import GridGeoSampler, Units
# from torchgeo.transforms import AugmentationSequential
## from .utils.torchgeo import NoBordersGridGeoSampler
# from .utils.trchg import NoBordersGridGeoSampler

from .utils.geo import get_mean_sd_by_band
from .utils.geo import merge_tiles
from .utils.misc import (QGISLogHandler, 
                         get_dir_size, 
                         get_model_size, 
                         remove_files, 
                         check_disk_space,
                         get_unique_filename,
                         save_parameters_to_json,
                         compute_md5_hash,
                         log_parameters_to_csv,
                         )
from .utils.trch import quantize_model

from .tg.datasets import RasterDataset
from .tg.utils import stack_samples, BoundingBox
from .tg.samplers import NoBordersGridGeoSampler, Units
from .tg.transforms import AugmentationSequential




class EncoderAlgorithm(QgsProcessingAlgorithm):
    """
    """

    FEAT_OPTION= 'FEAT_OPTION'
    INPUT = 'INPUT'
    CKPT = 'CKPT'
    BANDS = 'BANDS'
    STRIDE = 'STRIDE'
    SIZE = 'SIZE'
    EXTENT = 'EXTENT'
    QUANT = 'QUANT'
    OUTPUT = 'OUTPUT'
    RESOLUTION = 'RESOLUTION'
    CRS = 'CRS'
    CUDA = 'CUDA'
    BATCH_SIZE = 'BATCH_SIZE'
    CUDA_ID = 'CUDA_ID'
    BACKBONE_CHOICE = 'BACKBONE_CHOICE'
    BACKBONE_OPT = 'BACKBONE_OPT'
    MERGE_METHOD = 'MERGE_METHOD'
    WORKERS = 'WORKERS'
    PAUSES = 'PAUSES'
    REMOVE_TEMP_FILES = 'REMOVE_TEMP_FILES'
    TEMP_FILES_CLEANUP_FREQ = 'TEMP_FILES_CLEANUP_FREQ'
    JSON_PARAM = 'JSON_PARAM'
    OUT_DTYPE = 'OUT_DTYPE'
    

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        cwd = Path(__file__).parent.absolute()
        tmp_wd = os.path.join(tempfile.gettempdir(), "iamap_features")

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr(
                    'Input raster layer or image file path'),
            defaultValue=os.path.join(cwd,'assets','test.tif'),
            ),
        )

        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BANDS,
                description=self.tr('Selected Bands (defaults to all bands selected)'),
                defaultValue = None, 
                parentLayerParameterName=self.INPUT,
                optional=True,
                allowMultiple=True,
            )
        )

        crs_param = QgsProcessingParameterCrs(
            name=self.CRS,
            description=self.tr('Target CRS (default to original CRS)'),
            optional=True,
        )

        res_param = QgsProcessingParameterNumber(
            name=self.RESOLUTION,
            description=self.tr(
                'Target resolution in meters (default to native resolution)'),
            type=QgsProcessingParameterNumber.Double,
            optional=True,
            minValue=0,
            maxValue=100000
        )

        cuda_id_param = QgsProcessingParameterNumber(
            name=self.CUDA_ID,
            description=self.tr(
                'CUDA Device ID (choose which GPU to use, default to device 0)'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=9
        )
        nworkers_param = QgsProcessingParameterNumber(
            name=self.WORKERS,
            description=self.tr(
                'Number of CPU workers for dataloader (0 selects all)'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=10
        )
        pauses_param = QgsProcessingParameterNumber(
            name=self.PAUSES,
            description=self.tr(
                'Schedule pauses between batches to ease CPU usage (in seconds).'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=10000
        )

        tmp_files_cleanup_frq = QgsProcessingParameterNumber(
            name=self.TEMP_FILES_CLEANUP_FREQ,
            description=self.tr(
                'Frequencie at which temporary files should be cleaned up (zero means no cleanup).'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1000,
            minValue=1,
            maxValue=10000
        )

        remove_tmp_files = QgsProcessingParameterBoolean(
            name=self.REMOVE_TEMP_FILES,
            description=self.tr(
                'Remove temporary files after encoding. If you want to test different merging options, it may be better to keep the tiles.'),
            defaultValue=True,
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr(
                    'Processing extent (default to the entire image)'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.SIZE,
                description=self.tr(
                    'Sampling size (the raster will be sampled in a square with a side of that many pixel)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue = 224,
                minValue=1,
                maxValue=1024
            )
        )


        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.STRIDE,
                description=self.tr(
                    'Stride (If smaller than the sampling size, tiles will overlap. If larger, it may cause errors.)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue = 224,
                minValue=1,
                maxValue=1024
            )
        )

        chkpt_param = QgsProcessingParameterFile(
                name=self.CKPT,
                description=self.tr(
                    'Pretrained checkpoint'),
                # extension='pth',
                fileFilter='Checkpoint Files (*.pth *.pkl);; All Files (*.*)',
                optional=True,
                defaultValue=None
            )
        

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr(
                    "Output directory (choose the location that the image features will be saved)"),
            defaultValue=tmp_wd,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CUDA,
                self.tr("Use GPU if CUDA is available."),
                defaultValue=True
            )
        )
        self.backbone_opt = [
                            'ViT base DINO',
                            'ViT tiny Imagenet (smallest)', 
                            'ViT base MAE', 
                            'SAM', 
                            '--Empty--'
                            ]
        self.timm_backbone_opt = [
                            'vit_base_patch16_224.dino',
                            'vit_tiny_patch16_224.augreg_in21k',
                            'vit_base_patch16_224.mae',
                            'samvit_base_patch16.sa1b',
                            ]
        self.addParameter (
            QgsProcessingParameterEnum(
                name = self.BACKBONE_OPT,
                description = self.tr(
                    "Pre-selected backbones if you don't know what to pick"),
                defaultValue = 0,
                options = self.backbone_opt,
                
            )
        )
        self.addParameter (
            QgsProcessingParameterString(
                name = self.BACKBONE_CHOICE,
                description = self.tr(
                    'Enter a architecture name if you want to test another backbone (see huggingface.co/timm/)'),
                defaultValue = None,
                optional=True,
            )
        )
        

        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FEAT_OPTION,
                self.tr("Display features map"),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BATCH_SIZE,
                # large images will be sampled into patches in a grid-like fashion
                description=self.tr(
                    'Batch size (take effect if choose to use GPU and CUDA is available)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
                maxValue=1024
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.QUANT,
                self.tr("Quantization of the model to reduce space"),
                defaultValue=True
            )
        )

        self.merge_options = ['first', 'min', 'max','average','sum', 'count', 'last']
        merge_param = QgsProcessingParameterEnum(
                name=self.MERGE_METHOD,
                description=self.tr(
                    'Merge method at the end of inference.'),
                options=self.merge_options,
                defaultValue=0,
                )

        self.out_dtype_opt = ['float32', 'int8']
        dtype_param = QgsProcessingParameterEnum(
                name=self.OUT_DTYPE,
                description=self.tr(
                    'Data type of exported features (int8 saves space)'),
                options=self.out_dtype_opt,
                defaultValue=0,
                )

        json_param = QgsProcessingParameterFile(
                name=self.JSON_PARAM,
                description=self.tr(
                    'Pass parameters as json file'),
                # extension='pth',
                fileFilter='JSON Files (*.json)',
                optional=True,
                defaultValue=None
            )

        for param in (
                crs_param, 
                res_param, 
                dtype_param,
                chkpt_param, 
                cuda_id_param, 
                merge_param, 
                nworkers_param,
                pauses_param,
                remove_tmp_files,
                tmp_files_cleanup_frq,
                json_param,
                ):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)



    @torch.no_grad()
    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        parameters = self.load_parameters_as_json(feedback, parameters)
        self.process_options(parameters, context, feedback)

        ## compute parameters hash to have a unique identifier for the run
        ## some parameters do not change the encoding part of the algorithm
        keys_to_remove = ['MERGE_METHOD', 'WORKERS', 'PAUSES']
        subdir_hash = compute_md5_hash(parameters, keys_to_remove=keys_to_remove)
        output_subdir = os.path.join(self.output_dir,subdir_hash)
        output_subdir = Path(output_subdir)
        output_subdir.mkdir(parents=True, exist_ok=True)
        self.output_subdir = output_subdir
        feedback.pushInfo(f'output_subdir: {output_subdir}')
        feedback.pushInfo(f'saving parameters to json file')
        save_parameters_to_json(parameters, self.output_subdir)
        feedback.pushInfo(f'logging parameters to csv')
        log_parameters_to_csv(parameters,self.output_dir)

        RasterDataset.filename_glob = self.rlayer_name
        RasterDataset.all_bands = [
            self.rlayer.bandName(i_band) for i_band in range(1, self.rlayer.bandCount()+1)
        ]
        # currently only support rgb bands
        input_bands = [self.rlayer.bandName(i_band)
                       for i_band in self.selected_bands]

        feedback.pushInfo(f'create dataset')
        if self.crs == self.rlayer.crs():
            dataset = RasterDataset(
                paths=self.rlayer_dir, crs=None, res=self.res, bands=input_bands, cache=False)
        else:
            dataset = RasterDataset(
                paths=self.rlayer_dir, crs=self.crs.toWkt(), res=self.res, bands=input_bands, cache=False)
        extent_bbox = BoundingBox(minx=self.extent.xMinimum(), maxx=self.extent.xMaximum(), miny=self.extent.yMinimum(), maxy=self.extent.yMaximum(),
                                  mint=dataset.index.bounds[4], maxt=dataset.index.bounds[5])


        if feedback.isCanceled():
            feedback.pushWarning(
                self.tr("\n !!!Processing is canceled by user!!! \n"))
            return


        ### Custom logging to have more feedback during model loading
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger()

        # Attach the QGIS log handler
        logger.addHandler(QGISLogHandler(feedback))

        # Log a message
        logger.info("Starting model loading...")

        # Load the model
        feedback.pushInfo(f'creating model')
        model = timm.create_model(
            self.backbone_name,
            pretrained=True,
            in_chans=len(input_bands),
            num_classes=0,
            )
        logger.info("Model loaded succesfully !")
        logger.handlers.clear()


        if feedback.isCanceled():
            feedback.pushWarning(
                self.tr("\n !!!Processing is canceled by user!!! \n"))
            return

        feedback.pushInfo(f'model done')
        data_config = timm.data.resolve_model_data_config(model)
        _, h, w, = data_config['input_size']

        if self.quantization:

            try :
                feedback.pushInfo(f'before quantization : {get_model_size(model)}')

                quantize_model(model, device)
                feedback.pushInfo(f'after quantization : {get_model_size(model)}')

            except :

                feedback.pushInfo(f'quantization impossible, using original model.')


        transform = AugmentationSequential(
                T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
                K.Normalize(self.means,self.sds), # normalize occurs only on raster, not mask
                K.Resize((h, w)),  # resize to 224*224 pixels, regardless of sampling size
                data_keys=["image"],
                )
        dataset.transforms = transform


        # sampler = GridGeoSampler(
        #         dataset, 
        #         size=self.size, 
        #         stride=self.stride, 
        #         roi=extent_bbox, 
        #         units=Units.PIXELS
        #         )  # Units.CRS or Units.PIXELS
        sampler = NoBordersGridGeoSampler(
                dataset, 
                size=self.size, 
                stride=self.stride, 
                roi=extent_bbox, 
                units=Units.PIXELS
                )  # Units.CRS or Units.PIXELS

        if len(sampler) == 0:
            self.load_feature = False
            feedback.pushWarning(f'\n !!!No available patch sample inside the chosen extent!!! \n')

        if torch.cuda.is_available() and self.use_gpu:
            if self.cuda_id + 1 > torch.cuda.device_count():
                self.cuda_id = torch.cuda.device_count() - 1
            cuda_device = f'cuda:{self.cuda_id}'
            device = f'cuda:{self.cuda_id}'
        else:
            self.batch_size = 1
            device = 'cpu'

        feedback.pushInfo(f'Device id: {device}')

        feedback.pushInfo(f'model to dedvice')
        model.to(device=device)

        feedback.pushInfo(f'Batch size: {self.batch_size}')
        dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                sampler=sampler, 
                collate_fn=stack_samples,
                num_workers=self.nworkers,
                )

        feedback.pushInfo(f'Patch sample num: {len(sampler)}')
        feedback.pushInfo(f'Total batch num: {len(dataloader)}')
        feedback.pushInfo(f'\n\n{"-"*16}\nBegining inference \n{"-"*16}\n\n')



        last_batch_done = self.get_last_batch_done()
        if last_batch_done >= 0:
            feedback.pushInfo(f"\n\n {'-'*8} \n Resuming at batch number {last_batch_done}\n {'-'*8} \n\n")

        bboxes = [] # keep track of bboxes to have coordinates at the end
        elapsed_time_list = []
        total = 100 / len(dataloader) if len(dataloader) else 0

        ## will update if process is canceled by the user
        self.all_encoding_done = True

        for current, sample in enumerate(dataloader):

            if current <= last_batch_done:
                continue

            start_time = time.time()

            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                self.load_feature = False
                feedback.pushWarning(
                    self.tr("\n !!!Processing is canceled by user!!! \n"))
                self.all_encoding_done = False
                break
            
            feedback.pushInfo(f'\n{"-"*8}\nBatch no. {current} loaded')

            images = sample['image'].to(device)
            if len(images.shape) > 4:
                images = images.squeeze(1)
            
            feedback.pushInfo(f'Batch shape {images.shape}')

            features = model.forward_features(images)
            features = features[:,1:,:] # take only patch tokens
            
            if current <= last_batch_done + 1:
                n_patches = int(np.sqrt(features.shape[1]))   

            features = features.view(features.shape[0],n_patches,n_patches,features.shape[-1])
            features = features.detach().cpu().numpy()
            feedback.pushInfo(f'Features shape {features.shape}')

            self.save_features(features,sample['bbox'], current,dtype=self.out_dtype)
            feedback.pushInfo(f'Features saved')

            if current <= last_batch_done + 1:
                total_space, total_used_space, free_space = check_disk_space(self.output_subdir)

                used_outputsubdir = get_dir_size(str(self.output_subdir))
                
                to_use = ((len(dataloader) / (current+1)) - 1) * used_outputsubdir
                if to_use >= free_space:
                    feedback.pushWarning(
                        self.tr(f"\n !!! only {free_space} GB disk space remaining, canceling !!! \n"))
                    break

            bboxes.extend(sample['bbox'])

            if self.pauses != 0:
                time.sleep(self.pauses)

            end_time = time.time()
            # get the execution time of encoder, ms
            elapsed_time = (end_time - start_time)
            elapsed_time_list.append(elapsed_time)
            time_spent = sum(elapsed_time_list)
            time_remain = (time_spent / (current + 1)) * (len(dataloader) - current - 1)

            # TODO: show gpu usage info
            # if torch.cuda.is_available() and self.use_gpu:
            #     gpu_mem_used = torch.cuda.max_memory_reserved(self.sam_model.device) / (1024 ** 3)
            #     # gpu_mem_free = torch.cuda.mem_get_info(self.sam_model.device)[0] / (1024 ** 3)
            #     gpu_mem_total = torch.cuda.mem_get_info(self.sam_model.device)[1] / (1024 ** 3)
            #     feedback.pushInfo(
            #         f'GPU memory usage: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB')
            #     feedback.pushInfo(str(torch.cuda.memory_summary(self.sam_model.device)))

            feedback.pushInfo(f"Encoder executed with {elapsed_time:.3f}s")
            feedback.pushInfo(f"Time spent: {time_spent:.3f}s")
                  
            if time_remain <= 60:
                feedback.pushInfo(f"Estimated time remaining: {time_remain:.3f}s \n {'-'*8}")
            else:
                time_remain_m, time_remain_s = divmod(int(time_remain), 60)
                time_remain_h, time_remain_m = divmod(time_remain_m, 60)
                feedback.pushInfo(f"Estimated time remaining: {time_remain_h:d}h:{time_remain_m:02d}m:{time_remain_s:02d}s \n" )

            if ((current + 1) % self.cleanup_frq == 0) and self.remove_tmp_files:

                ## not the cleanest way to do for now
                ## but avoids to refactor all
                self.all_encoding_done = False
                feedback.pushInfo('Cleaning temporary files...')
                all_tiles = [os.path.join(self.output_subdir,f) for f in os.listdir(self.output_subdir) if f.endswith('_tmp.tif')]
                all_tiles = [f for f in all_tiles if not f.startswith('merged')]

                dst_path = Path(os.path.join(self.output_subdir,'merged_tmp.tif'))

                merge_tiles(
                        tiles = all_tiles, 
                        dst_path = dst_path,
                        method = self.merge_method,
                        dtype= self.out_dtype,
                        )
                self.remove_temp_files()
                self.all_encoding_done = True

            # Update the progress bar
            feedback.setProgress(int((current+1) * total))


        ## merging all temp tiles
        feedback.pushInfo(f"\n\n{'-'*8}\n Merging tiles \n{'-'*8}\n" )
        all_tiles = [os.path.join(self.output_subdir,f) for f in os.listdir(self.output_subdir) if f.endswith('_tmp.tif')]
        rlayer_name, ext = os.path.splitext(self.rlayer_name)

        if not self.all_encoding_done :
            dst_path = Path(os.path.join(self.output_subdir,'merged_tmp.tif'))
            layer_name = f'{rlayer_name} features tmp'
        else:
            # dst_path = Path(os.path.join(self.output_subdir,'merged.tif'))
            ## update filename if a merged.tif file allready exists
            
            dst_path, layer_name = get_unique_filename(self.output_subdir, f'merged.tif', f'{rlayer_name} features')
            dst_path = Path(dst_path)

        merge_tiles(
                tiles = all_tiles, 
                dst_path = dst_path,
                method = self.merge_method,
                )

        if self.remove_tmp_files:

            self.remove_temp_files()

        parameters['OUTPUT_RASTER']=dst_path

        return {"Output feature path": self.output_subdir, 'Patch samples saved': self.iPatch, 'OUTPUT_RASTER':dst_path, 'OUTPUT_LAYER_NAME':layer_name}

    def load_parameters_as_json(self, feedback, parameters):
        parameters['JSON_PARAM'] = str(parameters['JSON_PARAM'])
        json_param = parameters['JSON_PARAM']
        print(json_param)
        if json_param != 'NULL':
            with open(json_param) as json_file:
                parameters = json.load(json_file)
            feedback.pushInfo(f'Loading previous parameters from {json_param}')
            parameters.pop('JSON_PARAM',None)
        else:
            parameters.pop('JSON_PARAM',None)
        
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
                    if f.endswith('_tmp.tif') and not f.startswith(str(last_batch_done))
                    ]
            tiles_to_remove = [
                    f for f in tiles_to_remove
                    if not f.endswith('merged_tmp.tif')
                    ]

        ## else cleanup all temp files
        else : 
            tiles_to_remove = [os.path.join(self.output_subdir, f)
                 for f in os.listdir(self.output_subdir)
                 if f.endswith('_tmp.tif')]

        remove_files(tiles_to_remove)

        return

    def get_last_batch_done(self):

        ## get largest batch_number achieved
        ## files are saved with the pattern '{batch_number}_{image_id_within_batch}_tmp.tif'
        # Regular expression pattern to extract numbers
        # pattern = re.compile(r'^(\d+)_\d+\.tif$')
        pattern = re.compile(r'^(\d+)_\d+_tmp\.tif$')

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
            dtype: str = 'float32'
            ):

        if dtype == 'int8':
            feature = (feature * 127).astype(np.int8)
        # iterate over batch_size dimension
        for idx in range(feature.shape[0]):
            _, height, width, channels = feature.shape
            bbox = bboxes[idx]
            rio_transform = rasterio.transform.from_bounds(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height)  # west, south, east, north, width, height
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
                    transform=rio_transform
            ) as ds:
                ds.write(np.transpose(feature[idx, ...], (2, 0, 1)))
                tags = {
                    "model_type": self.backbone_name,
                }
                ds.update_tags(**tags)

            self.iPatch += 1

        return

    def process_options(self,parameters, context, feedback):
        self.iPatch = 0
        
        self.feature_dir = ""
        
        self.FEAT_OPTION = self.parameterAsBoolean(
            parameters, self.FEAT_OPTION, context)
        
        feedback.pushInfo(
                f'PARAMETERS :\n{parameters}')
        
        feedback.pushInfo(
                f'CONTEXT :\n{context}')
        
        feedback.pushInfo(
                f'FEEDBACK :\n{feedback}')

        rlayer = self.parameterAsRasterLayer(
            parameters, self.INPUT, context)
        
        if rlayer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT))

        self.selected_bands = self.parameterAsInts(
            parameters, self.BANDS, context)

        if len(self.selected_bands) == 0:
            self.selected_bands = list(range(1, rlayer.bandCount()+1))

        if max(self.selected_bands) > rlayer.bandCount():
            raise QgsProcessingException(
                self.tr("The chosen bands exceed the largest band number!")
            )

        ckpt_path = self.parameterAsFile(
            parameters, self.CKPT, context)


        ## Use the given backbone name is any, use preselected models otherwise.
        input_name = self.parameterAsString(
            parameters, self.BACKBONE_CHOICE, context)
        
        if input_name:
            self.backbone_name = input_name
        else:
            backbone_idx = self.parameterAsEnum(
                parameters, self.BACKBONE_OPT, context)
            self.backbone_name = self.timm_backbone_opt[backbone_idx]
            feedback.pushInfo(f'self.backbone_name:{self.backbone_name}')

        dtype_idx = self.parameterAsEnum(
            parameters, self.OUT_DTYPE, context)
        self.out_dtype = self.out_dtype_opt[dtype_idx]

        self.stride = self.parameterAsInt(
            parameters, self.STRIDE, context)
        self.size = self.parameterAsInt(
            parameters, self.SIZE, context)
        res = self.parameterAsDouble(
            parameters, self.RESOLUTION, context)
        crs = self.parameterAsCrs(
            parameters, self.CRS, context)
        extent = self.parameterAsExtent(
            parameters, self.EXTENT, context)
        self.quantization = self.parameterAsBoolean(
            parameters, self.QUANT, context)
        self.use_gpu = self.parameterAsBoolean(
            parameters, self.CUDA, context)
        self.batch_size = self.parameterAsInt(
            parameters, self.BATCH_SIZE, context)
        self.output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)
        self.cuda_id = self.parameterAsInt(
            parameters, self.CUDA_ID, context)
        self.pauses = self.parameterAsInt(
            parameters, self.PAUSES, context)
        self.cleanup_frq = self.parameterAsInt(
            parameters, self.TEMP_FILES_CLEANUP_FREQ, context)
        self.nworkers = self.parameterAsInt(
            parameters, self.WORKERS, context)
        merge_method_idx = self.parameterAsEnum(
            parameters, self.MERGE_METHOD, context)
        self.merge_method = self.merge_options[merge_method_idx]
        self.remove_tmp_files = self.parameterAsBoolean(
            parameters, self.REMOVE_TEMP_FILES, context)

        rlayer_data_provider = rlayer.dataProvider()

        # handle crs
        if crs is None or not crs.isValid():
            crs = rlayer.crs()
            feedback.pushInfo(
                f'Layer CRS unit is {crs.mapUnits()}')  # 0 for meters, 6 for degrees, 9 for unknown
            feedback.pushInfo(
                f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
            if crs.mapUnits() == Qgis.DistanceUnit.Degrees:
                crs = self.estimate_utm_crs(rlayer.extent())

        # target crs should use meters as units
        if crs.mapUnits() != Qgis.DistanceUnit.Meters:
            feedback.pushInfo(
                f'Layer CRS unit is {crs.mapUnits()}')
            feedback.pushInfo(
                f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
            raise QgsProcessingException(
                self.tr("Only support CRS with the units as meters")
            )

        # 0 for meters, 6 for degrees, 9 for unknown
        UNIT_METERS = 0
        UNIT_DEGREES = 6
        if rlayer.crs().mapUnits() == UNIT_DEGREES: # Qgis.DistanceUnit.Degrees:
            layer_units = 'degrees'
        else:
            layer_units = 'meters'
        # if res is not provided, get res info from rlayer
        if np.isnan(res) or res == 0:
            res = rlayer.rasterUnitsPerPixelX()  # rasterUnitsPerPixelY() is negative
            target_units = layer_units
        else:
            # when given res in meters by users, convert crs to utm if the original crs unit is degree
            if crs.mapUnits() != UNIT_METERS: # Qgis.DistanceUnit.Meters:
                if rlayer.crs().mapUnits() == UNIT_DEGREES: # Qgis.DistanceUnit.Degrees:
                    # estimate utm crs based on layer extent
                    crs = self.estimate_utm_crs(rlayer.extent())
                else:
                    raise QgsProcessingException(
                        f"Resampling of image with the CRS of {crs.authid()} in meters is not supported.")
            target_units = 'meters'
            # else:
            #     res = (rlayer_extent.xMaximum() -
            #            rlayer_extent.xMinimum()) / rlayer.width()
        self.res = res

        # handle extent
        if extent.isNull():
            extent = rlayer.extent()  # QgsProcessingUtils.combineLayerExtents(layers, crs, context)
            extent_crs = rlayer.crs()
        else:
            if extent.isEmpty():
                raise QgsProcessingException(
                    self.tr("The extent for processing can not be empty!"))
            extent_crs = self.parameterAsExtentCrs(
                parameters, self.EXTENT, context)
        # if extent crs != target crs, convert it to target crs
        if extent_crs != crs:
            transform = QgsCoordinateTransform(
                extent_crs, crs, context.transformContext())
            # extent = transform.transformBoundingBox(extent)
            # to ensure coverage of the transformed extent
            # convert extent to polygon, transform polygon, then get boundingBox of the new polygon
            extent_polygon = QgsGeometry.fromRect(extent)
            extent_polygon.transform(transform)
            extent = extent_polygon.boundingBox()
            extent_crs = crs

        # check intersects between extent and rlayer_extent
        if rlayer.crs() != crs:
            transform = QgsCoordinateTransform(
                rlayer.crs(), crs, context.transformContext())
            rlayer_extent = transform.transformBoundingBox(
                rlayer.extent())
        else:
            rlayer_extent = rlayer.extent()
        if not rlayer_extent.intersects(extent):
            raise QgsProcessingException(
                self.tr("The extent for processing is not intersected with the input image!"))

        feedback.pushInfo(f'backbne type : {self.backbone_name}')
        
        img_width_in_extent = round(
            (extent.xMaximum() - extent.xMinimum())/self.res)
        img_height_in_extent = round(
            (extent.yMaximum() - extent.yMinimum())/self.res)

        # Send some information to the user
        feedback.pushInfo(
            f'Layer path: {rlayer_data_provider.dataSourceUri()}')
        # feedback.pushInfo(
        #     f'Layer band scale: {rlayer_data_provider.bandScale(self.selected_bands[0])}')
        feedback.pushInfo(f'Layer name: {rlayer.name()}')
        if rlayer.crs().authid():
            feedback.pushInfo(f'Layer CRS: {rlayer.crs().authid()}')
        else:
            feedback.pushInfo(
                f'Layer CRS in WKT format: {rlayer.crs().toWkt()}')
        feedback.pushInfo(
            f'Layer pixel size: {rlayer.rasterUnitsPerPixelX()}, {rlayer.rasterUnitsPerPixelY()} {layer_units}')

        feedback.pushInfo(f'Bands selected: {self.selected_bands}')

        if crs.authid():
            feedback.pushInfo(f'Target CRS: {crs.authid()}')
        else:
            feedback.pushInfo(f'Target CRS in WKT format: {crs.toWkt()}')
        # feedback.pushInfo('Band number is {}'.format(rlayer.bandCount()))
        # feedback.pushInfo('Band name is {}'.format(rlayer.bandName(1)))
        feedback.pushInfo(f'Target resolution: {self.res} {target_units}')
        # feedback.pushInfo('Layer display band name is {}'.format(
        #     rlayer.dataProvider().displayBandName(1)))
        feedback.pushInfo(
            (f'Processing extent: minx:{extent.xMinimum():.6f}, maxx:{extent.xMaximum():.6f},'
             f'miny:{extent.yMinimum():.6f}, maxy:{extent.yMaximum():.6f}'))
        feedback.pushInfo(
            (f'Processing image size: (width {img_width_in_extent}, '
             f'height {img_height_in_extent})'))

        # feedback.pushInfo(
        #     f'SAM Image Size: {self.sam_model.image_encoder.img_size}')

        self.rlayer_path = rlayer.dataProvider().dataSourceUri()
        self.rlayer_dir = os.path.dirname(self.rlayer_path)
        self.rlayer_name = os.path.basename(self.rlayer_path)

        # get mean and sd of dataset from raster metadata
        feedback.pushInfo(f'Computing means and sds for normalization')
        means, sds = get_mean_sd_by_band(self.rlayer_path)
        # subset with selected_bands
        feedback.pushInfo(f'Selected bands: {self.selected_bands}')
        self.means = [means[i-1] for i in self.selected_bands]
        self.sds = [sds[i-1] for i in self.selected_bands]
        feedback.pushInfo(f'Means for normalization: {self.means}')
        feedback.pushInfo(f'Std. dev. for normalization: {self.sds}')

        ## passing parameters to self once everything has been processed
        self.extent = extent
        self.rlayer = rlayer
        self.crs = crs


    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        return {}


    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

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
        return 'encoder'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Image Encoder')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return ''

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Generate image features using a deep learning backbone.")

    def icon(self):
        return 'E'

