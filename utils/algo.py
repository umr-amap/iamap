import os
import ast
import tempfile
import numpy as np
import inspect
import joblib
from collections import Counter
from pathlib import Path
from typing import Dict, Any
from qgis.core import (Qgis,
                       QgsGeometry,
                       QgsCoordinateTransform,
                       QgsProcessingException,
                       QgsProcessingAlgorithm, QgsProcessingParameterBoolean,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterBand,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterString,
                       QgsProcessingParameterCrs,
                       QgsProcessingParameterDefinition, QgsProcessingParameterVectorLayer,
                       )
import rasterio
from rasterio import windows
from rasterio.enums import Resampling

import geopandas as gpd
from shapely.geometry import box

import torch
import torch.nn as nn

import sklearn.decomposition as decomposition
import sklearn.cluster as cluster
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score, silhouette_samples


if __name__ != "__main__":
    from .misc import get_unique_filename, calculate_chunk_size
    from .geo import get_random_samples_in_gdf


def get_sklearn_algorithms_with_methods(module, required_methods):
    # Get all classes in the module that are subclasses of BaseEstimator
    algorithms = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseEstimator) and not name.startswith('_'):
            # Check if the class has all the required methods
            if all(hasattr(obj, method) for method in required_methods):
                algorithms.append(name)
    return algorithms

def instantiate_sklearn_algorithm(module, algorithm_name, **kwargs):
    # Retrieve the class from the module by name
    AlgorithmClass = getattr(module, algorithm_name)
    # Instantiate the class with the provided parameters
    return AlgorithmClass(**kwargs)


def get_arguments(module, algorithm_name):
    AlgorithmClass = getattr(module, algorithm_name)
    # Get the signature of the __init__ method
    init_signature = inspect.signature(AlgorithmClass.__init__)
    
    # Retrieve the parameters of the __init__ method
    parameters = init_signature.parameters
    default_kwargs = {}
    
    for param_name, param in parameters.items():
        # Skip 'self'
        if param_name != 'self':
            # if param.default == None:
            #     required_kwargs[param_name] = None  # Placeholder for the required value
            # else:
            default_kwargs[param_name] = param.default
    
    # return required_kwargs, default_kwargs
    return default_kwargs


def get_iter(model, fit_raster):

    iter = None
    if hasattr(model, 'partial_fit') and hasattr(model, 'max_iter'):
        iter = range(model.max_iter)

    if hasattr(model, 'partial_fit') and not hasattr(model, 'max_iter'):
        chunk_size = calculate_chunk_size(fit_raster)
        iter = range(0, len(fit_raster), chunk_size) 

    return iter


class IAMAPAlgorithm(QgsProcessingAlgorithm):
    """
    """

    INPUT = 'INPUT'
    BANDS = 'BANDS'
    EXTENT = 'EXTENT'
    OUTPUT = 'OUTPUT'
    RESOLUTION = 'RESOLUTION'
    RANDOM_SEED = 'RANDOM_SEED'
    CRS = 'CRS'
    COMPRESS = 'COMPRESS'
    TMP_DIR = 'iamap_tmp'
    

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.init_input_raster()
        self.init_seed()

        compress_param = QgsProcessingParameterBoolean(
            name=self.COMPRESS,
            description=self.tr(
                'Compress final result to JP2'),
            defaultValue=True,
            optional=True,
        )

        for param in (
                compress_param,
                ):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)


    def init_input_output_raster(self):
        self.cwd = Path(__file__).parent.absolute()
        tmp_wd = os.path.join(tempfile.gettempdir(), self.TMP_DIR)

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr(
                    'Input raster layer or image file path'),
            # defaultValue=os.path.join(self.cwd,'assets','test.tif'),
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

        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr(
                    'Processing extent (default to the entire image)'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr(
                    "Output directory (choose the location that the image features will be saved)"),
            defaultValue=tmp_wd,
            )
        )

        for param in (
                crs_param, 
                res_param, 
                ):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)

    def init_seed(self):
        seed_param = QgsProcessingParameterNumber(
            name=self.RANDOM_SEED,
            description=self.tr(
                'Random seed'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=42,
            minValue=0,
            maxValue=100000
        )
        seed_param.setFlags(
            seed_param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(seed_param)



    def process_geo_parameters(self,parameters, context, feedback):
        """
        Handle geographic parameters that are common to all algorithms (CRS, resolution, extent, selected bands).
        """

        rlayer = self.parameterAsRasterLayer(
            parameters, self.INPUT, context)
        
        if rlayer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT))

        self.rlayer_path = rlayer.dataProvider().dataSourceUri()
        self.rlayer_dir = os.path.dirname(self.rlayer_path)
        self.rlayer_name = os.path.basename(self.rlayer_path)

        self.selected_bands = self.parameterAsInts(
            parameters, self.BANDS, context)

        if len(self.selected_bands) == 0:
            self.selected_bands = list(range(1, rlayer.bandCount()+1))

        if max(self.selected_bands) > rlayer.bandCount():
            raise QgsProcessingException(
                self.tr("The chosen bands exceed the largest band number!")
            )
        res = self.parameterAsDouble(
            parameters, self.RESOLUTION, context)
        crs = self.parameterAsCrs(
            parameters, self.CRS, context)
        extent = self.parameterAsExtent(
            parameters, self.EXTENT, context)

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
        # if res is not provided, get res info from rlayer
        if np.isnan(res) or res == 0:
            res = rlayer.rasterUnitsPerPixelX()  # rasterUnitsPerPixelY() is negative
        else:
            # when given res in meters by users, convert crs to utm if the original crs unit is degree
            if crs.mapUnits() != UNIT_METERS: # Qgis.DistanceUnit.Meters:
                if rlayer.crs().mapUnits() == UNIT_DEGREES: # Qgis.DistanceUnit.Degrees:
                    # estimate utm crs based on layer extent
                    crs = self.estimate_utm_crs(rlayer.extent())
                else:
                    raise QgsProcessingException(
                        f"Resampling of image with the CRS of {crs.authid()} in meters is not supported.")
            # else:
            #     res = (rlayer_extent.xMaximum() -
            #            rlayer_extent.xMinimum()) / rlayer.width()

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


        img_width_in_extent = round(
            (extent.xMaximum() - extent.xMinimum())/res)
        img_height_in_extent = round(
            (extent.yMaximum() - extent.yMinimum())/res)

        feedback.pushInfo(
            (f'Processing extent: minx:{extent.xMinimum():.6f}, maxx:{extent.xMaximum():.6f},'
             f'miny:{extent.yMinimum():.6f}, maxy:{extent.yMaximum():.6f}'))
        feedback.pushInfo(
            (f'Processing image size: (width {img_width_in_extent}, '
             f'height {img_height_in_extent})'))

        # Send some information to the user
        feedback.pushInfo(
            f'Layer path: {rlayer.dataProvider().dataSourceUri()}')
        # feedback.pushInfo(
        #     f'Layer band scale: {rlayer_data_provider.bandScale(self.selected_bands[0])}')
        feedback.pushInfo(f'Layer name: {rlayer.name()}')

        feedback.pushInfo(f'Bands selected: {self.selected_bands}')

        self.extent = extent
        self.rlayer = rlayer
        self.crs = crs
        self.res = res

    def tiff_to_jp2(self, parameters, feedback):
        """
        Compress final file to JP2.
        """
        
        feedback.pushInfo(f'Compressing to JP2')

        file = parameters['OUTPUT_RASTER']
        dst_path = Path(file).with_suffix('.jp2')

        ## update in the parameters
        parameters['OUTPUT_RASTER'] = dst_path

        with rasterio.open(file) as src:
            # Read the data
            float_data = src.read(resampling=Resampling.nearest)
    
            # Initialize an array for the normalized uint16 data
            uint16_data = np.empty_like(float_data, dtype=np.uint16)
            
            # Loop through each band to normalize individually
            for i in range(float_data.shape[0]):
                band = float_data[i]
                
                # Find min and max of the current band
                band_min = np.min(band)
                band_max = np.max(band)
                
                # Normalize to the range [0, 1]
                normalized_band = (band - band_min) / (band_max - band_min)
                
                # Scale to the uint16 range [0, 65535]
                uint16_data[i] = (normalized_band * 65535).astype(np.uint16)
            
            # Define metadata for the output JP2
            profile = src.profile
            profile.update(
                driver='JP2OpenJPEG',   # Specify JP2 driver
                dtype='uint16',        # Keep data as float32
                compress='jp2',         # Compression type (note: might be driver-specific)
                crs=src.crs,            # Coordinate system
                transform=src.transform # Affine transform
            )
            # profile.update(tiled=False)
            profile.update(tiled=True, blockxsize=256, blockysize=256)

            # Write to JP2 file
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(uint16_data)

        return dst_path

    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        return {}



class SKAlgorithm(IAMAPAlgorithm):
    """
    Common class that handles helper functions for sklearn algorithms.
    Behaviour defaults to projection algorithms (PCA etc...)
    """

    LOAD = 'LOAD'
    OUTPUT = 'OUTPUT'
    MAIN_PARAM = 'MAIN_PARAM'
    SUBSET = 'SUBSET'
    METHOD = 'METHOD'
    SAVE_MODEL = 'SAVE_MODEL'
    COMPRESS = 'COMPRESS'
    SK_PARAM = 'SK_PARAM'
    TMP_DIR = 'iamap_reduction'
    TYPE = 'proj'
    

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.init_input_output_raster()
        self.init_seed()

        proj_methods = ['fit', 'transform']
        clust_methods = ['fit', 'fit_predict']
        if self.TYPE == 'proj':
            method_opt1 = get_sklearn_algorithms_with_methods(decomposition, proj_methods)
            method_opt2 = get_sklearn_algorithms_with_methods(cluster, proj_methods)
            self.method_opt = method_opt1 + method_opt2

            self.addParameter(
                QgsProcessingParameterNumber(
                    name=self.MAIN_PARAM,
                    description=self.tr(
                        'Number of target components'),
                    type=QgsProcessingParameterNumber.Integer,
                    defaultValue = 3,
                    minValue=1,
                    maxValue=1024
                )
            )
            default_index = self.method_opt.index('PCA')
        else :
            self.method_opt = get_sklearn_algorithms_with_methods(cluster, clust_methods)
            self.addParameter(
                QgsProcessingParameterNumber(
                    name=self.MAIN_PARAM,
                    description=self.tr(
                        'Number of target clusters'),
                    type=QgsProcessingParameterNumber.Integer,
                    defaultValue = 3,
                    minValue=1,
                    maxValue=1024
                )
            )
            default_index = self.method_opt.index('KMeans')

        self.addParameter (
            QgsProcessingParameterEnum(
                name = self.METHOD,
                description = self.tr(
                    'Sklearn algorithm used'),
                defaultValue = default_index,
                options = self.method_opt,
            )
        )

        self.addParameter (
            QgsProcessingParameterString(
                name = self.SK_PARAM,
                description = self.tr(
                    'Arguments for the initialisation of the algorithm. If empty this goes to sklearn default. It will overwrite cluster or components arguments.'),
                defaultValue = '',
                optional=True,
            )
        )

        subset_param = QgsProcessingParameterNumber(
                name=self.SUBSET,
                description=self.tr(
                    'Select a subset of random pixels of the image to fit transform'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=None,
                minValue=1,
                maxValue=10_000,
                optional=True,
                )

        save_param = QgsProcessingParameterBoolean(
                self.SAVE_MODEL,
                self.tr("Save projection model after fit."),
                defaultValue=True
                )

        for param in (
                subset_param, 
                save_param
                ):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)


    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        self.process_geo_parameters(parameters, context, feedback)
        self.process_common_sklearn(parameters, context)

        fit_raster, scaler = self.get_fit_raster(feedback)

        rlayer_basename = os.path.basename(self.rlayer_path)
        rlayer_name, ext = os.path.splitext(rlayer_basename)

        ### Handle args that differ between clustering and projection methods
        if self.TYPE == 'proj':
            self.out_dtype = 'float32'
            self.dst_path, self.layer_name = get_unique_filename(self.output_dir, f'{self.TYPE}.tif', f'{rlayer_name} reduction')
        else:
            self.out_dtype = 'uint8'
            self.dst_path, self.layer_name = get_unique_filename(self.output_dir, f'{self.TYPE}.tif', f'{rlayer_name} cluster')

        parameters['OUTPUT_RASTER']=self.dst_path

        try:
            default_args = get_arguments(decomposition, self.method_name)
        except AttributeError:
            default_args = get_arguments(cluster, self.method_name)

        kwargs = self.update_kwargs(default_args)

        ## some clustering algorithms need the entire dataset.
        do_fit_predict = False

        try:
            model = instantiate_sklearn_algorithm(decomposition, self.method_name, **kwargs)
        except AttributeError:
            model = instantiate_sklearn_algorithm(cluster, self.method_name, **kwargs)
            ## if model does not have a 'predict()' method, then we do a fit_predict in one go 
            if not hasattr(model, 'predict'):
                do_fit_predict = True
        except:
            feedback.pushWarning(f'{self.method_name} not properly initialized ! Try passing custom parameters')
            return {'OUTPUT_RASTER':self.dst_path, 'OUTPUT_LAYER_NAME':self.layer_name}

        if do_fit_predict:
            proj_img, model = self.fit_predict(model, feedback)

        else:
            iter = get_iter(model, fit_raster)
            model = self.fit_model(model, fit_raster, iter, feedback)

        self.print_transform_metrics(model, feedback)
        self.print_cluster_metrics(model,fit_raster, feedback)
        feedback.pushInfo(f'Fitting done, saving model\n')
        save_file = f'{self.method_name}.pkl'.lower()
        if self.save_model:
            out_path = os.path.join(self.output_dir, save_file)
            joblib.dump(model, out_path)

        if not do_fit_predict:
            feedback.pushInfo(f'Inference over raster\n')
            self.infer_model(model, feedback, scaler)

        return {'OUTPUT_RASTER':self.dst_path, 'OUTPUT_LAYER_NAME':self.layer_name}


    def process_common_sklearn(self,parameters, context):

        self.subset = self.parameterAsInt(
            parameters, self.SUBSET, context)
        self.seed = self.parameterAsInt(
            parameters, self.RANDOM_SEED, context)
        self.main_param = self.parameterAsInt(
            parameters, self.MAIN_PARAM, context)
        self.subset = self.parameterAsInt(
            parameters, self.SUBSET, context)

        method_idx = self.parameterAsEnum(
            parameters, self.METHOD, context)
        self.method_name = self.method_opt[method_idx]

        str_kwargs = self.parameterAsString(
                parameters, self.SK_PARAM, context)
        if str_kwargs != '':
            self.passed_kwargs = ast.literal_eval(str_kwargs)
        else:
            self.passed_kwargs = {}

        self.input_bands = [i_band -1 for i_band in self.selected_bands]

        self.save_model = self.parameterAsBoolean(
                parameters, self.SAVE_MODEL, context)
        output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def get_fit_raster(self, feedback):

        with rasterio.open(self.rlayer_path) as ds:

            transform = ds.transform
            win = windows.from_bounds(
                    self.extent.xMinimum(), 
                    self.extent.yMinimum(), 
                    self.extent.xMaximum(), 
                    self.extent.yMaximum(), 
                    transform=transform
                    )
            raster = ds.read(window=win)
            transform = ds.window_transform(win)
            raster = np.transpose(raster, (1,2,0))
            raster = raster[:,:,self.input_bands]
            fit_raster = raster.reshape(-1, raster.shape[-1])
            scaler = StandardScaler()
            scaler.fit(fit_raster)

            if self.subset:

                feedback.pushInfo(f'Using a random subset of {self.subset} pixels, random seed is {self.seed}')

                fit_raster = raster.reshape(-1, raster.shape[-1])
                nsamples = fit_raster.shape[0]
    
                # Generate random indices to select subset_size number of samples
                np.random.seed(self.seed)
                random_indices = np.random.choice(nsamples, size=self.subset, replace=False)
                fit_raster = fit_raster[random_indices,:]

                # remove nans
                fit_raster = fit_raster[~np.isnan(fit_raster).any(axis=1)]

            fit_raster = scaler.transform(fit_raster)
            np.nan_to_num(fit_raster) # NaN to zero after normalisation

        return fit_raster, scaler

    def fit_model(self, model, fit_raster, iter,feedback):

        feedback.pushInfo(f'Starting fit. If it goes for too long, consider setting a subset.\n')

        ## if fitting can be divided, we provide the possibility to cancel and to have progression
        if iter and hasattr(model, 'partial_fit'):
            for i in iter:
                if feedback.isCanceled():
                    feedback.pushWarning(
                        self.tr("\n !!!Processing is canceled by user!!! \n"))
                    break
                model.partial_fit(fit_raster)
                feedback.setProgress((i / len(iter)) * 100)

        ## else, all in one go
        else:
            model.fit(fit_raster)

        return model

    def fit_predict(self, model, feedback):

        with rasterio.open(self.rlayer_path) as ds:

            transform = ds.transform
            crs = ds.crs
            win = windows.from_bounds(
                    self.extent.xMinimum(), 
                    self.extent.yMinimum(), 
                    self.extent.xMaximum(), 
                    self.extent.yMaximum(), 
                    transform=transform
                    )
            raster = ds.read(window=win)
            transform = ds.window_transform(win)
            raster = np.transpose(raster, (1,2,0))
            raster = raster[:,:,self.input_bands]


            # raster = (raster-np.mean(raster))/np.std(raster)
            scaler = StandardScaler()

            raster = scaler.fit_transform(raster)
            np.nan_to_num(raster) # NaN to zero after normalisation

            proj_img = model.fit_predict(raster.reshape(-1, raster.shape[-1]))

            proj_img = proj_img.reshape((raster.shape[0], raster.shape[1],-1))
            height, width, channels = proj_img.shape

            feedback.pushInfo(f'Export to geotif\n')
            with rasterio.open(self.dst_path, 'w', driver='GTiff',
                               height=height, 
                               width=width, 
                               count=channels, 
                               dtype=self.out_dtype,
                               crs=crs, 
                               transform=transform) as dst_ds:
                dst_ds.write(np.transpose(proj_img, (2, 0, 1)))
            feedback.pushInfo(f'Export to geotif done\n')

        return model


    def infer_model(self, model, feedback, scaler=None):

        with rasterio.open(self.rlayer_path) as ds:

            transform = ds.transform
            crs = ds.crs
            win = windows.from_bounds(
                    self.extent.xMinimum(), 
                    self.extent.yMinimum(), 
                    self.extent.xMaximum(), 
                    self.extent.yMaximum(), 
                    transform=transform
                    )
            raster = ds.read(window=win)
            transform = ds.window_transform(win)
            raster = np.transpose(raster, (1,2,0))
            raster = raster[:,:,self.input_bands]


            inf_raster = raster.reshape(-1, raster.shape[-1])
            if scaler:
                inf_raster = scaler.transform(inf_raster)
            np.nan_to_num(inf_raster) # NaN to zero after normalisation

            if self.TYPE == 'cluster':
                proj_img = model.predict(inf_raster)

            else:
                proj_img = model.transform(inf_raster)

            proj_img = proj_img.reshape((raster.shape[0], raster.shape[1],-1))
            height, width, channels = proj_img.shape

            feedback.pushInfo(f'Export to geotif\n')
            with rasterio.open(self.dst_path, 'w', driver='GTiff',
                               height=height, 
                               width=width, 
                               count=channels, 
                               dtype=self.out_dtype,
                               crs=crs, 
                               transform=transform) as dst_ds:
                dst_ds.write(np.transpose(proj_img, (2, 0, 1)))
            feedback.pushInfo(f'Export to geotif done\n')

    def update_kwargs(self, kwargs_dict):

        if 'n_clusters' in kwargs_dict.keys():
            kwargs_dict['n_clusters'] = self.main_param
        if 'n_components' in kwargs_dict.keys():
            kwargs_dict['n_components'] = self.main_param

        if 'random_state' in kwargs_dict.keys():
            kwargs_dict['random_state'] = self.seed

        for key, value in self.passed_kwargs.items():
            if key in kwargs_dict.keys():
                kwargs_dict[key] = value

        return kwargs_dict

    def get_help_sk_methods(self):
        """
        Generate help string with default arguments of supported sklearn algorithms.
        """
            
        proj_methods = ['fit', 'transform']
        clust_methods = ['fit', 'fit_predict']
        help_str = '\n\n Here are the default arguments of the supported algorithms:\n\n'

        if self.TYPE == 'proj':
            algos = get_sklearn_algorithms_with_methods(decomposition, proj_methods)
            for algo in algos :
                args = get_arguments(decomposition,algo)
                help_str += f'- {algo}:\n'
                help_str += f'{args}\n'
            algos = get_sklearn_algorithms_with_methods(cluster, proj_methods)
            for algo in algos :
                args = get_arguments(cluster,algo)
                help_str += f'- {algo}:\n'
                help_str += f'{args}\n'

        if self.TYPE == 'cluster':
            algos = get_sklearn_algorithms_with_methods(cluster, clust_methods)
            for algo in algos :
                args = get_arguments(cluster,algo)
                help_str += f'- {algo}:\n'
                help_str += f'{args}\n'

        return help_str

    def print_transform_metrics(self, model, feedback):
        """
        Log common metrics after a PCA.
        """
        
        if hasattr(model, 'explained_variance_ratio_'):
            # Explained variance ratio
            explained_variance_ratio = model.explained_variance_ratio_

            # Cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance_ratio)

            # Loadings (Principal axes)
            loadings = model.components_.T * np.sqrt(model.explained_variance_)

            feedback.pushInfo(f'Explained Variance Ratio : \n{explained_variance_ratio}')
            feedback.pushInfo(f'Cumulative Explained Variance : \n{cumulative_variance}')
            feedback.pushInfo(f'Loadings (Principal axes) : \n{loadings}')

    def print_cluster_metrics(self, model, fit_raster ,feedback):
        """
        Log common metrics after a Kmeans.
        """
        
        if hasattr(model, 'inertia_'):

            feedback.pushInfo(f'Inertia : \n{model.inertia_}')
            feedback.pushInfo(f'Cluster sizes : \n{Counter(model.labels_)}')
            ## silouhette score seem to heavy for now
            # feedback.pushInfo(f'Silhouette Score : \n{silhouette_score(fit_raster, model.labels_)}')
            # feedback.pushInfo(f'Silouhette Values : \n{silhouette_values(fit_raster, model.labels_)}')

    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        return {}



class SHPAlgorithm(IAMAPAlgorithm):
    """
    Common class for algorithms relying on shapefile data.
    """

    TEMPLATE = 'TEMPLATE'
    RANDOM_SAMPLES = 'RANDOM_SAMPLES'
    TMP_DIR = 'iamap_sim'
    DEFAULT_TEMPLATE = 'template.shp'
    TYPE = 'similarity'
    

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.init_input_output_raster()
        self.init_seed()
        self.init_input_shp()


    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        self.process_geo_parameters(parameters, context, feedback)
        self.process_common_shp(parameters, context, feedback)

        fit_raster = self.get_fit_raster()

        self.inf_raster(fit_raster)

        return {'OUTPUT_RASTER':self.dst_path, 'OUTPUT_LAYER_NAME':self.layer_name, 'USED_SHP':self.used_shp_path}

    def init_input_shp(self):
        samples_param = QgsProcessingParameterNumber(
            name=self.RANDOM_SAMPLES,
            description=self.tr(
                'Random samples taken if input is not in point geometry'),
            type=QgsProcessingParameterNumber.Integer,
            optional=True,
            minValue=0,
            defaultValue=500,
            maxValue=100_000
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                name=self.TEMPLATE,
                description=self.tr(
                    'Input shapefile path'),
            # defaultValue=os.path.join(self.cwd,'assets',self.DEFAULT_TEMPLATE),
            ),
        )

        samples_param.setFlags(
            samples_param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(samples_param)


    def get_fit_raster(self):

        with rasterio.open(self.rlayer_path) as ds:
            gdf = self.gdf.to_crs(ds.crs)
            pixel_values = []

            transform = ds.transform
            win = windows.from_bounds(
                    self.extent.xMinimum(), 
                    self.extent.yMinimum(), 
                    self.extent.xMaximum(), 
                    self.extent.yMaximum(), 
                    transform=transform
                    )
            raster = ds.read(window=win)
            transform = ds.window_transform(win)
            raster = raster[self.input_bands,:,:]

            for index, data in gdf.iterrows():
                # Get the coordinates of the point in the raster's pixel space
                x, y = data.geometry.x, data.geometry.y

                # Convert point coordinates to pixel coordinates within the window
                col, row = ~transform * (x, y)  # Convert from map coordinates to pixel coordinates
                col, row = int(col), int(row)
                pixel_values.append(list(raster[:,row, col]))

            return np.asarray(pixel_values)


    def inf_raster(self, fit_raster):

        with rasterio.open(self.rlayer_path) as ds:

            transform = ds.transform
            crs = ds.crs
            win = windows.from_bounds(
                    self.extent.xMinimum(), 
                    self.extent.yMinimum(), 
                    self.extent.xMaximum(), 
                    self.extent.yMaximum(), 
                    transform=transform
                    )
            raster = ds.read(window=win)
            transform = ds.window_transform(win)
            raster = raster[self.input_bands,:,:]

            raster = np.transpose(raster, (1,2,0))

            template = torch.from_numpy(fit_raster).to(torch.float32)
            template = torch.mean(template, dim=0)
        
            feat_img = torch.from_numpy(raster)
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
            sim = cos(feat_img,template)
            sim = sim.unsqueeze(-1)
            sim = sim.numpy()
            height, width, channels = sim.shape
                        
            with rasterio.open(self.dst_path, 'w', driver='GTiff',
                               height=height, width=width, count=channels, dtype=self.out_dtype,
                               crs=crs, transform=transform) as dst_ds:
                dst_ds.write(np.transpose(sim, (2, 0, 1)))



    def process_common_shp(self, parameters, context, feedback):

        output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = self.parameterAsInt(
            parameters, self.RANDOM_SEED, context)

        self.input_bands = [i_band -1 for i_band in self.selected_bands]

        template = self.parameterAsVectorLayer(
            parameters, self.TEMPLATE, context)
        self.template = template.dataProvider().dataSourceUri()
        random_samples = self.parameterAsInt(
            parameters, self.RANDOM_SAMPLES, context)

        gdf = gpd.read_file(self.template)
        gdf = gdf.to_crs(self.crs.toWkt())

        feedback.pushInfo(f'before sampling: {len(gdf)}')
        ## If gdf is not point geometry, we take random samples in it
        gdf = get_random_samples_in_gdf(gdf, random_samples, seed=self.seed)
        feedback.pushInfo(f'after samples:\n {len(gdf)}')

        self.used_shp_path = os.path.join(self.output_dir, 'used.shp')
        feedback.pushInfo(f'saving used dataframe to: {self.used_shp_path}')
        gdf.to_file(self.used_shp_path)


        feedback.pushInfo(f'before extent: {len(gdf)}')
        bounds = box(
                self.extent.xMinimum(), 
                self.extent.yMinimum(), 
                self.extent.xMaximum(), 
                self.extent.yMaximum(), 
                )
        self.gdf = gdf[gdf.within(bounds)]
        feedback.pushInfo(f'after extent: {len(self.gdf)}')

        if len(self.gdf) == 0:
            feedback.pushWarning("No template points within extent !")
            return False

        rlayer_basename = os.path.basename(self.rlayer_path)
        rlayer_name, ext = os.path.splitext(rlayer_basename)

        ### Handle args that differ between clustering and projection methods
        if self.TYPE == 'similarity':
            self.dst_path, self.layer_name = get_unique_filename(self.output_dir, f'{self.TYPE}.tif', f'{rlayer_name} similarity')
        else:
            self.dst_path, self.layer_name = get_unique_filename(self.output_dir, f'{self.TYPE}.tif', f'{rlayer_name} ml')

        ## default to float32 until overriden if ML algo
        self.out_dtype = 'float32'


    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        return {}

if __name__ == "__main__":


    methods = ['fit_predict', 'partial_fit'] 
    algos = get_sklearn_algorithms_with_methods(cluster, methods)
    print(algos)
    methods = ['predict', 'fit'] 
    algos = get_sklearn_algorithms_with_methods(cluster, methods)
    print(algos)

    for algo in algos :
        args = get_arguments(cluster,algo)
        print(algo, args)


    methods = ['transform', 'fit'] 
    algos = get_sklearn_algorithms_with_methods(decomposition, methods)
    for algo in algos :
        args = get_arguments(decomposition,algo)
        print(algo, args)
