import os
import tempfile
import numpy as np
import inspect
import joblib
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
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterCrs,
                       QgsProcessingParameterDefinition,
                       )
import rasterio
from rasterio import windows
from rasterio.enums import Resampling

import sklearn.decomposition as decomposition
import sklearn.cluster as cluster
from sklearn.base import BaseEstimator

from .misc import get_unique_filename, calculate_chunk_size


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
    
    if parameter_name in parameters:
        param = parameters[parameter_name]
        
        # Check if the parameter has a default value
        if param.default == inspect.Parameter.empty:
            return f"'{parameter_name}' is required for {cls.__name__}."
        else:
            return f"'{parameter_name}' is optional for {cls.__name__}, default: {param.default}."
    else:
        return f"'{parameter_name}' is not a parameter for {cls.__name__}."

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
    CRS = 'CRS'
    COMPRESS = 'COMPRESS'
    TMP_DIR = 'iamap_tmp'
    

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        cwd = Path(__file__).parent.absolute()
        tmp_wd = os.path.join(tempfile.gettempdir(), self.TMP_DIR)

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

        compress_param = QgsProcessingParameterBoolean(
            name=self.COMPRESS,
            description=self.tr(
                'Compress final result to JP2'),
            defaultValue=True,
            optional=True,
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

        self.out_dtype_opt = ['float32', 'int8']
        dtype_param = QgsProcessingParameterEnum(
                name=self.OUT_DTYPE,
                description=self.tr(
                    'Data type of exported features (int8 saves space)'),
                options=self.out_dtype_opt,
                defaultValue=0,
                )


        for param in (
                crs_param, 
                res_param, 
                dtype_param,
                compress_param,
                ):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)


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
    """

    LOAD = 'LOAD'
    OUTPUT = 'OUTPUT'
    MAIN_PARAM = 'MAIN_PARAM'
    SUBSET = 'SUBSET'
    METHOD = 'METHOD'
    SAVE_MODEL = 'SAVE_MODEL'
    COMPRESS = 'COMPRESS'
    RANDOM_SEED = 'RANDOM_SEED'
    TMP_DIR = 'iamap_sk'
    TYPE = 'proj'
    

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        cwd = Path(__file__).parent.parent.absolute()
        tmp_wd = os.path.join(tempfile.gettempdir(),self.TMP_DIR)
        proj_methods = ['fit', 'transform']
        clust_methods = ['fit', 'predict']

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
                description=self.tr(
                    'Selected Bands (defaults to all bands selected)'),
                defaultValue=None,
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

        seed_param = QgsProcessingParameterNumber(
            name=self.RANDOM_SEED,
            description=self.tr(
                'Random seed'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=42,
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

        if self.TYPE == 'proj':
            method_opt = get_sklearn_algorithms_with_methods(decomposition, proj_methods)
            print(method_opt)

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
        else :
            method_opt = get_sklearn_algorithms_with_methods(cluster, proj_methods)
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


        self.method_opt = method_opt +['--Empty--']
        print(self.method_opt)
        self.addParameter (
            QgsProcessingParameterEnum(
                name = self.METHOD,
                description = self.tr(
                    'Sklearn algorithm used'),
                defaultValue = 0,
                options = self.method_opt,
            )
        )

        # self.addParameter(
        #     QgsProcessingParameterNumber(
        #         name = self.THRESOLD_PCA,
        #         description = self.tr (
        #             'Thresold for displaying contribution of each variables when using PCA (between 0 and 1)'
        #         ),
        #         type = QgsProcessingParameterNumber.Double,
        #         minValue = 0,
        #         defaultValue = 0.5,
        #         maxValue = 1
        #     )
        # )

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


        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr(
                    "Output directory (choose the location that the image features will be saved)"),
            # defaultValue=os.path.join(cwd,'models'),
            defaultValue=tmp_wd,
            )
        )


        for param in (
                crs_param, 
                res_param, 
                subset_param, 
                seed_param,
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
        self.process_common_sklearn(parameters, context, feedback)

        fit_raster = self.get_fit_raster(feedback)
        kwargs = {
                'n_components': self.main_param,
                'random_state': self.seed,
                  }

        if self.TYPE == 'proj':
            model = instantiate_sklearn_algorithm(decomposition, self.method_name, **kwargs)
        else:
            model = instantiate_sklearn_algorithm(cluster, self.method_name, self.main_param)

        iter = get_iter(model, fit_raster)

        model = self.fit_model(model, fit_raster, iter, feedback)

        feedback.pushInfo(f'Fitting done, saving model\n')
        save_file = f'{self.method_name}.pkl'.lower()
        if self.save_model:
            out_path = os.path.join(self.output_dir, save_file)
            joblib.dump(model, out_path)

        feedback.pushInfo(f'Inference over raster\n')
        dst_path, layer_name = self.infer_model(model, feedback)

        parameters['OUTPUT_RASTER']=dst_path

        return {'OUTPUT_RASTER':dst_path, 'OUTPUT_LAYER_NAME':layer_name}


    def process_common_sklearn(self,parameters, context, feedback):

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
            fit_raster = raster.reshape(-1, raster.shape[-1])

            if self.subset:

                feedback.pushInfo(f'Using a random subset of {self.subset} pixels, random seed is {RANDOM_SEED}')

                fit_raster = raster.reshape(-1, raster.shape[-1])
                nsamples = fit_raster.shape[0]
    
                # Generate random indices to select subset_size number of samples
                np.random.seed(self.seed)
                random_indices = np.random.choice(nsamples, size=self.subset, replace=False)
                fit_raster = fit_raster[random_indices,:]

                # remove nans
                fit_raster = fit_raster[~np.isnan(fit_raster).any(axis=1)]

                feedback.pushInfo(f"Mean raster : {np.mean(raster)}")
                feedback.pushInfo(f"Standart dev : {np.std(raster)}")
                fit_raster = (fit_raster-np.mean(raster))/np.std(raster)
                feedback.pushInfo(f"Mean raster normalized : {np.mean(fit_raster)}")
                feedback.pushInfo(f"Standart dev normalized : {np.std(fit_raster)}")

        return fit_raster

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

    def infer_model(self, model, feedback):

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


            raster = (raster-np.mean(raster))/np.std(raster)
            np.nan_to_num(raster) # NaN to zero after normalisation

            if self.TYPE == 'proj':
                proj_img = model.transform(raster.reshape(-1, raster.shape[-1]))
            else:
                proj_img = model.predict(raster.reshape(-1, raster.shape[-1]))

            proj_img = proj_img.reshape((raster.shape[0], raster.shape[1],-1))
            height, width, channels = proj_img.shape

            if self.TYPE == 'proj':
                dtype = 'float32'
            else:
                dtype = 'uint8'

            feedback.pushInfo(f'Export to geotif\n')
            rlayer_basename = os.path.basename(self.rlayer_path)
            rlayer_name, ext = os.path.splitext(rlayer_basename)

            if self.TYPE == 'proj':
                dst_path, layer_name = get_unique_filename(self.output_dir, f'{self.TYPE}.tif', f'{rlayer_name} reduction')
            else:
                dst_path, layer_name = get_unique_filename(self.output_dir, f'{self.TYPE}.tif', f'{rlayer_name} cluster')

            with rasterio.open(dst_path, 'w', driver='GTiff',
                               height=height, 
                               width=width, 
                               count=channels, 
                               dtype=dtype,
                               crs=crs, 
                               transform=transform) as dst_ds:
                dst_ds.write(np.transpose(proj_img, (2, 0, 1)))

            return dst_path, layer_name


    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        return {}
