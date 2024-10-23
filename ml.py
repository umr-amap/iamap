import os
import ast
import numpy as np
from pathlib import Path
from typing import Dict, Any
import joblib
import json
import tempfile

import rasterio
from rasterio import windows
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (Qgis,
                       QgsGeometry,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterEnum,
                       QgsCoordinateTransform,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterString,
                       QgsProcessingParameterBand,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterCrs,
                       QgsProcessingParameterDefinition,
                       )
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .utils.misc import get_unique_filename
from .utils.geo import get_random_samples_in_gdf, get_unique_col_name
from .utils.algo import (
                        SHPAlgorithm,
                        get_sklearn_algorithms_with_methods,
                        instantiate_sklearn_algorithm,
                        get_arguments,
                        )

import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
from sklearn.base import ClassifierMixin, RegressorMixin
def check_model_type(model):
    if isinstance(model, ClassifierMixin):
        return "classification"
    elif isinstance(model, RegressorMixin):
        return "regression"
    else:
        return "unknown"


class MLAlgorithm(SHPAlgorithm):
    """
    """

    GT_COL = 'GT_COL'
    DO_KFOLDS = 'DO_KFOLDS'
    FOLD_COL = 'FOLD_COL'
    NFOLDS = 'NFOLDS'
    SK_PARAM = 'SK_PARAM'
    TEMPLATE_TEST = 'TEMPLATE_TEST'
    METHOD = 'METHOD'
    TMP_DIR = 'iamap_ml'
    DEFAULT_TEMPLATE = 'ml_poly.shp'
    TYPE = 'ml'

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.init_input_output_raster()
        self.init_seed()
        self.init_input_shp()

        self.method_opt = self.get_algorithms()
        default_index = self.method_opt.index('RandomForestClassifier')
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


        self.addParameter(
            QgsProcessingParameterFile(
                name=self.TEMPLATE,
                description=self.tr(
                    'Input shapefile path for training data set for random forest (if no test data_set, will be devised in train and test)'),
            # defaultValue=os.path.join(self.cwd,'assets',self.DEFAULT_TEMPLATE),
            ),
        )
        
        self.addParameter(
            QgsProcessingParameterFile(
                name=self.TEMPLATE_TEST,
                description=self.tr(
                    'Input shapefile path for test dataset.'),
                optional = True
            ),
        )


        self.addParameter (
            QgsProcessingParameterString(
                name = self.GT_COL,
                description = self.tr(
                    'Name of the column containing ground truth values.'),
                defaultValue = 'Type',
            )
        )

        self.addParameter (
            QgsProcessingParameterBoolean(
                name = self.DO_KFOLDS,
                description = self.tr(
                    'Perform cross-validation'),
                defaultValue = True,
            )
        )
        self.addParameter (
            QgsProcessingParameterString(
                name = self.FOLD_COL,
                description = self.tr(
                    'Name of the column defining folds in case of cross-validation. If none is selected, random sampling is used.'),
                defaultValue = '',
                optional=True,
            )
        )

        nfold_param = QgsProcessingParameterNumber(
            name=self.NFOLDS,
            description=self.tr(
                'Number of folds performed'),
            type=QgsProcessingParameterNumber.Integer,
            optional=True,
            minValue=2,
            defaultValue=5,
            maxValue=10
        )

        for param in (
                nfold_param,
                ):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)


    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        self.process_geo_parameters(parameters, context, feedback)
        self.process_common_shp(parameters, context, feedback)
        self.process_ml_shp(parameters, context, feedback)
        self.process_ml_options(parameters, context, feedback)

        if self.test_gdf:
            self.get_fit_raster()

        if self.do_kfold:
            pass

        return {'OUTPUT_RASTER':self.dst_path, 'OUTPUT_LAYER_NAME':self.layer_name}

        # fit_raster = self.get_fit_raster()

        # # self.inf_raster(fit_raster)

        # return {'OUTPUT_RASTER':self.dst_path, 'OUTPUT_LAYER_NAME':self.layer_name}


    def process_ml_shp(self, parameters, context, feedback):

        template_test = self.parameterAsFile(
            parameters, self.TEMPLATE_TEST, context)

        self.test_gdf=None

        if template_test != '' :
            random_samples = self.parameterAsInt(
                parameters, self.RANDOM_SAMPLES, context)

            gdf = gpd.read_file(template_test)
            gdf = gdf.to_crs(self.crs.toWkt())

            feedback.pushInfo(f'before samples: {len(gdf)}')
            ## get random samples if geometry is not point based
            gdf = get_random_samples_in_gdf(gdf, random_samples)

            feedback.pushInfo(f'before extent: {len(gdf)}')
            bounds = box(
                    self.extent.xMinimum(), 
                    self.extent.yMinimum(), 
                    self.extent.xMaximum(), 
                    self.extent.yMaximum(), 
                    )
            self.test_gdf = gdf[gdf.within(bounds)]
            feedback.pushInfo(f'after extent: {len(self.test_gdf)}')

            if len(self.test_gdf) == 0:
                feedback.pushWarning("No template points within extent !")
                return False

    def process_ml_options(self, parameters, context, feedback):

        self.do_kfold = self.parameterAsBoolean(
            parameters, self.DO_KFOLDS, context)
        self.gt_col = self.parameterAsString(
            parameters, self.GT_COL, context)
        fold_col = self.parameterAsString(
            parameters, self.FOLD_COL, context)
        nfolds = self.parameterAsInt(
            parameters, self.NFOLDS, context)

        ## If no test set is provided and the option to perform kfolds is true,
        ## we perform kfolds
        str_kwargs = self.parameterAsString(
                parameters, self.SK_PARAM, context)

        if str_kwargs != '':
            self.passed_kwargs = ast.literal_eval(str_kwargs)
        else:
            self.passed_kwargs = {}
        ## If a fold column is provided, this defines the folds. Otherwise, random split
        if self.test_gdf == None and self.do_kfold:
            if fold_col != '':
                self.gdf['fold'] = self.gdf[fold_col]
            else:
                self.gdf['fold'] = np.random.randint(1, nfolds + 1, size=len(self.gdf))
                print(self.gdf)
        method_idx = self.parameterAsEnum(
            parameters, self.METHOD, context)
        self.method_name = self.method_opt[method_idx]

        try:
            default_args = get_arguments(ensemble, self.method_name)
        except AttributeError:
            default_args = get_arguments(neighbors, self.method_name)

        kwargs = self.update_kwargs(default_args)

        try:
            self.model = instantiate_sklearn_algorithm(ensemble, self.method_name, **kwargs)
        except AttributeError:
            self.model = instantiate_sklearn_algorithm(neighbors, self.method_name, **kwargs)

    def get_algorithms(self):
        required_methods = ['fit', 'predict']
        ensemble_algos = get_sklearn_algorithms_with_methods(ensemble, required_methods)
        neighbors_algos = get_sklearn_algorithms_with_methods(neighbors, required_methods)
        return sorted(ensemble_algos+neighbors_algos)

    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        return {}


    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return MLAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'ml'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Machine Learning')

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
        return self.tr("Fit a Machine Learning model using input template")

    def icon(self):
        return 'E'








#class RFAlgorithm(QgsProcessingAlgorithm):
#    """
#    """

#    INPUT = 'INPUT'
#    BANDS = 'BANDS'
#    EXTENT = 'EXTENT'
#    LOAD = 'LOAD'
#    OUTPUT = 'OUTPUT'
#    RESOLUTION = 'RESOLUTION'
#    CRS = 'CRS'
#    TEMPLATE = 'TEMPLATE'
#    COLONNE_RF = 'COLONNE_RF'
#    TEMPLATE_TEST = 'TEMPLATE_TEST'

#    def initAlgorithm(self, config=None):
#        """
#        Here we define the inputs and output of the algorithm, along
#        with some other properties.
#        """
#        cwd = Path(__file__).parent.absolute()
#        tmp_wd = os.path.join(tempfile.gettempdir(), "iamap_rf")

#        self.addParameter(
#            QgsProcessingParameterRasterLayer(
#                name=self.INPUT,
#                description=self.tr(
#                    'Input raster layer or image file path'),
#            defaultValue=os.path.join(cwd,'assets','test.tif'),
#            ),
#        )

#        self.addParameter(
#            QgsProcessingParameterBand(
#                name=self.BANDS,
#                description=self.tr(
#                    'Selected Bands (defaults to all bands selected)'),
#                defaultValue=None,
#                parentLayerParameterName=self.INPUT,
#                optional=True,
#                allowMultiple=True,
#            )
#        )

#        crs_param = QgsProcessingParameterCrs(
#            name=self.CRS,
#            description=self.tr('Target CRS (default to original CRS)'),
#            optional=True,
#        )

#        res_param = QgsProcessingParameterNumber(
#            name=self.RESOLUTION,
#            description=self.tr(
#                'Target resolution in meters (default to native resolution)'),
#            type=QgsProcessingParameterNumber.Double,
#            optional=True,
#            minValue=0,
#            maxValue=100000
#        )

#        self.addParameter(
#            QgsProcessingParameterExtent(
#                name=self.EXTENT,
#                description=self.tr(
#                    'Processing extent (default to the entire image)'),
#                optional=True
#            )
#        )

#        self.addParameter(
#            QgsProcessingParameterFile(
#                name=self.TEMPLATE,
#                description=self.tr(
#                    'Input shapefile path for training data set for random forest (if no test data_set, will be devised in train and test)'),
#            # defaultValue=os.path.join(cwd,'assets','rf.gpkg'),
#            defaultValue=os.path.join(cwd,'assets','rf.shp'),
#            ),
#        )
        
#        self.addParameter(
#            QgsProcessingParameterFile(
#                name=self.TEMPLATE_TEST,
#                description=self.tr(
#                    'Input shapefile path for test data set for random forest (optional)'),
#                optional = True
#            ),
#        )


#        self.addParameter(
#            QgsProcessingParameterFolderDestination(
#                self.OUTPUT,
#                self.tr(
#                    "Output directory (choose the location that the image features will be saved)"),
#            defaultValue=tmp_wd,
#            )
#        )
        
        
#        self.addParameter (
#            QgsProcessingParameterString(
#                name = self.COLONNE_RF,
#                description = self.tr(
#                    'Name of the column you want random forest to work on'),
#                defaultValue = 'Type',
#            )
#        )


#        for param in (crs_param, res_param):
#            param.setFlags(
#                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
#            self.addParameter(param)


#    def processAlgorithm(self, parameters, context, feedback):
#        """
#        Here is where the processing itself takes place.
#        """
#        self.process_options(parameters, context, feedback)

#        gdf = gpd.read_file(self.template)
#        feedback.pushInfo(f"self_template_test : {self.template_test}")
#        DATA_SET_TEST=False
#        if(self.template_test != ''):
#            gdf_test = gpd.read_file(self.template_test)
#            gdf_test = gdf_test.to_crs(self.crs.toWkt())
#            DATA_SET_TEST = True
#            feedback.pushInfo(f"In good loop !")

#        gdf = gdf.to_crs(self.crs.toWkt())
        
#        feedback.pushInfo(f"DATA_SET_TEST : {DATA_SET_TEST}")
        

#        feedback.pushInfo(f'before extent: {len(gdf)}')
#        bounds = box(
#                self.extent.xMinimum(), 
#                self.extent.yMinimum(), 
#                self.extent.xMaximum(), 
#                self.extent.yMaximum(), 
#                )
#        feedback.pushInfo(f'xmin: {self.extent.xMinimum()},ymin: {self.extent.yMinimum()}, xmax: {self.extent.xMaximum()}, ymax: {self.extent.yMaximum()} ')
#        gdf = gdf[gdf.within(bounds)]
#        feedback.pushInfo(f'after extent: {len(gdf)}')

#        if len(gdf) == 0:
#            feedback.pushWarning("No template points within extent !")
#            return False

        


#        input_bands = [i_band -1 for i_band in self.selected_bands]


#        with rasterio.open(self.rlayer_path) as ds:
            
        
            
            
#            gdf = gdf.to_crs(ds.crs)
            
#            pixel_values_test = []
            
#            pixel_values = []

#            transform = ds.transform
#            crs = ds.crs
#            win = windows.from_bounds(
#                    self.extent.xMinimum(), 
#                    self.extent.yMinimum(), 
#                    self.extent.xMaximum(), 
#                    self.extent.yMaximum(), 
#                    transform=transform
#                    )
#            raster = ds.read(window=win)
#            transform = ds.window_transform(win)
#            raster = raster[input_bands,:,:]

#            if (DATA_SET_TEST == True):
#                gdf_test = gdf_test.to_crs(ds.crs)
#                for index, data in gdf_test.iterrows():
#                    # Get the coordinates of the point in the raster's pixel space
#                    x, y = data.geometry.x, data.geometry.y
#                    feedback.pushInfo (f"x : {x}, y : {y}")
#                    feedback.pushInfo (f"gdf geometry : {gdf['geometry']}")

#                    # Convert point coordinates to pixel coordinates within the window
#                    col, row = ~transform * (x, y)  # Convert from map coordinates to pixel coordinates
#                    col, row = int(col), int(row)
#                    feedback.pushInfo(f'after extent: {row, col}')
#                    pixel_values_test.append(list(raster[:,row, col]))
#                    template_npy_test = np.asarray(pixel_values_test)
#                    template_test = torch.from_numpy(template_npy_test).to(torch.float32)

#            for index, data in gdf.iterrows():
#                # Get the coordinates of the point in the raster's pixel space
#                x, y = data.geometry.x, data.geometry.y
#                feedback.pushInfo (f"x : {x}, y : {y}")
#                feedback.pushInfo (f"gdf geometry : {gdf['geometry']}")

#                # Convert point coordinates to pixel coordinates within the window
#                col, row = ~transform * (x, y)  # Convert from map coordinates to pixel coordinates
#                col, row = int(col), int(row)
#                feedback.pushInfo(f'after extent: {row, col}')
#                pixel_values.append(list(raster[:,row, col]))

#            raster = np.transpose(raster, (1,2,0))

#            feedback.pushInfo(f'{raster.shape}')

#            template_npy = np.asarray(pixel_values)
            
#            feedback.pushInfo(f'points : {template_npy}')
#            feedback.pushInfo(f'dim points : {template_npy.shape}')
#            template = torch.from_numpy(template_npy).to(torch.float32)
            

            
        
#            feat_img = torch.from_numpy(raster)
            
#            #template contient les valeurs
#            #y = gdf['Type']
#            #y=gdf ['Desc_']

#            if (DATA_SET_TEST == False):
                
#                if self.colonne_rf in gdf.columns :
                    
#                    y = gdf[self.colonne_rf]
#                else :
#                    feedback.pushWarning (f'{self.colonne_rf} is not a valid column name of the dataset !!')
                
                
                
#                X_train, X_test, y_train, y_test = train_test_split(template, y, test_size=0.4, random_state=55)
                
#                rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42)
#                rf_classifier.fit(X_train, y_train)
                
#            if (DATA_SET_TEST == True):
                
#                y_train=gdf[self.colonne_rf]
#                y_test=gdf_test[self.colonne_rf]
            
#                X_train = template
                
                
#                X_test = template_test
                
                
#                rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42)
                
#                rf_classifier.fit(X_train, y_train)

#            params = rf_classifier.get_params()
            
#            #joblib.dump(rf_classifier, model_file)
                
#            y_pred = rf_classifier.predict(X_test)
#            accuracy = accuracy_score(y_test, y_pred)
#            feedback.pushInfo(f"Accuracy ; {accuracy}")
            
#            predicted_types = rf_classifier.predict(raster.reshape(-1, raster.shape[-1]))
#            feedback.pushInfo(f"predicted types ; {predicted_types.shape}")
#            predicted_types_image = predicted_types.reshape(raster.shape[:-1])
#            feedback.pushInfo(f"predicted_types_image ; {predicted_types_image.shape}")
            
#            label_encoder = LabelEncoder()
#            predicted_types_numeric = label_encoder.fit_transform(predicted_types_image.flatten())
#            feedback.pushInfo(f'carte avant transfo : {predicted_types_numeric.shape}')
#            predicted_types_numeric = predicted_types_numeric.reshape(predicted_types_image.shape)
#            feedback.pushInfo(f'carte après transfo : {predicted_types_numeric.shape}')
        

#            height, width = predicted_types_numeric.shape
#            channels = 1
            
            
            
            

#            dst_path = os.path.join(self.output_dir,'random_forest.tif')
#            params_file = os.path.join(self.output_dir, 'random_forest_parameters.json')
            

            
#            rlayer_basename = os.path.basename(self.rlayer_path)
#            rlayer_name, ext = os.path.splitext(rlayer_basename)
#            dst_path, layer_name = get_unique_filename(self.output_dir, 'random_forest.tif', f'{rlayer_name} random forest')
#            # if os.path.exists(dst_path):
#            #         i = 1
#            #         while True:
#            #             modified_output_file = os.path.join(self.output_dir, f"random_forest_{i}.tif")
#            #             if not os.path.exists(modified_output_file):
#            #                 dst_path = modified_output_file
#            #                 break
#            #             i += 1
                        
#            if os.path.exists(params_file):
#                    i = 1
#                    while True:
#                        modified_output_file_params = os.path.join(self.output_dir, f"random_forest_parameters_{i}.json")
#                        if not os.path.exists(modified_output_file_params):
#                            params_file = modified_output_file_params
#                            break
#                        i += 1

#            with rasterio.open(dst_path, 'w', driver='GTiff',
#                               height=height, width=width, count=channels, dtype='float32',
#                               crs=crs, transform=transform) as dst_ds:
#                dst_ds.write(predicted_types_numeric, 1)
                
#            with open(params_file, 'w') as f:
#                json.dump(params, f, indent=4)
#            feedback.pushInfo(f"Parameters saved to {params_file}")

#            parameters['OUTPUT_RASTER']=dst_path

#        return {'OUTPUT_RASTER':dst_path, 'OUTPUT_LAYER_NAME':layer_name}

#    def process_options(self,parameters, context, feedback):
#        self.iPatch = 0
        
#        self.feature_dir = ""
        
#        feedback.pushInfo(
#                f'PARAMETERS :\n{parameters}')
        
#        feedback.pushInfo(
#                f'CONTEXT :\n{context}')
        
#        feedback.pushInfo(
#                f'FEEDBACK :\n{feedback}')

#        rlayer = self.parameterAsRasterLayer(
#            parameters, self.INPUT, context)
        
#        if rlayer is None:
#            raise QgsProcessingException(
#                self.invalidRasterError(parameters, self.INPUT))

#        self.selected_bands = self.parameterAsInts(
#            parameters, self.BANDS, context)

#        if len(self.selected_bands) == 0:
#            self.selected_bands = list(range(1, rlayer.bandCount()+1))

#        if max(self.selected_bands) > rlayer.bandCount():
#            raise QgsProcessingException(
#                self.tr("The chosen bands exceed the largest band number!")
#            )

#        self.template = self.parameterAsFile(
#            parameters, self.TEMPLATE, context)
        
#        self.template_test = self.parameterAsFile(
#            parameters, self.TEMPLATE_TEST, context)
        
#        self.colonne_rf = self.parameterAsString(
#            parameters, self.COLONNE_RF, context)

#        res = self.parameterAsDouble(
#            parameters, self.RESOLUTION, context)
#        crs = self.parameterAsCrs(
#            parameters, self.CRS, context)
#        extent = self.parameterAsExtent(
#            parameters, self.EXTENT, context)
#        output_dir = self.parameterAsString(
#            parameters, self.OUTPUT, context)
#        self.output_dir = Path(output_dir)
#        self.output_dir.mkdir(parents=True, exist_ok=True)


#        rlayer_data_provider = rlayer.dataProvider()

#        # handle crs
#        if crs is None or not crs.isValid():
#            crs = rlayer.crs()
#            feedback.pushInfo(f'crs : {crs}')
            
#            """
#            feedback.pushInfo(
#                f'Layer CRS unit is {crs.mapUnits()}')  # 0 for meters, 6 for degrees, 9 for unknown
#            feedback.pushInfo(
#                f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
#            if crs.mapUnits() == Qgis.DistanceUnit.Degrees:
#                crs = self.estimate_utm_crs(rlayer.extent())

#        # target crs should use meters as units
#        if crs.mapUnits() != Qgis.DistanceUnit.Meters:
#            feedback.pushInfo(
#                f'Layer CRS unit is {crs.mapUnits()}')
#            feedback.pushInfo(
#                f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
#            raise QgsProcessingException(
#                self.tr("Only support CRS with the units as meters")
#            )
#        """

#        # 0 for meters, 6 for degrees, 9 for unknown
#        UNIT_METERS = 0
#        UNIT_DEGREES = 6
#        if rlayer.crs().mapUnits() == UNIT_DEGREES: # Qgis.DistanceUnit.Degrees:
#            layer_units = 'degrees'
#        else:
#            layer_units = 'meters'
#        # if res is not provided, get res info from rlayer
#        if np.isnan(res) or res == 0:
#            res = rlayer.rasterUnitsPerPixelX()  # rasterUnitsPerPixelY() is negative
#            target_units = layer_units
#        else:
#            # when given res in meters by users, convert crs to utm if the original crs unit is degree
#            if crs.mapUnits() != UNIT_METERS: # Qgis.DistanceUnit.Meters:
#                if rlayer.crs().mapUnits() == UNIT_DEGREES: # Qgis.DistanceUnit.Degrees:
#                    # estimate utm crs based on layer extent
#                    crs = self.estimate_utm_crs(rlayer.extent())
#                else:
#                    raise QgsProcessingException(
#                        f"Resampling of image with the CRS of {crs.authid()} in meters is not supported.")
#            target_units = 'meters'
#            # else:
#            #     res = (rlayer_extent.xMaximum() -
#            #            rlayer_extent.xMinimum()) / rlayer.width()
#        self.res = res

#        # handle extent
#        if extent.isNull():
#            extent = rlayer.extent()  # QgsProcessingUtils.combineLayerExtents(layers, crs, context)
#            extent_crs = rlayer.crs()
#        else:
#            if extent.isEmpty():
#                raise QgsProcessingException(
#                    self.tr("The extent for processing can not be empty!"))
#            extent_crs = self.parameterAsExtentCrs(
#                parameters, self.EXTENT, context)
#        # if extent crs != target crs, convert it to target crs
#        if extent_crs != crs:
#            transform = QgsCoordinateTransform(
#                extent_crs, crs, context.transformContext())
#            # extent = transform.transformBoundingBox(extent)
#            # to ensure coverage of the transformed extent
#            # convert extent to polygon, transform polygon, then get boundingBox of the new polygon
#            extent_polygon = QgsGeometry.fromRect(extent)
#            extent_polygon.transform(transform)
#            extent = extent_polygon.boundingBox()
#            extent_crs = crs

#        # check intersects between extent and rlayer_extent
#        if rlayer.crs() != crs:
#            transform = QgsCoordinateTransform(
#                rlayer.crs(), crs, context.transformContext())
#            rlayer_extent = transform.transformBoundingBox(
#                rlayer.extent())
#        else:
#            rlayer_extent = rlayer.extent()
#        if not rlayer_extent.intersects(extent):
#            raise QgsProcessingException(
#                self.tr("The extent for processing is not intersected with the input image!"))

#        img_width_in_extent = round(
#            (extent.xMaximum() - extent.xMinimum())/self.res)
#        img_height_in_extent = round(
#            (extent.yMaximum() - extent.yMinimum())/self.res)

#        # Send some information to the user
#        feedback.pushInfo(
#            f'Layer path: {rlayer_data_provider.dataSourceUri()}')
#        # feedback.pushInfo(
#        #     f'Layer band scale: {rlayer_data_provider.bandScale(self.selected_bands[0])}')
#        feedback.pushInfo(f'Layer name: {rlayer.name()}')
#        if rlayer.crs().authid():
#            feedback.pushInfo(f'Layer CRS: {rlayer.crs().authid()}')
#        else:
#            feedback.pushInfo(
#                f'Layer CRS in WKT format: {rlayer.crs().toWkt()}')
#        feedback.pushInfo(
#            f'Layer pixel size: {rlayer.rasterUnitsPerPixelX()}, {rlayer.rasterUnitsPerPixelY()} {layer_units}')

#        feedback.pushInfo(f'Bands selected: {self.selected_bands}')

#        if crs.authid():
#            feedback.pushInfo(f'Target CRS: {crs.authid()}')
#        else:
#            feedback.pushInfo(f'Target CRS in WKT format: {crs.toWkt()}')
#        feedback.pushInfo(f'Target resolution: {self.res} {target_units}')
#        feedback.pushInfo(
#            (f'Processing extent: minx:{extent.xMinimum():.6f}, maxx:{extent.xMaximum():.6f},'
#             f'miny:{extent.yMinimum():.6f}, maxy:{extent.yMaximum():.6f}'))
#        feedback.pushInfo(
#            (f'Processing image size: (width {img_width_in_extent}, '
#             f'height {img_height_in_extent})'))

#        self.rlayer_path = rlayer.dataProvider().dataSourceUri()

#        feedback.pushInfo(f'Selected bands: {self.selected_bands}')

#        ## passing parameters to self once everything has been processed
#        self.extent = extent
#        self.rlayer = rlayer
#        self.crs = crs


#    # used to handle any thread-sensitive cleanup which is required by the algorithm.
#    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
#        return {}


#    def tr(self, string):
#        """
#        Returns a translatable string with the self.tr() function.
#        """
#        return QCoreApplication.translate('Processing', string)

#    def createInstance(self):
#        return RFAlgorithm()

#    def name(self):
#        """
#        Returns the algorithm name, used for identifying the algorithm. This
#        string should be fixed for the algorithm, and must not be localised.
#        The name should be unique within each provider. Names should contain
#        lowercase alphanumeric characters only and no spaces or other
#        formatting characters.
#        """
#        return 'Random_forest'

#    def displayName(self):
#        """
#        Returns the translated algorithm name, which should be used for any
#        user-visible display of the algorithm name.
#        """
#        return self.tr('Random_forest')

#    def group(self):
#        """
#        Returns the name of the group this algorithm belongs to. This string
#        should be localised.
#        """
#        return self.tr('')

#    def groupId(self):
#        """
#        Returns the unique ID of the group this algorithm belongs to. This
#        string should be fixed for the algorithm, and must not be localised.
#        The group id should be unique within each provider. Group id should
#        contain lowercase alphanumeric characters only and no spaces or other
#        formatting characters.
#        """
#        return ''

#    def shortHelpString(self):
#        """
#        Returns a localised short helper string for the algorithm. This string
#        should provide a basic description about what the algorithm does and the
#        parameters and outputs associated with it..
#        """
#        return self.tr("Compute Random forest using input template")

#    def icon(self):
#        return 'E'




