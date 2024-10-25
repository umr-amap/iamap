import os
import ast
import numpy as np
from typing import Dict, Any
import joblib
import json

import rasterio
from rasterio import windows
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterString,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterDefinition
                       )

from .icons import QIcon_RandomforestTool
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
from sklearn.metrics import (
                            accuracy_score,
                            precision_score, 
                            recall_score, 
                            f1_score, 
                            confusion_matrix, 
                            classification_report
                            )
from sklearn.metrics import (
                            mean_absolute_error, 
                            mean_squared_error, 
                            r2_score,
                            )


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
    SAVE_MODEL = 'SAVE_MODEL'
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
            QgsProcessingParameterVectorLayer(
                name=self.TEMPLATE,
                description=self.tr(
                    'Input shapefile path for training data set for random forest (if no test data_set, will be devised in train and test)'),
            # defaultValue=os.path.join(self.cwd,'assets',self.DEFAULT_TEMPLATE),
            ),
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
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
                defaultValue = '',
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

        save_param = QgsProcessingParameterBoolean(
                self.SAVE_MODEL,
                self.tr("Save model after fit."),
                defaultValue=True
                )

        for param in (
                nfold_param,
                save_param,
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

        if self.test_gdf is not None:
            metrics_dict = self.train_test_loop(feedback)
            self.best_model = self.model

        if self.do_kfold:
            best_metric = 0
            best_metrics_dict = {}
            for fold in sorted(self.gdf[self.fold_col].unique()): # pyright: ignore[reportAttributeAccessIssue]
                feedback.pushInfo(f'==== Fold {fold} ====')
                self.test_gdf = self.gdf.loc[self.gdf[self.fold_col] == fold]
                self.train_gdf = self.gdf.loc[self.gdf[self.fold_col] != fold]
                metrics_dict = self.train_test_loop(feedback)

                if 'accuracy' in metrics_dict.keys():
                    used_metric = metrics_dict['accuracy']
                if 'r2' in metrics_dict.keys():
                    used_metric = metrics_dict['accuracy']
                if used_metric >= best_metric:
                    best_metric = used_metric
                    best_metrics_dict = metrics_dict
                    self.best_model = self.model

        if (self.test_gdf is None) and not self.do_kfold:
            train_set, train_gts = self.get_raster(mode='train')
            self.model.fit(train_set, train_gts)
            feedback.pushWarning(f'No test set was provided and no cross-validation is done, unable to assess model quality !')
            self.best_model = self.model

        
        feedback.pushInfo(f'Fitting done, saving model\n')
        save_file = f'{self.method_name}.pkl'.lower()
        metrics_save_file = f'{self.method_name}-metrics.json'.lower()
        if self.save_model:
            out_path = os.path.join(self.output_dir, save_file)
            joblib.dump(self.best_model, out_path)
            with open(os.path.join(self.output_dir, metrics_save_file), "w") as json_file:
                ## confusion matrix is a np array that does not fit in a json
                best_metrics_dict.pop('conf_matrix', None)
                best_metrics_dict.pop('class_report', None)
                json.dump(best_metrics_dict, json_file, indent=4)

        self.infer_model(feedback)

        return {'OUTPUT_RASTER':self.dst_path, 'OUTPUT_LAYER_NAME':self.layer_name, 'USED_SHP':self.used_shp_path}


    def train_test_loop(self, feedback):
        train_set, train_gts = self.get_raster(mode='train')
        test_set, test_gts = self.get_raster(mode='test')

        self.model.fit(train_set, train_gts)
        predictions = self.model.predict(test_set)
        return self.get_metrics(test_gts,predictions, feedback)


    def infer_model(self, feedback):

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
            np.nan_to_num(inf_raster) # NaN to zero after normalisation

            proj_img = self.best_model.predict(inf_raster)

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


    def process_ml_shp(self, parameters, context, feedback):

        template_test = self.parameterAsVectorLayer(
            parameters, self.TEMPLATE_TEST, context)
        feedback.pushInfo(f'template_test: {template_test}')

        self.test_gdf=None

        if template_test is not None :
            random_samples = self.parameterAsInt(
                parameters, self.RANDOM_SAMPLES, context)

            gdf = gpd.read_file(template_test.dataProvider().dataSourceUri())
            gdf = gdf.to_crs(self.crs.toWkt())

            feedback.pushInfo(f'before samples: {len(gdf)}')
            ## get random samples if geometry is not point based
            gdf = get_random_samples_in_gdf(gdf, random_samples, seed=self.seed)

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

        self.save_model = self.parameterAsBoolean(
                parameters, self.SAVE_MODEL, context)
        self.do_kfold = self.parameterAsBoolean(
            parameters, self.DO_KFOLDS, context)
        gt_col = self.parameterAsString(
            parameters, self.GT_COL, context)
        fold_col = self.parameterAsString(
            parameters, self.FOLD_COL, context)
        nfolds = self.parameterAsInt(
            parameters, self.NFOLDS, context)
        str_kwargs = self.parameterAsString(
                parameters, self.SK_PARAM, context)

        ## If a fold column is provided, this defines the folds. Otherwise, random split
        ## check that no column with name 'fold' exists, otherwise we use 'fold1' etc..
        ## we also make a new column containing gt values
        self.fold_col = get_unique_col_name(self.gdf, 'fold')
        self.gt_col = get_unique_col_name(self.gdf, 'gt')

        ## Instantiate model
        if str_kwargs != '':
            self.passed_kwargs = ast.literal_eval(str_kwargs)
        else:
            self.passed_kwargs = {}

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


        ## different behaviours if we are doing classification or regression
        ## If classification, we create a new col with unique integers for each classes
        ## to ease inference
        self.task_type = check_model_type(self.model)
        
        if self.task_type == 'classification':
            self.out_dtype = 'int8'
            self.gdf[self.gt_col] = pd.factorize(self.gdf[gt_col])[0] # unique int for each class
        else:
            self.gt_col = gt_col


        ## If no test set is provided and the option to perform kfolds is true, we perform kfolds
        if self.test_gdf == None and self.do_kfold:
            if fold_col.strip() != '' :
                self.gdf[self.fold_col] = self.gdf[fold_col]
            else:
                np.random.seed(self.seed)
                self.gdf[self.fold_col] = np.random.randint(1, nfolds + 1, size=len(self.gdf))
        ## Else, self.gdf is the train set
        else:
            self.train_gdf = self.gdf

        feedback.pushInfo(f'saving modified dataframe to: {self.used_shp_path}')
        self.gdf.to_file(self.used_shp_path)


    def get_raster(self, mode='train'):

        if mode == 'train':
            gdf = self.train_gdf
        else:
            gdf = self.test_gdf

        with rasterio.open(self.rlayer_path) as ds:

            gdf = gdf.to_crs(ds.crs)
            pixel_values = []
            gts = []

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
                gts.append(data[self.gt_col])


        return np.asarray(pixel_values), np.asarray(gts)


    def update_kwargs(self, kwargs_dict):

        for key, value in self.passed_kwargs.items():
            if key in kwargs_dict.keys():
                kwargs_dict[key] = value
        
        kwargs_dict['random_state'] = self.seed

        return kwargs_dict


    def get_metrics(self, test_gts, predictions, feedback):

        metrics_dict = {}
        if self.task_type == 'classification':
            # Evaluate the model
            metrics_dict['accuracy'] = accuracy_score(test_gts, predictions)
            metrics_dict['precision'] = precision_score(test_gts, predictions, average='weighted')  # Modify `average` for multiclass if necessary
            metrics_dict['recall'] = recall_score(test_gts, predictions, average='weighted')
            metrics_dict['f1'] = f1_score(test_gts, predictions, average='weighted')
            metrics_dict['conf_matrix'] = confusion_matrix(test_gts, predictions)
            metrics_dict['class_report'] = classification_report(test_gts, predictions)


        elif self.task_type == 'regression':

            metrics_dict['mae'] = mean_absolute_error(test_gts, predictions)
            metrics_dict['mse'] = mean_squared_error(test_gts, predictions)
            metrics_dict['rmse'] = np.sqrt(metrics_dict['mse'])
            metrics_dict['r2'] = r2_score(test_gts, predictions)

        else:
            feedback.pushWarning('Unable to evaluate the model !!')

        for key, value in metrics_dict.items():
            feedback.pushInfo(f'{key}:\t {value}')

        return metrics_dict
        

    def get_algorithms(self):
        required_methods = ['fit', 'predict']
        ensemble_algos = get_sklearn_algorithms_with_methods(ensemble, required_methods)
        neighbors_algos = get_sklearn_algorithms_with_methods(neighbors, required_methods)
        return sorted(ensemble_algos+neighbors_algos)


    def get_help_sk_methods(self):
        """
        Generate help string with default arguments of supported sklearn algorithms.
        """
            
        help_str = '\n\n Here are the default arguments of the supported algorithms:\n\n'

        required_methods = ['fit', 'predict']

        ensemble_algos = get_sklearn_algorithms_with_methods(ensemble, required_methods)
        for algo in ensemble_algos :
            args = get_arguments(ensemble,algo)
            help_str += f'- {algo}:\n'
            help_str += f'{args}\n'

        neighbors_algos = get_sklearn_algorithms_with_methods(neighbors, required_methods)
        for algo in neighbors_algos :
            args = get_arguments(neighbors,algo)
            help_str += f'- {algo}:\n'
            help_str += f'{args}\n'

        return help_str

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
        return self.tr(f"Fit a Machine Learning model using input template. Only RandomForestClassifier is throughfully tested. \n{self.get_help_sk_methods()}")

    def icon(self):
        return QIcon_RandomforestTool
