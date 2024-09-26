import os
import numpy as np
from pathlib import Path
from typing import Dict, Any
import joblib
import tempfile
import json

import rasterio
from rasterio import windows
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (Qgis,
                       QgsGeometry,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterEnum,
                       QgsCoordinateTransform,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterBand,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterCrs,
                       QgsProcessingParameterDefinition,
                       )


from sklearn.cluster import KMeans

from .utils.misc import get_unique_filename


class ClusterAlgorithm(QgsProcessingAlgorithm):
    """
    """

    INPUT = 'INPUT'
    BANDS = 'BANDS'
    EXTENT = 'EXTENT'
    LOAD = 'LOAD'
    OUTPUT = 'OUTPUT'
    RESOLUTION = 'RESOLUTION'
    CRS = 'CRS'
    CLUSTERS = 'CLUSTERS'
    SUBSET = 'SUBSET'
    METHOD = 'METHOD'
    SAVE_MODEL = 'SAVE_MODEL'
    

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        cwd = Path(__file__).parent.absolute()
        tmp_wd = os.path.join(tempfile.gettempdir(), "iamap_clustering")

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

        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr(
                    'Processing extent (default to the entire image)'),
                optional=True
            )
        )
        self.method_opt = ['K-means', '--Empty--']
        self.addParameter (
            QgsProcessingParameterEnum(
                name = self.METHOD,
                description = self.tr(
                    'Method for the dimension reduction'),
                defaultValue = 0,
                options = self.method_opt,
                
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.CLUSTERS,
                description=self.tr(
                    'Number of target clusters'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue = 5,
                minValue=1,
                maxValue=1024
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


        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr(
                    "Output directory (choose the location that the image features will be saved)"),
            # defaultValue=os.path.join(cwd,'models'),
            defaultValue=tmp_wd,
            )
        )


        for param in (crs_param, res_param, subset_param, save_param):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)



    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        self.process_options(parameters, context, feedback)

        input_bands = [i_band -1 for i_band in self.selected_bands]

        if self.method == 'K-means':
            proj = KMeans(int(self.nclusters))
            save_file = 'kmeans_cluster.pkl'
            params = proj.get_params()

        out_path = os.path.join(self.output_dir, save_file)

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
            raster = raster[:,:,input_bands]

            feedback.pushInfo(f'{raster.shape}')
            feedback.pushInfo(f'{raster.reshape(-1, raster.shape[0]).shape}')

            if self.subset:
                feedback.pushInfo(f'Using a random subset of {self.subset} pixels')
                fit_raster = raster.reshape(-1, raster.shape[-1])
                nsamples = fit_raster.shape[0]
    
                # Generate random indices to select subset_size number of samples
                np.random.seed(42)
                random_indices = np.random.choice(nsamples, size=self.subset, replace=False)
                fit_raster = fit_raster[random_indices,:]
                feedback.pushInfo(f'Starting fit\n')
                proj.fit(fit_raster)
                if self.save_model:
                    joblib.dump(proj, out_path)

                feedback.pushInfo(f'starting inference\n')
                proj_img = proj.predict(raster.reshape(-1, raster.shape[-1]))


            else:
                proj_img = proj.fit_predict(raster.reshape(-1, raster.shape[-1]))
                if self.save_model:
                    joblib.dump(proj, out_path)

            proj_img = proj_img.reshape((raster.shape[0], raster.shape[1],-1))
            height, width, channels = proj_img.shape


            dst_path = os.path.join(self.output_dir,'cluster.tif')
            params_file = os.path.join(self.output_dir, 'cluster_parameters.json')
            
            
            # if os.path.exists(dst_path):
            #         i = 1
            #         while True:
            #             modified_output_file = os.path.join(self.output_dir, f"cluster_{i}.tif")
            #             if not os.path.exists(modified_output_file):
            #                 dst_path = modified_output_file
            #                 break
            #             i += 1
                        
            dst_path, layer_name = get_unique_filename(self.output_dir, 'cluster.tif', 'clustered features')
            if os.path.exists(params_file):
                    i = 1
                    while True:
                        modified_output_file_params = os.path.join(self.output_dir, f"cluster_parameters_{i}.json")
                        if not os.path.exists(modified_output_file_params):
                            params_file = modified_output_file_params
                            break
                        i += 1


            with rasterio.open(dst_path, 'w', driver='GTiff',
                               height=height, width=width, count=channels, dtype='int8',
                               crs=crs, transform=transform) as dst_ds:
                dst_ds.write(np.transpose(proj_img, (2, 0, 1)))
                
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=4)
            feedback.pushInfo(f"Parameters saved to {params_file}")

            parameters['OUTPUT_RASTER']=dst_path

        return {'OUTPUT_RASTER':dst_path, 'OUTPUT_LAYER_NAME':layer_name}

    def process_options(self,parameters, context, feedback):
        self.iPatch = 0
        
        self.feature_dir = ""
        
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

        self.nclusters = self.parameterAsInt(
            parameters, self.CLUSTERS, context)
        self.subset = self.parameterAsInt(
            parameters, self.SUBSET, context)
        method_idx = self.parameterAsEnum(
            parameters, self.METHOD, context)
        self.method = self.method_opt[method_idx]

        res = self.parameterAsDouble(
            parameters, self.RESOLUTION, context)
        crs = self.parameterAsCrs(
            parameters, self.CRS, context)
        extent = self.parameterAsExtent(
            parameters, self.EXTENT, context)
        output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)
        self.save_model = self.parameterAsBoolean(
            parameters, self.SAVE_MODEL, context)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rlayer_data_provider = rlayer.dataProvider()

        # handle crs
        if crs is None or not crs.isValid():
            crs = rlayer.crs()
            """
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
            """

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
        feedback.pushInfo(f'Target resolution: {self.res} {target_units}')
        feedback.pushInfo(
            (f'Processing extent: minx:{extent.xMinimum():.6f}, maxx:{extent.xMaximum():.6f},'
             f'miny:{extent.yMinimum():.6f}, maxy:{extent.yMaximum():.6f}'))
        feedback.pushInfo(
            (f'Processing image size: (width {img_width_in_extent}, '
             f'height {img_height_in_extent})'))

        self.rlayer_path = rlayer.dataProvider().dataSourceUri()

        feedback.pushInfo(f'Selected bands: {self.selected_bands}')

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
        return ClusterAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'cluster'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Clustering')

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
        return self.tr("Cluster a raster.")

    def icon(self):
        return 'E'



