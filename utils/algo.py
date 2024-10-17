import os
import tempfile
import numpy as np
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
from rasterio.enums import Resampling



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
