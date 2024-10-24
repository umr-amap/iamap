import processing
from PyQt5.QtWidgets import (
    QAction,
    QToolBar,
    QApplication,
    QDialog
)
from PyQt5.QtCore import pyqtSignal, QObject
from qgis.core import QgsApplication
from qgis.gui import QgisInterface
from .provider import IAMapProvider
from .icons import (QIcon_EncoderTool, 
                    QIcon_ReductionTool, 
                    QIcon_ClusterTool, 
                    QIcon_SimilarityTool, 
                    QIcon_RandomforestTool,
                    )


class IAMap(QObject):
    execute_iamap = pyqtSignal()

    def __init__(self, iface: QgisInterface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd

    def initProcessing(self):
        self.provider = IAMapProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

        self.toolbar: QToolBar = self.iface.addToolBar('IAMap Toolbar')
        self.toolbar.setObjectName('IAMapToolbar')
        self.toolbar.setToolTip('IAMap Toolbar')

        self.actionEncoder = QAction(
            QIcon_EncoderTool,
            "Deep Learning Image Encoder",
            self.iface.mainWindow()
        )
        self.actionReducer = QAction(
            QIcon_ReductionTool,
            "Reduce dimensions",
            self.iface.mainWindow()
        )
        self.actionCluster = QAction(
            QIcon_ClusterTool,
            "Cluster raster",
            self.iface.mainWindow()
        )
        self.actionSimilarity = QAction(
            QIcon_SimilarityTool,
            "Compute similarity",
            self.iface.mainWindow()
        )
        self.actionRF = QAction(
            QIcon_RandomforestTool,
            "Fit Machine Learning algorithm",
            self.iface.mainWindow()
        )
        self.actionEncoder.setObjectName("mActionEncoder")
        self.actionReducer.setObjectName("mActionReducer")
        self.actionCluster.setObjectName("mActionCluster")
        self.actionSimilarity.setObjectName("mactionSimilarity")
        self.actionRF.setObjectName("mactionRF")

        self.actionEncoder.setToolTip(
            "Encode a raster with a deep learning backbone")
        self.actionReducer.setToolTip(
            "Reduce raster dimensions")
        self.actionCluster.setToolTip(
            "Cluster raster")
        self.actionSimilarity.setToolTip(
            "Compute similarity")
        self.actionRF.setToolTip(
            "Fit ML model")

        self.actionEncoder.triggered.connect(self.encodeImage)
        self.actionReducer.triggered.connect(self.reduceImage)
        self.actionCluster.triggered.connect(self.clusterImage)
        self.actionSimilarity.triggered.connect(self.similarityImage)
        self.actionRF.triggered.connect(self.rfImage)

        self.toolbar.addAction(self.actionEncoder)
        self.toolbar.addAction(self.actionReducer)
        self.toolbar.addAction(self.actionCluster)
        self.toolbar.addAction(self.actionSimilarity)
        self.toolbar.addAction(self.actionRF)

    def unload(self):
        # self.wdg_select.setVisible(False)
        self.iface.removeToolBarIcon(self.actionEncoder)
        self.iface.removeToolBarIcon(self.actionReducer)
        self.iface.removeToolBarIcon(self.actionCluster)
        self.iface.removeToolBarIcon(self.actionSimilarity)
        self.iface.removeToolBarIcon(self.actionRF)

        del self.actionEncoder
        del self.actionReducer
        del self.actionCluster
        del self.actionSimilarity
        del self.actionRF
        del self.toolbar
        QgsApplication.processingRegistry().removeProvider(self.provider)

    def encodeImage(self):
        '''
        '''
        result = processing.execAlgorithmDialog('iamap:encoder', {})
        print(result)
                # Check if algorithm execution was successful
        if result:
            # Retrieve output parameters from the result dictionary
            if 'OUTPUT_RASTER' in result:
                output_raster_path = result['OUTPUT_RASTER']
                output_layer_name = result['OUTPUT_LAYER_NAME']

                # Add the output raster layer to the map canvas
                self.iface.addRasterLayer(str(output_raster_path),output_layer_name)
            else:
                # Handle missing or unexpected output
                print('Output raster not found in algorithm result.')
        else:
            # Handle algorithm execution failure or cancellation
            print('Algorithm execution was not successful.')
        # processing.execAlgorithmDialog('', {})
        # self.close_all_dialogs()


    def reduceImage(self):
        '''
        '''
        result = processing.execAlgorithmDialog('iamap:reduction', {})
        print(result)
                # Check if algorithm execution was successful
        if result:
            # Retrieve output parameters from the result dictionary
            if 'OUTPUT_RASTER' in result:
                output_raster_path = result['OUTPUT_RASTER']
                output_layer_name = result['OUTPUT_LAYER_NAME']

                # Add the output raster layer to the map canvas
                self.iface.addRasterLayer(str(output_raster_path), output_layer_name)
            else:
                # Handle missing or unexpected output
                print('Output raster not found in algorithm result.')
        else:
            # Handle algorithm execution failure or cancellation
            print('Algorithm execution was not successful.')
        # processing.execAlgorithmDialog('', {})


    def clusterImage(self):
        '''
        '''
        result = processing.execAlgorithmDialog('iamap:cluster', {})
        print(result)
                # Check if algorithm execution was successful
        if result:
            # Retrieve output parameters from the result dictionary
            if 'OUTPUT_RASTER' in result:
                output_raster_path = result['OUTPUT_RASTER']
                output_layer_name = result['OUTPUT_LAYER_NAME']

                # Add the output raster layer to the map canvas
                self.iface.addRasterLayer(str(output_raster_path), output_layer_name)
            else:
                # Handle missing or unexpected output
                print('Output raster not found in algorithm result.')
        else:
            # Handle algorithm execution failure or cancellation
            print('Algorithm execution was not successful.')
        # processing.execAlgorithmDialog('', {})


    def similarityImage(self):
        '''
        '''
        result = processing.execAlgorithmDialog('iamap:similarity', {})
        print(result)
                # Check if algorithm execution was successful
        if result:
            # Retrieve output parameters from the result dictionary
            if 'OUTPUT_RASTER' in result:
                output_raster_path = result['OUTPUT_RASTER']
                output_layer_name = result['OUTPUT_LAYER_NAME']
                used_shp = result['USED_SHP']

                # Add the output raster layer to the map canvas
                self.iface.addRasterLayer(str(output_raster_path), output_layer_name)
                self.iface.addVectorLayer(str(used_shp), 'used points', "ogr")
            else:
                # Handle missing or unexpected output
                print('Output raster not found in algorithm result.')
        else:
            # Handle algorithm execution failure or cancellation
            print('Algorithm execution was not successful.')
        # processing.execAlgorithmDialog('', {})
        
    def rfImage(self):
        '''
        '''
        result = processing.execAlgorithmDialog('iamap:ml', {})
        print(result)
                # Check if algorithm execution was successful
        if result:
            # Retrieve output parameters from the result dictionary
            if 'OUTPUT_RASTER' in result:
                output_raster_path = result['OUTPUT_RASTER']
                output_layer_name = result['OUTPUT_LAYER_NAME']
                used_shp = result['USED_SHP']

                # Add the output raster layer to the map canvas
                self.iface.addRasterLayer(str(output_raster_path), output_layer_name)
                self.iface.addVectorLayer(str(used_shp), 'used points', "ogr")
            else:
                # Handle missing or unexpected output
                print('Output raster not found in algorithm result.')
        else:
            # Handle algorithm execution failure or cancellation
            print('Algorithm execution was not successful.')
        # processing.execAlgorithmDialog('', {})



    def close_all_dialogs(self):
        # Get the main QGIS window (QgisInterface)
        qgis_main_window = self.iface.mainWindow()

        # Get all open dialogs associated with the main window
        open_dialogs = qgis_main_window.findChildren(QDialog)

        # Iterate through the open dialogs and close them
        for dialog in open_dialogs:
            # Check if the dialog is visible (to avoid closing hidden dialogs)
            if dialog.isVisible():
                # Close the dialog
                dialog.close()







