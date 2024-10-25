"""
This QGIS plugin requires some Python packages to be installed and available.
This tool allows to install them in a local directory, if they are not installed yet.
"""

import importlib
import logging
import os
import subprocess
import sys
import traceback
import urllib
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import List

from PyQt5.QtWidgets import (
    QAction,
    QToolBar,
    QDialog,
    QTextBrowser,
    QApplication,
    QMessageBox,
    QDialog
)
from PyQt5.QtCore import pyqtSignal, QObject
from qgis.core import QgsApplication
from qgis.gui import QgisInterface

from qgis.PyQt import QtCore, uic
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QCloseEvent
# from qgis.PyQt.QtWidgets import QDialog, QMessageBox, QTextBrowser
from ...icons import (QIcon_EncoderTool, 
                    QIcon_ReductionTool, 
                    QIcon_ClusterTool, 
                    QIcon_SimilarityTool, 
                    QIcon_RandomforestTool,
                    )

PLUGIN_NAME = "iamap"

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.realpath(os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}')


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'packages_installer_dialog.ui'))

_ERROR_COLOR = '#ff0000'

if sys.platform == "linux" or sys.platform == "linux2":
    PYTHON_EXECUTABLE_PATH = sys.executable
elif sys.platform == "darwin":  # MacOS
    PYTHON_EXECUTABLE_PATH = str(Path(sys.prefix) / 'bin' / 'python3')  # sys.executable yields QGIS in macOS
elif sys.platform == "win32":
    PYTHON_EXECUTABLE_PATH = 'python'  # sys.executable yields QGis.exe in Windows
else:
    raise Exception("Unsupported operating system!")


@dataclass
class PackageToInstall:
    name: str
    version: str
    import_name: str  # name while importing package

    def __str__(self):
        return f'{self.name}{self.version}'


class PackagesInstallerDialog(QDialog, FORM_CLASS):
    """
    Dialog witch controls the installation process of packages.
    UI design defined in the `packages_installer_dialog.ui` file.
    """

    signal_log_line = pyqtSignal(str)  # we need to use signal because we cannot edit GUI from another thread

    INSTALLATION_IN_PROGRESS = False  # to make sure we will not start the installation twice

    def __init__(self, iface, packages_to_install, device, parent=None):
        super(PackagesInstallerDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.tb = self.textBrowser_log  # type: QTextBrowser
        self.packages_to_install=packages_to_install
        self.device=device
        self._create_connections()
        self._setup_message()
        self.aborted = False
        self.thread = None

    def move_to_top(self):
        """ Move the window to the top.
        Although if installed from plugin manager, the plugin manager will move itself to the top anyway.
        """
        self.setWindowState((self.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)

        if sys.platform == "linux" or sys.platform == "linux2":
            pass
        elif sys.platform == "darwin":  # MacOS
            self.raise_()  # FIXME: this does not really work, the window is still behind the plugin manager
        elif sys.platform == "win32":
            self.activateWindow()
        else:
            raise Exception("Unsupported operating system!")

    def _create_connections(self):
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_install_packages.clicked.connect(self._run_packages_installation)
        self.signal_log_line.connect(self._log_line)

    def _log_line(self, txt):
        txt = txt \
            .replace('  ', '&nbsp;&nbsp;') \
            .replace('\n', '<br>')
        self.tb.append(txt)

    def log(self, txt):
        self.signal_log_line.emit(txt)

    def _setup_message(self) -> None:
          
        self.log(f'<h2><span style="color: #000080;"><strong>  '
                 f'Plugin {PLUGIN_NAME} - Packages installer </strong></span></h2> \n'
                 f'\n'
                 f'<b>This plugin requires the following Python packages to be installed:</b>')
        
        for package in self.packages_to_install:
            self.log(f'\t- {package.name}{package.version}')

        self.log('\n\n'
                 f'If this packages are not installed in the global environment '
                 f'(or environment in which QGIS is started) '
                 f'you can install these packages in the local directory (which is included to the Python path).\n\n'
                 f'This Dialog does it for you! (Though you can still install these packages manually instead).\n'
                 f'<b>Please click "Install packages" button below to install them automatically, </b>'
                 f'or "Test and Close" if you installed them manually...\n')

    def _run_packages_installation(self):
        if self.INSTALLATION_IN_PROGRESS:
            self.log(f'Error! Installation already in progress, cannot start again!')
            return
        self.aborted = False
        self.INSTALLATION_IN_PROGRESS = True
        self.thread = Thread(target=self._install_packages)
        self.thread.start()

    def _install_packages(self) -> None:
        self.log('\n\n')
        self.log('=' * 60)
        self.log(f'<h3><b>Attempting to install required packages...</b></h3>')
        os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)

        self._install_pip_if_necessary()

        self.log(f'<h3><b>Attempting to install required packages...</b></h3>\n')
        try:
            self._pip_install_packages(self.packages_to_install)
        except Exception as e:
            msg = (f'\n <span style="color: {_ERROR_COLOR};"><b> '
                   f'Packages installation failed with exception: {e}!\n'
                   f'Please try to install the packages again. </b></span>'
                   f'\nCheck if there is no error related to system packages, '
                   f'which may be required to be installed by your system package manager, e.g. "apt". '
                   f'Copy errors from the stack above and google for possible solutions. '
                   f'Please report these as an issue on the plugin repository tracker!')
            self.log(msg)

        # finally, validate the installation, if there was no error so far...
        self.log('\n\n <b>Installation of required packages finished. Validating installation...</b>')
        self._check_packages_installation_and_log()
        self.INSTALLATION_IN_PROGRESS = False

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self.aborted = True
        if self._check_packages_installation_and_log():
            res = QMessageBox.information(self.iface.mainWindow(),
                                       f'{PLUGIN_NAME} - Installation done !',
                                       'Restart QGIS for the plugin to load properly.',
                                       QMessageBox.Ok)
            if res == QMessageBox.Ok:
                log_msg = 'User accepted to restart QGIS'
                event.accept()
            return

        res = QMessageBox.question(self.iface.mainWindow(),
                                   f'{PLUGIN_NAME} - skip installation?',
                                   'Are you sure you want to abort the installation of the required python packages? '
                                   'The plugin may not function correctly without them!',
                                   QMessageBox.No, QMessageBox.Yes)
        log_msg = 'User requested to close the dialog, but the packages are not installed correctly!\n'
        if res == QMessageBox.Yes:
            log_msg += 'And the user confirmed to close the dialog, knowing the risk!'
            event.accept()
        else:
            log_msg += 'The user reconsidered their decision, and will try to install the packages again!'
            event.ignore()
        log_msg += '\n'
        self.log(log_msg)

    def _install_pip_if_necessary(self):
        """
        Install pip if not present.
        It happens e.g. in flatpak applications.

        TODO - investigate whether we can also install pip in local directory
        """

        self.log(f'<h4><b>Making sure pip is installed...</b></h4>')
        if check_pip_installed():
            self.log(f'<em>Pip is installed, skipping installation...</em>\n')
            return

        install_pip_command = [PYTHON_EXECUTABLE_PATH, '-m', 'ensurepip']
        self.log(f'<em>Running command to install pip: \n  $ {" ".join(install_pip_command)} </em>')
        with subprocess.Popen(install_pip_command,
                              stdout=subprocess.PIPE,
                              universal_newlines=True,
                              stderr=subprocess.STDOUT,
                              env={'SETUPTOOLS_USE_DISTUTILS': 'stdlib'}) as process:
            try:
                self._do_process_output_logging(process)
            except InterruptedError as e:
                self.log(str(e))
                return False

        if process.returncode != 0:
            msg = (f'<span style="color: {_ERROR_COLOR};"><b>'
                   f'pip installation failed! Consider installing it manually.'
                   f'<b></span>')
            self.log(msg)
        self.log('\n')

    def _pip_install_packages(self, packages: List[PackageToInstall]) -> None:
        cmd = [PYTHON_EXECUTABLE_PATH, '-m', 'pip', 'install', '-U', f'--target={PACKAGES_INSTALL_DIR}']               
        cmd_string = ' '.join(cmd)
        
        for pck in packages:
            if ("index-url") not in pck.version:
                cmd.append(f" {pck}")
                cmd_string += f" {pck}"
            
            elif pck.name == 'torch':
                torch_url = pck.version.split("index-url ")[-1]
        
                cmd_torch = [PYTHON_EXECUTABLE_PATH, '-m', 'pip', 'install', '-U', f'--target={PACKAGES_INSTALL_DIR}', 'torch', f"--index-url={torch_url}"] 
                cmd_torch_string = ' '.join(cmd_torch)

                self.log(f'<em>Running command: \n  $ {cmd_torch_string} </em>')
                with subprocess.Popen(cmd_torch,
                                    stdout=subprocess.PIPE,
                                    universal_newlines=True,
                                    stderr=subprocess.STDOUT) as process:
                    self._do_process_output_logging(process)


        self.log(f'<em>Running command: \n  $ {cmd_string} </em>')
        with subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              universal_newlines=True,
                              stderr=subprocess.STDOUT) as process:
            self._do_process_output_logging(process)

        if process.returncode != 0:
            raise RuntimeError('Installation with pip failed')

        msg = (f'\n<b>'
               f'Packages installed correctly!'
               f'<b>\n\n')
        self.log(msg)

    def _do_process_output_logging(self, process: subprocess.Popen) -> None:
        """
        :param process: instance of 'subprocess.Popen'
        """
        for stdout_line in iter(process.stdout.readline, ""):
            if stdout_line.isspace():
                continue
            txt = f'<span style="color: #999999;">{stdout_line.rstrip(os.linesep)}</span>'
            self.log(txt)
            if self.aborted:
                raise InterruptedError('Installation aborted by user')

    def _check_packages_installation_and_log(self) -> bool:
        packages_ok = are_packages_importable(self.device)
        self.pushButton_install_packages.setEnabled(not packages_ok)

        if packages_ok:
            msg1 = f'All required packages are importable! You can close this window now! Be sure to restart QGIS.'
            self.log(msg1)
            return True

        try:
            import_packages(self.device)
            raise Exception("Unexpected successful import of packages?!? It failed a moment ago, we shouldn't be here!")
        except Exception:
            msg_base = '<b>Python packages required by the plugin could not be loaded due to the following error:</b>'
            logging.exception(msg_base)
            tb = traceback.format_exc()
            msg1 = (f'<span style="color: {_ERROR_COLOR};">'
                    f'{msg_base} \n '
                    f'{tb}\n\n'
                    f'<b>Please try installing the packages again.<b>'
                    f'</span>')
            self.log(msg1)

        return False

def get_pytorch_version(cuda_version):
    # Map CUDA versions to PyTorch versions
    ## cf. https://pytorch.org/get-started/locally/
    cuda_to_pytorch = {
        "11.8": " --index-url https://download.pytorch.org/whl/cu118",
        "12.1": "",
        "12.4": " --index-url https://download.pytorch.org/whl/cu124",
        "12.6": " --index-url https://download.pytorch.org/whl/cu124",
    }
    return cuda_to_pytorch.get(cuda_version, None)


def get_packages_to_install(device):

    requirements_path = os.path.join(PLUGIN_ROOT_DIR, 'requirements.txt')
    packages_to_install = []

    if device == 'cpu':
        pass

    else :
        if device == 'amd':
            packages_to_install.append(
                    PackageToInstall(
                        name='torch', 
                        version=' --index-url https://download.pytorch.org/whl/rocm6.1', 
                        import_name='torch'
                        )
                    )

        else:
            packages_to_install.append(
                    PackageToInstall(
                        name='torch', 
                        version=get_pytorch_version(device), 
                        import_name='torch'
                        )
                    )
        


    with open(requirements_path, 'r') as f:
        raw_txt = f.read()

    libraries_versions = {}

    for line in raw_txt.split('\n'):
        if line.startswith('#') or not line.strip():
            continue

        line = line.split(';')[0]

        if '==' in line:
            lib, version = line.split('==')
            libraries_versions[lib] = '==' + version
        elif '>=' in line:
            lib, version = line.split('>=')
            libraries_versions[lib] = '>=' + version
        elif '<=' in line:
            lib, version = line.split('<=')
            libraries_versions[lib] = '<=' + version
        else:
            libraries_versions[line] = ''


    for lib, version in libraries_versions.items():

        import_name = lib[:-1]

        if lib == 'scikit-learn ':
            import_name = 'sklearn'
        if lib == 'umap-learn ':
            import_name = 'umap'

        packages_to_install.append(
                PackageToInstall(
                    name=lib, 
                    version=version, 
                    import_name=import_name
                    )
                )
    return packages_to_install



def import_package(package: PackageToInstall):
    importlib.import_module(package.import_name)


def import_packages(device):
    packages_to_install = get_packages_to_install(device)
    for package in packages_to_install:
        import_package(package)


def are_packages_importable(device) -> bool:
    try:
        import_packages(device)
    except Exception:
        logging.exception(f'Python packages required by the plugin could not be loaded due to the following error:')
        return False

    return True


def check_pip_installed() -> bool:
    try:
        subprocess.check_output([PYTHON_EXECUTABLE_PATH, '-m', 'pip', '--version'])
        return True
    except subprocess.CalledProcessError:
        return False


dialog = None
def check_required_packages_and_install_if_necessary(iface, device='cpu'):
    os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
    if PACKAGES_INSTALL_DIR not in sys.path:
        sys.path.append(PACKAGES_INSTALL_DIR)  # TODO: check for a less intrusive way to do this

    if are_packages_importable(device):
        # if packages are importable we are fine, nothing more to do then
        return True

    global dialog
    packages_to_install = get_packages_to_install(device)
    dialog = PackagesInstallerDialog(iface, packages_to_install=packages_to_install, device=device)
    dialog.setWindowModality(QtCore.Qt.WindowModal)
    dialog.show()
    dialog.move_to_top()
    return False


class IAMapEmpty(QObject):
    execute_iamap = pyqtSignal()

    def __init__(self, iface: QgisInterface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd

    def initProcessing(self):
        # self.provider = IAMapProvider()
        # QgsApplication.processingRegistry().addProvider(self.provider)
        return

    def initGui(self):
        self.initProcessing()

        self.toolbar: QToolBar = self.iface.addToolBar('IAMap Toolbar')
        self.toolbar.setObjectName('IAMapToolbar')
        self.toolbar.setToolTip('IAMap Toolbar')

        self.actionEncoder = QAction(
            QIcon_EncoderTool,
            "Install dependencies and restart QGIS ! - Deep Learning Image Encoder",
            self.iface.mainWindow()
        )
        self.actionReducer = QAction(
            QIcon_ReductionTool,
            "Install dependencies and restart QGIS ! - Reduce dimensions",
            self.iface.mainWindow()
        )
        self.actionCluster = QAction(
            QIcon_ClusterTool,
            "Install dependencies and restart QGIS ! - Cluster raster",
            self.iface.mainWindow()
        )
        self.actionSimilarity = QAction(
            QIcon_SimilarityTool,
            "Install dependencies and restart QGIS ! - Compute similarity",
            self.iface.mainWindow()
        )
        self.actionRF = QAction(
            QIcon_RandomforestTool,
            "Install dependencies and restart QGIS ! - Fit Machine Learning algorithm",
            self.iface.mainWindow()
        )
        self.actionEncoder.setObjectName("mActionEncoder")
        self.actionReducer.setObjectName("mActionReducer")
        self.actionCluster.setObjectName("mActionCluster")
        self.actionSimilarity.setObjectName("mactionSimilarity")
        self.actionRF.setObjectName("mactionRF")

        self.actionEncoder.setToolTip(
            "Install dependencies and restart QGIS ! - Encode a raster with a deep learning backbone")
        self.actionReducer.setToolTip(
            "Install dependencies and restart QGIS ! - Reduce raster dimensions")
        self.actionCluster.setToolTip(
            "Install dependencies and restart QGIS ! - Cluster raster")
        self.actionSimilarity.setToolTip(
            "Install dependencies and restart QGIS ! - Compute similarity")
        self.actionRF.setToolTip(
            "Install dependencies and restart QGIS ! - Fit ML model")

        # self.actionEncoder.triggered.connect()
        # self.actionReducer.triggered.connect()
        # self.actionCluster.triggered.connect()
        # self.actionSimilarity.triggered.connect()
        # self.actionRF.triggered.connect()

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
