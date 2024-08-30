import os
import inspect

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]

def classFactory(iface):
    from .dialogs.packages_installer import packages_installer_dialog
    packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)
    from .iamap import IAMap
    return IAMap(iface, cmd_folder)
