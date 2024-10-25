import os
import inspect

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]

def classFactory(iface):
    from .dialogs.check_gpu import has_gpu
    from .dialogs.packages_installer import packages_installer_dialog
    device = has_gpu()
    packages_installed_allready = packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface, device=device)
    # packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)
    if packages_installed_allready:
        from .iamap import IAMap
        return IAMap(iface, cmd_folder)

    else:
        from .dialogs.packages_installer.packages_installer_dialog import IAMapEmpty
        return IAMapEmpty(iface, cmd_folder)
