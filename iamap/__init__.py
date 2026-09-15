import os
import inspect

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]


def classFactory(iface):
    from .dialogs.check_gpu import has_gpu
    from .dialogs.packages_installer import packages_installer_dialog
    from qgis.core import QgsSettings

    ## we find out if a GPU is available and if all dependencies are here
    device = has_gpu()
    packages_installed_allready = (
        packages_installer_dialog.check_required_packages(
            iface=iface, device=device
        )
    )
    
    ## if dependencies are installed, the plugin launches
    if packages_installed_allready:
        from .iamap import IAMap
        return IAMap(iface, cmd_folder)

    ## else the dialog pop up shows with an empty plugin behind
    else:
        ## if the user does not want the install pop up to appear, it does not show again
        settings = QgsSettings()
        dont_show_again = settings.value("iamap/dont_show_install_pop_up", False, type=bool)

        if dont_show_again:
            from .dialogs.packages_installer.packages_installer_dialog import IAMapEmpty
            return IAMapEmpty(iface, cmd_folder)

        from .dialogs.packages_installer.packages_installer_dialog import IAMapEmpty
        from .dialogs.packages_installer.packages_installer_dialog import show_install_pop_up
        show_install_pop_up(iface, device)
        return IAMapEmpty(iface, cmd_folder)
