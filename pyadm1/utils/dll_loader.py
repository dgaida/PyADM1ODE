import os
import platform
import logging

logger = logging.getLogger(__name__)


def load_dlls():
    """
    Load C# DLLs required for PyADM1 ODE implementation.

    This function initializes the CLR (Common Language Runtime) and adds
    references to all necessary DLLs in the pyadm1/dlls directory.
    This centralized approach ensures all dependencies, including
    'toolbox.dll' and its internal assemblies like 'optim_params',
    are correctly loaded.
    """
    if platform.system() == "Darwin":
        logger.warning("CLR features are not supported on macOS.")
        return False

    try:
        import clr
    except ImportError:
        logger.warning("pythonnet (clr) not found. CLR features will be unavailable.")
        return False

    try:
        dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dlls"))

        # List of DLLs to load. 'toolbox' is crucial for resolving 'optim_params' dependencies.
        dlls = ["biogas", "substrates", "physchem", "plant", "toolbox"]

        for dll in dlls:
            full_path = os.path.join(dll_path, dll)
            if os.path.exists(full_path + ".dll"):
                clr.AddReference(full_path)
                logger.debug(f"Successfully referenced {dll}.dll")
            else:
                logger.warning(f"DLL not found: {full_path}.dll")

        return True
    except Exception as e:
        logger.error(f"Failed to load C# DLLs: {e}")
        return False
