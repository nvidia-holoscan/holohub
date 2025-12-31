import logging
from types import ModuleType

import numpy

logger = logging.getLogger(__name__)

try:
    import cupy  # type: ignore

    logger.info("CuPy is available for GPU computations.")
except ImportError:
    cupy = None
    logger.info("CuPy is not available; falling back to NumPy for CPU computations.")


def get_array_module(use_gpu: bool = False) -> tuple[ModuleType, bool]:
    """Get the appropriate array module (NumPy or CuPy) based on GPU availability.
    Args:
        use_gpu: Whether to attempt to use GPU arrays.
    Returns:
        A tuple containing the array module and a boolean indicating if GPU is used.
    """
    if use_gpu and cupy is not None:
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count > 0:
            return cupy, True
        else:
            logger.warning("No GPU devices found; using NumPy instead.")

    return numpy, False
