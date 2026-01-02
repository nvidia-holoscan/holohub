import logging
from types import ModuleType

logger = logging.getLogger(__name__)

try:
    import cupy  # type: ignore

    logger.info("CuPy is available for GPU computations.")
except ImportError:
    cupy = None
    logger.info("CuPy is not available.")


def get_array_module(use_gpu: bool = False) -> tuple[ModuleType, bool]:
    """Get the appropriate array module (NumPy or CuPy) based on GPU availability.
    Args:
        use_gpu: Whether to attempt to use GPU arrays.
    Returns:
        A tuple containing the array module and a boolean indicating if GPU is used.
    """
    # GPU-only refactor: reconstruction pipeline should always run on CuPy when requested.
    if not use_gpu:
        raise ValueError("GPU-only pipeline: get_array_module(use_gpu=False) is not supported")
    if cupy is None:
        raise ImportError("GPU-only pipeline requires CuPy, but it is not installed")
    device_count = cupy.cuda.runtime.getDeviceCount()
    if device_count <= 0:
        raise RuntimeError("GPU-only pipeline requested, but no CUDA devices were found")
    return cupy, True
