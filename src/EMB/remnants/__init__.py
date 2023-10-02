# Setup module-shared objects
from EMB.utils import logger
LOGGER = logger.get_module_logger(
    name="remnants",
    filename=f".logs/remnants_{logger.get_today(time=False)}.log",
    mode='a',
    file_level=10,
    console_level=30)
__all__ = ["LOGGER"]

# Bring module source into scope
from . import observer
from .observer import *

# Expose module internals to upstream imports
__all__.extend(observer.__all__)
