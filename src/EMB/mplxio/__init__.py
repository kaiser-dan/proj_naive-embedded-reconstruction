# Setup module-shared objects
from EMB.utils import logger
LOGGER = logger.get_module_logger(
    name="mplxio",
    filename=f".logs/mplxio_{logger.get_today(time=False)}.log",
    mode='a',
    file_level=10,
    console_level=30)
__all__ = ["LOGGER"]

# Bring module source into scope
from . import readers
from . import writers
from .readers import *
from .writers import *

# Expose module internals to upstream imports
__all__.extend(readers.__all__)
__all__.extend(writers.__all__)
