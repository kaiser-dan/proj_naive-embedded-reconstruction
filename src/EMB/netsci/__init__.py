# Setup module-shared objects
from EMB.utils import logger
LOGGER = logger.get_module_logger(
    name="networks",
    filename=f".logs/networks_{logger.get_today(time=False)}.log",
    mode='a',
    file_level=10,
    console_level=30)
__all__ = ["LOGGER"]

# Bring module source into scope
from . import models
from . import utils

from .models import *
from .utils import *

# Expose module internals to upstream imports
# __all__.extend(preprocessing.__all__)
