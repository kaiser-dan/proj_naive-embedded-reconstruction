# Setup module-shared objects
from EMB.utils import logger
LOGGER = logger.get_module_logger(
    name="classifiers",
    filename=f".logs/classifiers_{logger.get_today(time=False)}.log",
    mode='a',
    file_level=40,
    console_level=30)
__all__ = ["LOGGER"]

# Bring module source into scope
from . import logreg
from . import features
from .logreg import *
from .features import *

# Expose module internals to upstream imports
__all__.extend(logreg.__all__)
__all__.extend(features.__all__)
