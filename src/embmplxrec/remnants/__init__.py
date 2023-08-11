from embmplxrec.utils import get_module_logger, get_today

LOGGER = get_module_logger(
    name=__name__,
    filename=f".logs/remnants_{get_today(time=True)}.log",
    mode='a',
    file_level=10,
    console_level=30)

__all__ = ["LOGGER"]

from . import observer
from . import remnant

from .observer import *
from .remnant import *

__all__.extend([observer.__all__, remnant.__all__])