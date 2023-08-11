from embmplxrec.utils import get_module_logger, get_today

LOGGER = get_module_logger(
    name=__name__,
    filename=f".logs/features_{get_today(time=True)}.log",
    mode='a',
    file_level=10,
    console_level=30)

__all__ = ["LOGGER"]

from . import distances
from . import degrees
from . import formatters

from .distances import *
from .degrees import *
from .formatters import *

__all__.extend([distances.__all__, degrees.__all__, formatters.__all__])
