from . import embedding
from . import helpers
from . import HOPE
from . import Isomap
from . import N2V
from . import LE

from .embedding import *
from .HOPE import *
from .Isomap import *
from .N2V import *
from .LE import *

__all__ = []
# __all__.extend(*[embedding.__all__, HOPE.__all__, Isomap.__all__, N2V.__all__, LE.__all__])

from embmplxrec.utils import get_module_logger, get_today
LOGGER = get_module_logger(
    name=__name__,
    filename=f".logs/embeddings_{get_today(time=True)}.log",
    mode='a',
    file_level=10,
    console_level=10)

__all__.append(["LOGGER"])