# Setup module-shared objects
from EMB.utils import logger

LOGGER = logger.get_module_logger(
    name="embeddings",
    filename=f".logs/embeddings_{logger.get_today(time=False)}.log",
    mode="a",
    file_level=30,
    console_level=20,
)
__all__ = ["LOGGER"]

# Bring module source into scope
from . import vectors
from . import N2V
from . import LE
from . import Isomap
from . import HOPE

from .vectors import *
from .N2V import *
from .LE import *
from .Isomap import *
from .HOPE import *

# Expose module internals to upstream imports
__all__.extend(vectors.__all__)
__all__.extend(N2V.__all__)
__all__.extend(LE.__all__)
__all__.extend(Isomap.__all__)
__all__.extend(HOPE.__all__)

# Notate accepted embeddings
ACCEPTED_EMBEDDINGS = [
    "N2V",
    "LE",
    "Isomap",
    "HOPE",
]
__all__.extend(ACCEPTED_EMBEDDINGS)
