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
__all__ = []
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
