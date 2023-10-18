# Bring module source into scope
from . import observer
from .observer import *

# Expose module internals to upstream imports
__all__ = []
__all__.extend(observer.__all__)
