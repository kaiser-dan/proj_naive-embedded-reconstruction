import sys
from rich.traceback import install
from loguru import logger

# Set pretty printy tracebacks
install(show_locals=False)

# Set global logger
logger.remove(0)
logger.add(sys.stderr, level="INFO")
logger.add(".logs/emb_log.log", compression="zip", level="DEBUG")
