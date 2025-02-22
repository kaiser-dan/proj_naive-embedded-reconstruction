# --- Standard library ---
import os
import logging
from datetime import datetime

# --- Network science ---
import networkx as nx

# ========== FUNCTIONS ===========
# --- Graph calculations ---
def get_component_mapping(graph):
    mapping = {}  # node -> component
    components = nx.connected_components(graph)  # [[nodes in component], ..., [nodes in component]]

    # Enumerate over components, associating included nodes to that component
    for component_id, component_nodes in enumerate(components):
        for node in component_nodes:
            mapping[node] = component_id

    return mapping


# --- File management ---


# --- Logging & debugging ---
def get_today(time=False):
    if time:
        return datetime.today().strftime('%Y%m%d-%H%M%S')
    else:
        return datetime.today().strftime('%Y%m%d')

def get_module_logger(
        # Logger
        name="main",
        # File handler
        filename=f".logs/log_{datetime.today().strftime('%Y%m%d-%H%M%S')}.log",
        mode='a',
        file_level = 10,
        # Console handler
        console_level = 20):
    # Initialize logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Setup formatters
    formatter_longform = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter_shortform = logging.Formatter('%(levelname)s - %(message)s')

    # Setup stdout handler
    handler_console = logging.StreamHandler()
    handler_console.setFormatter(formatter_shortform)
    handler_console.setLevel(console_level)

    # Setup logfile handler
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    handler_logfile = logging.FileHandler(filename=filename, mode=mode)
    handler_logfile.setFormatter(formatter_longform)
    handler_logfile.setLevel(file_level)

    # Add handlers
    if console_level > 0:
        logger.addHandler(handler_console)
    if file_level > 0:
        logger.addHandler(handler_logfile)

    return logger
