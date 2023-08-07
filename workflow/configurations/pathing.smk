from os.path import join as jn

# ========== DIRECTORY ALIASES ==========
# Roots
# Assume pathing from project root!
DATA = jn("data", "")
SCRIPTS = jn("workflow", "scripts", "")

# Major sections
DIR_INPUT = jn(DATA, "input", "")
DIR_INTERIM = jn(DATA, "interim", "")
DIR_OUTPUT = jn(DATA, "output", "")

# Minor sections/targets
DIR_EDGELISTS = jn(DIR_INPUT, "edgelists", "")
DIR_PARTITIONS = jn(DIR_INPUT, "partitions", "")
DIR_REMNANTS = jn(DIR_INPUT, "remnants", "")

DIR_EMBEDDINGS = jn(DIR_INTERIM, "embeddings", "")
DIR_MODELS = jn(DIR_INTERIM, "models", "")

DIR_DATAFRAMES = jn(DIR_OUTPUT, "dataframes", "")
DIR_FIGURES = jn(DIR_OUTPUT, "figures", "")
DIR_REPORTS = jn(DIR_OUTPUT, "reports", "")