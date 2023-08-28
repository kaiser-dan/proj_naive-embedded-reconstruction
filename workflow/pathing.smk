from os.path import join as jn

# Assumes running workflow from project root
DIR_LOGS = jn(".logs", "")

DIR_DATA = jn("data", "")

DIR_EDGELISTS = jn(DIR_DATA, "edgelists", "")
DIR_REMNANTS = jn(DIR_DATA, "remnants", "")

DIR_EMBEDDINGS = jn(DIR_DATA, "embeddings", "")
DIR_MODELS = jn(DIR_DATA, "models", "")

DIR_DATAFRAMES = jn(DIR_DATA, "dataframes", "")
DIR_FIGURES = jn(DIR_DATA, "figures", "")