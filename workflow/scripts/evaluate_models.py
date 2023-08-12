"""CLI to evaluate all models currently available and form a dataframe of results.
"""
# ================= SET-UP =======================
# --- Standard library ---
import os
import sys
import pickle

# --- Data handling ---
import pandas as pd

# --- Source code ---
from embmplxrec import classifiers
import embmplxrec.utils

# --- Globals ---
## Parameters

## Filepaths & templates
# * Relative to project root!
DIR_MODELS = os.path.join("data", "interim", "debug_models", "")
DIR_DATAFRAMES = os.path.join("data", "outut", "dataframes", "")

## Logging
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    filename=f".logs/evaluate_models_{embmplxrec.utils.get_today(time=True)}.log",
    console_level=10)

# ========== FUNCTIONS ==========
def evaluate_model(model_filepath):
    # Instantiate record and add parameters to record
    record = dict()
    identifiers = model_filepath.split('_')
    identifiers = [
        identifier
        for identifier in identifiers
        if '-' in identifier
    ]
    for identifier in identifiers:
        attr, val = identifier.split('-')
        record[attr] = val

    # Load model and training data
    with open(model_filepath, 'rb') as _fh:
        model, X_test, Y_test = pickle.load(_fh)

    logger.debug(model)
    logger.debug(model.X)
    logger.debug(model.Y)
    logger.debug(X_test)
    logger.debug(Y_test)

    # Evaluate model on test data, add to record
    record["auroc"] = model.auroc(X_test, Y_test, class_ = 0)
    record["accuracy"] = model.accuracy(X_test, Y_test)
    record["pr"] = model.pr(X_test, Y_test, class_ = 0)

    return record

# ========== MAIN ==========
def main():
    records = []
    for model_filepath in sys.argv[1:-1]:
        try:
            record = evaluate_model(model_filepath)

        except Exception as err:
            print(err)
            print(model_filepath)

            continue
        else:
            # Add record to rolling list
            records.append(record)

    df = pd.DataFrame.from_records(records)
    df.to_csv(sys.argv[-1], index=False)

if __name__ == "__main__":
    main()
