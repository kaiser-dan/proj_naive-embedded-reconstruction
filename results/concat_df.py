import os
import pandas as pd

dfs = [pd.read_parquet(f"dataframes/{f}") for f in os.listdir("dataframes/") if "EMB_ex04R" in os.path.basename(f)]

df = pd.concat(dfs)

df.to_parquet("dataframes/dataframe-merged_EMB_ex04R_DK_20220928.parquet")
