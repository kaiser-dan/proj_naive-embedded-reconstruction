# ============= SET-UP =================
import pandas as pd


# ============= MAIN ==============
if __name__ == "__main__":
    records = []
    for record_fh in snakemake.input:
        with open(record_fh, "rb") as _fh:
            record = pickle.load(_fh)
        records.append(record)

    df = pd.DataFrame.from_records(records)

    df.to_parquet(snakemake.output[0])
