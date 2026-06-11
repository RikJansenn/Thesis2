import pandas as pd
from pathlib import Path

folder = Path(r"D:\Data\deep")

df = pd.concat(
    [pd.read_csv(f) for f in folder.glob("*.csv")],
    ignore_index=True
)

df.to_csv(folder / "deep_params_results.csv", index=False)