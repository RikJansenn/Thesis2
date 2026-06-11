import pandas as pd
import glob
import os

df = pd.read_csv("sweep_results_nsynth.csv")

summary = (
    df.groupby(["N", "sr", "lr", "sigma"])["accuracy"]
      .agg(["mean", "std", "count"])
      .reset_index()
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print(summary)