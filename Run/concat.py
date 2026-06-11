from pathlib import Path
import pandas as pd

root = Path(r"D:\Data\result_deep_mnist")

dfs = []

for csv_path in root.rglob("*.csv"):
    if csv_path.name == "combined_results.csv":
        continue

    folder = csv_path.parent.name.lower()
    df = pd.read_csv(csv_path)

    df["IP"] = "ip" in folder
    df["tono"] = "tono" in folder

    if "linear" in folder:
        df["input_type"] = "linear"
    elif "mel" in folder:
        df["input_type"] = "mel"
    elif "coch" in folder:
        df["input_type"] = "coch"
    else:
        raise ValueError(f"Could not infer input_type from folder: {folder}")

    # sigma is only meaningful for IP runs
    if not df["IP"].iloc[0]:
        df["sigma"] = pd.NA

    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

summary = (
    combined
    .groupby(["IP", "tono", "input_type", "N", "sr", "lr", "sigma"], dropna=False)
    ["accuracy"]
    .mean()
    .reset_index(name="mean_accuracy")
)

summary.to_csv(root / "average_accuracy_by_condition_and_params.csv", index=False)

print(summary)