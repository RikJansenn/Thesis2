import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
#
# # Read data
df = pd.read_csv("D:\Data\THESIS RESULTS\Deep NSynth params [COMPLETE]/full_results_new.csv")
filtered_df = df[df["layer"] == 11]

df2 = pd.read_csv("D:\Data\THESIS RESULTS\Deep NSynth params [COMPLETE]/full_results_old.csv")

df_full = pd.concat([filtered_df, df2])
df_full.to_csv("D:\Data\THESIS RESULTS\Deep NSynth params [COMPLETE]/full_results.csv", index=False)

# Compute mean and std accuracy for each layer
summary = (
    df.groupby(["layer", "sr", "lr", "sigma"])["accuracy"]
      .agg(["mean", "std"])
      .reset_index()
      .sort_values("mean", ascending=False)
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print(summary)

# One figure per parameter set
for (sr, lr), sr_lr_group in summary.groupby(["sr", "lr"]):
    plt.figure(figsize=(7, 5))

    for sigma, group in sr_lr_group.groupby("sigma"):
        group = group.sort_values("layer")

        plt.errorbar(
            group["layer"],
            group["mean"],
            yerr=group["std"],
            marker="o",
            capsize=3,
            label=f"σ={sigma}"
        )

    plt.title(f"sr={sr}, lr={lr}")
    plt.xlabel("Number of layers")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Sigma")
    plt.tight_layout()
    plt.show()
# #
# # from pathlib import Path
# # import re
# # import pandas as pd
# #
# # # Folder containing your csv files
# # input_dir = Path(r"D:\Data\deep\params_nsynth_ip")
# #
# # # Output file
# # output_file = input_dir / "combined_results_nsynth_ip_params.csv"
# #
# # pattern = re.compile(
# #     r"params_deep_nsynth_([0-9.]+)_([0-9.]+)_([0-9.]+)_10\.csv$"
# # )
# #
# # all_dfs = []
# #
# # for file in input_dir.glob("params_deep_nsynth_*_*_*_10.csv"):
# #     match = pattern.match(file.name)
# #
# #     if not match:
# #         print(f"Skipping unexpected filename: {file.name}")
# #         continue
# #
# #     sr, lr, sigma = map(float, match.groups())
# #
# #     df = pd.read_csv(file)
# #
# #     # Keep only needed columns from the file
# #     df = df[["layer", "accuracy"]].copy()
# #
# #     # Add parameters parsed from filename
# #     df["sr"] = sr
# #     df["lr"] = lr
# #     df["sigma"] = sigma
# #
# #     # Reorder columns
# #     df = df[["layer", "sr", "lr", "sigma", "accuracy"]]
# #
# #     all_dfs.append(df)
# #
# # combined = pd.concat(all_dfs, ignore_index=True)
# #
# # combined.to_csv(output_file, index=False)
# #
# # print(f"Saved {len(combined)} rows to:")
# # print(output_file)

# import pandas as pd
#
# path = r"D:\Data\THESIS RESULTS\Deep NSynth results\deep_params_results.csv"
#
# df = pd.read_csv(path)
#
# # Put the current index back as first column
# df = df.reset_index()
#
# # Drop the broken/empty last column
# df = df.drop(columns=["accuracy"])
#
# # Rename columns correctly
# df.columns = ["N", "sr", "lr", "sigma", "IP", "esr", "kl", "accuracy"]
#
# df = df.drop_duplicates()
#
# df.to_csv(
#     r"D:\Data\THESIS RESULTS\Deep NSynth results\deep_params_results_noduplicates.csv",
#     index=False
# )
#
# print(df.head())

# import pandas as pd
#
# # Read the two files
# df = pd.read_csv("D:\Data\THESIS RESULTS\Shallow NSynth params/full_results.csv")
#
# # df = df[
# #     df["sr"].isin([0.2, 0.4]) &
# #     df["lr"].isin([0.2, 0.4])
# # ]
#
# df["N"] = 1200
#
# summary = (
#     df.groupby(["N", "sr", "lr", "sigma"])["accuracy"]
#       .agg(["mean", "std"])
#       .reset_index()
#       .sort_values("mean", ascending=False)
# )
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# print(summary)

