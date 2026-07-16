import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

df1 = pd.read_csv("D:/Data/THESIS RESULTS/Deep MNIST params/almost_full.csv")
df2 = pd.read_csv("D:/Data/THESIS RESULTS/Deep MNIST params/bonus.csv")

df = pd.concat([df1, df2])

df.to_csv("D:/Data/THESIS RESULTS/Deep MNIST params/full_results.csv")

summary = (
    df.groupby(["layers", "sr", "lr", "sigma"])["accuracy"]
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
        group = group.sort_values("layers")

        plt.errorbar(
            group["layers"],
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

