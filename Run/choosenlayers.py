import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.use('tkagg')

df = pd.read_csv("diffsrlrparams.csv")

summary = (
    df.groupby(["sr", "lr", "layer"])
      .agg(
          mean_accuracy=("accuracy", "mean"),
          std_accuracy=("accuracy", "std")
      )
      .reset_index()
)

plt.figure(figsize=(10, 6))

for (sr, lr), group in summary.groupby(["sr", "lr"]):
    group = group.sort_values("layer")

    plt.errorbar(
        group["layer"],
        group["mean_accuracy"],
        yerr=group["std_accuracy"],
        marker="o",
        capsize=4,
        label=f"sr={sr}, lr={lr}"
    )

plt.xlabel("Number of layers")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()