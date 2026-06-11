import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.use('tkagg')

# All initial sr/lr combinations
# df1 = pd.read_csv("srlrtest2.csv")
# df2 = pd.read_csv("srlr_fixed.csv")
# df3 = pd.read_csv("srlrtest3.csv")
# df = df1._append(df2)
# df = df._append(df3)
#
# # Current lr with different sr:
# df4 = pd.read_csv("srlrtest4.csv")
# df = df._append(df4)

df = pd.read_csv("unscaledsrlrtest.csv")

# Mean accuracy for each SR/LR combination
means = (
    df.groupby(["sr", "lr"])["accuracy"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(6, 4))

for sr in sorted(means["sr"].unique()):
    subset = means[means["sr"] == sr]
    plt.plot(
        subset["lr"],
        subset["accuracy"],
        marker="o",
        label=f"SR={sr}"
    )

plt.xlabel("LR")
plt.ylabel("Mean Accuracy")
plt.title("SR × LR Interaction")
plt.legend()
plt.grid(True, alpha=0.3)

plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.tight_layout()
plt.show()
