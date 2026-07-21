import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
from matplotlib.patches import Patch

matplotlib.use("tKagg")

df = pd.read_csv("combined_results.csv")

# Ensure ordering is consistent everywhere
df["input_type"] = df["input_type"].replace({"coch": "cochleagram"})
df["input_type"] = df["input_type"].replace({"linear": "linear spectrogram"})
df["input_type"] = df["input_type"].replace({"mel": "mel spectrogram"})
df["input_type"] = pd.Categorical(
    df["input_type"],
    categories=["linear spectrogram", "mel spectrogram", "cochleagram"],
    ordered=True
)

accuracy = df["accuracy"]
input_type = df["input_type"]
ip = df["ip"]
tono = df["tono"]
kl_divergence = df["kl_divergence"]
df["accuracy_pct"] = df["accuracy"] * 100
accuracy_pct = df["accuracy_pct"]


# ##### PLOT FOR INPUT REPRESENTATION #####
# g = sns.catplot(
#     data=df,
#     x="input_type",
#     y="accuracy_pct",
#     hue="ip",
#     col="tono",
#     kind="box",
#     height=5,
#     aspect=1,
#     hue_order=[False, True],
#     palette=["#0c97ed", "#eb5234"]
# )
#
# # Axis labels
# g.set_axis_labels("Input representation", "Accuracy (%)", fontsize=13, fontweight="bold")
#
# # Cleaner facet titles
# g.set_titles(col_template="Tonotopic mapping: {col_name}", fontsize=13)
#
# for ax in g.axes.flat:
#     ax.title.set_fontsize(13)
#
#     # Increase x-axis category label size
#     ax.tick_params(axis='x', labelsize=11)
#
# # Improve legend
# colors = ["#0c97ed", "#eb5234"]
#
# legend_handles = [
#     Patch(facecolor=colors[0], label="No IP"),
#     Patch(facecolor=colors[1], label="With IP")
# ]
#
# # Remove seaborn's automatic legend
# if g._legend is not None:
#     g._legend.remove()
#
# # Add custom legend
# legend = g.figure.legend(
#     handles=legend_handles,
#     title="",
#     loc="upper center",
#     bbox_to_anchor=(0.5, 0.95),
#     ncol=2,
#     frameon=False,
#     fontsize=11
# )
#
# # Add overall title
# g.figure.suptitle("Model Accuracy by Input Representation", fontsize=15)
#
# # Make room for the title
# g.figure.tight_layout()
# #g.figure.subplots_adjust(top=0.95)
#
# plt.show()
#
# # =============================================
# PLOT IP THING #
# pivot = df.pivot_table(
#     index=["input_type", "tono"],
#     columns="ip",
#     values="accuracy_pct",
#     aggfunc="mean"
# )
#
# pivot["ip_gain"] = pivot[True] - pivot[False]
# pivot = pivot.reset_index()
#
# sns.barplot(
#     data=pivot,
#     x="input_type",
#     y="ip_gain",
#     hue="tono",
#     palette=["#0c97ed", "#eb5234"],
#     errorbar="sd"
# )
#
# plt.ylabel("Accuracy improvement from IP (%)", fontsize=13, fontweight="bold")
# plt.xlabel("Input representation", fontsize=13, fontweight="bold")
# plt.title("Effect of IP across input representations", fontsize=15)
#
# plt.legend(title="Tonotopy")
# plt.xticks(fontsize=11)
# plt.show()
#
# # ==========================================
# PLOT TONO THING #
pivot = df.pivot_table(
    index=["input_type", "ip"],
    columns="tono",
    values="accuracy_pct",
    aggfunc="mean"
)

pivot["tono_gain"] = pivot[True] - pivot[False]
pivot = pivot.reset_index()

sns.barplot(
    data=pivot,
    x="input_type",
    y="tono_gain",
    hue="ip",
    palette=["#0c97ed", "#eb5234"],
    errorbar="sd"
)

plt.ylabel("Accuracy improvement from Tonotopic Mapping (%)", fontsize=13, fontweight="bold")
plt.xlabel("Input representation", fontsize=13, fontweight="bold")
plt.title("Effect of Tonotopic mapping across input representations", fontsize=15)

plt.legend(title="IP")
plt.xticks(fontsize=11)

plt.show()
#
# # ==========================================
#### PLOT KL VS ACC ####
# g = sns.catplot(
#     data=df,
#     x="input_type",
#     y="accuracy",
#     hue="ip",
#     col="tono",
#     kind="point",
#     height=5
# )
#
# plt.xlabel("Input representation")
# plt.ylabel("Accuracy")
# plt.title("Average Accuracy by Input Representation")
# plt.show()
#
# result = (
#     df.groupby(["input_type", "tono", "ip"])["accuracy"]
#     .agg(["mean", "std"])
# )
#
# print(result)

# g = sns.relplot(
#     data=df[df["ip"]],
#     x="kl_divergence",
#     y="accuracy_pct",
#     col="input_type",
#     kind="scatter",
#     height=4,
#     aspect=1,
#     alpha=0.7
# )
# g.set_axis_labels("KL Divergence", "Accuracy (%)")
#
# # Panel titles
# g.set_titles("Input Representation: {col_name}")
#
# # Overall figure title
# g.fig.suptitle("Accuracy vs KL Divergence", y=1.05)
#
# plt.show()

### Get mean and std of things ###
filtered = df[(df["ip"] == True) & (df["tono"] == True)]

result = (
    df
    .groupby(["input_type", "ip"])
    .agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),  # I assume you meant std, not mean twice
        count=("accuracy", "count")  # optional, useful for sanity check
    )
    .reset_index()
)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
# pretty_result = (
#     result
#     .sort_values(["input_type", "N", "sigma", "lr"])
#     .set_index(["input_type", "N", "sigma", "lr"])
# )

print(result.to_string())
