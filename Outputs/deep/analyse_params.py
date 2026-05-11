import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

# Load your combined file
df = pd.read_csv("deep_sweep.csv")

# Optional: make sure sorting looks nice
df["lr"] = df["lr"].astype(float)
df["sr"] = df["sr"].astype(float)
df["sigma"] = df["sigma"].astype(float)
df["N"] = df["N"].astype(int)

sns.set(style="whitegrid")

# ---- LR vs Accuracy ----
plt.figure()
sns.violinplot(x="lr", y="accuracy", data=df)
plt.title("Accuracy distribution per learning rate (lr)")
plt.show()

# ---- N vs Accuracy ----
plt.figure()
sns.violinplot(x="N", y="accuracy", data=df)
plt.title("Accuracy distribution per N")
plt.show()

# ---- Sigma vs Accuracy ----
plt.figure()
sns.violinplot(x="sigma", y="accuracy", data=df)
plt.title("Accuracy distribution per sigma")
plt.show()