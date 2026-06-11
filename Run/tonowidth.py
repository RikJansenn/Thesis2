import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

df = pd.read_csv("tonotest2.csv")

plt.figure(figsize=(6, 4))

plt.scatter(
    df["input_width"],
    df["accuracy"],
)

plt.show()
