import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

df1 = pd.read_csv("tonowidthnsynth_INCREASE2.csv")
df2 = pd.read_csv("tonowidthnsynth_INCREASE3.csv")
df3 = pd.read_csv("tonowidthnsynth_INCREASE4.csv")

df = pd.concat([df1, df2, df3])

# df = pd.read_csv("tonowidthnsynth.csv")


plt.figure(figsize=(6, 4))

plt.scatter(
    df["width_increase"],
    df["accuracy"],
)

plt.show()
