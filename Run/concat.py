import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import librosa
from reservoirpy.observables import spectral_radius, effective_spectral_radius
from sklearn.model_selection import KFold, train_test_split
matplotlib.use('tKagg')

# Load the CSV files
df = pd.read_csv("results_deep_params_0.94_0.8_0.2_1000_IP=True.csv")

# Group by N_layers and compute mean accuracy
df_avg = df.groupby('N_layers')['accuracy'].mean().reset_index()

plt.figure()
plt.plot(df_avg['N_layers'], df_avg['accuracy'], marker='o')
plt.xlabel('N_layers')
plt.ylabel('Average accuracy')
plt.show()


import pandas as pd
import plotly.express as px

# Load CSV

stats = df.groupby("N_layers")["accuracy"].agg(["mean", "std"])
print(stats)