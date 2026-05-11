import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import librosa
from reservoirpy.observables import spectral_radius, effective_spectral_radius
from sklearn.model_selection import KFold, train_test_split
matplotlib.use('tKagg')

# Load the CSV files
df = pd.read_csv("results_deep_params_0.97_0.8_0.2_1200_IP=True.csv")

# Compute mean and std
df_stats = df.groupby('N_layers')['accuracy'].agg(['mean', 'std']).reset_index()

plt.figure()
plt.errorbar(
    df_stats['N_layers'],
    df_stats['mean'],
    yerr=df_stats['std'],
    marker='o',
    capsize=5
)

plt.xlabel('N_layers')
plt.ylabel('Average accuracy')
plt.show()


import pandas as pd
import plotly.express as px

# Load CSV

stats = df.groupby("N_layers")["accuracy"].agg(["mean", "std"])
print(stats)