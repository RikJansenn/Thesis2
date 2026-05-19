import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import librosa
from reservoirpy.observables import spectral_radius, effective_spectral_radius
from sklearn.model_selection import KFold, train_test_split
matplotlib.use('tKagg')

# Load the CSV files
df = pd.read_csv("results_deep_params_0.94_0.4_0.1_1000_IP=False_secondversion.csv")

# Compute mean and std
df_stats = df.groupby('layer')['accuracy'].agg(['mean', 'std']).reset_index()

plt.figure()
plt.errorbar(
    df_stats['layer'],
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

stats = df.groupby("layer")["accuracy"].agg(["mean", "std"])
print(stats)