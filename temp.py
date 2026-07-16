import pandas as pd

# Load both files
df1 = pd.read_csv("D:/Data/THESIS RESULTS/Deep NSynth params/combined_results_nsynth_ip_params.csv")
df2 = pd.read_csv("D:/Data/THESIS RESULTS/Deep NSynth params/extra_layers.csv")

# Combine them
combined = pd.concat([df1, df2], ignore_index=True)

# Optional: remove duplicate rows
combined = combined.drop_duplicates()

# Save
combined.to_csv("D:/Data/THESIS RESULTS/Deep NSynth params/all_layers_results.csv", index=False)