import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

# Load CSV files
preds = pd.read_csv("timesteps_csv.csv", header=None)
labels = pd.read_csv("labels_csv.csv", header=None)

majority_preds = []
true_labels = []

for pred_row, label_row in zip(preds.values, labels.values):

    # Keep only positions where neither prediction nor label is 10
    mask = (pred_row != 10) & (label_row != 10)

    filtered_preds = pred_row[mask]
    filtered_labels = label_row[mask]

    # Skip rows with no remaining entries
    if len(filtered_preds) == 0:
        continue

    # Majority vote on remaining predictions
    majority_pred = mode(filtered_preds, keepdims=False).mode

    # All remaining labels should still be identical, so take the first
    true_label = filtered_labels[0]

    majority_preds.append(majority_pred)
    true_labels.append(true_label)

# Compute confusion matrix
cm = confusion_matrix(true_labels, majority_preds)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (ignoring class 10)")
plt.show()