import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

# Read CSVs.
# header=0 means: use the first row 0,1,2,...,250 as column names
pred_timesteps = pd.read_csv("fulldatatest/timesteps_csv_2_energied.csv", header=0).to_numpy()
label_timesteps = pd.read_csv("fulldatatest/labels_csv_2_energied.csv", header=0).to_numpy()


# Majority vote per sample
y_pred = mode(pred_timesteps, axis=1, keepdims=False).mode.astype(int)

# First timestep label
y_true = label_timesteps[:, 0].astype(int)

# Metrics
print(classification_report(y_true, y_pred))

# Confusion matrix
labels = sorted(set(y_true) | set(y_pred))
cm = confusion_matrix(y_true, y_pred, labels=labels)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot(values_format="d")
# plt.title("Confusion matrix: majority-vote predictions")
# plt.show()

cm_norm = confusion_matrix(
    y_true,
    y_pred,
    normalize="true"
)

disp = ConfusionMatrixDisplay(cm_norm)
disp.plot(values_format=".2f")
plt.show()

correct = (pred_timesteps == y_true[:, None])
# timestep_acc = correct.mean(axis=0)

# plt.plot(timestep_acc)
# plt.xlabel("Timestep")
# plt.ylabel("Accuracy")
# plt.show()

classes = np.sort(np.unique(y_true))

plt.figure(figsize=(8, 5))

for cls in classes:
    mask = (y_true == cls)

    # Accuracy at each timestep for this class
    class_acc = correct[mask].mean(axis=0)

    plt.plot(class_acc, label=f"Class {cls}")

plt.xlabel("Timestep")
plt.ylabel("Accuracy")
plt.title("Accuracy per timestep by class")
plt.legend()
plt.grid(True)
plt.show()

majority_classes = []
confidences = []

for pred in pred_timesteps:
    counts = np.bincount(pred)
    majority = counts.argmax()
    conf = counts.max() / len(pred)

    majority_classes.append(majority)
    confidences.append(conf)

correct = majority_classes == label_timesteps[:, 0]

majority_classes = np.array(majority_classes)
confidences = np.array(confidences)

print(confidences[correct].mean())
print(confidences[~correct].mean())
