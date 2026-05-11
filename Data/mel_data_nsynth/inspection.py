import numpy as np

data = np.load("npz.npz", allow_pickle=True)

specs = data["specs"]
targets = data["targets"]

silence_idx = 11
num_classes = targets.shape[-1]

sample_counts = np.zeros(num_classes)

for y in targets:
    labels = np.argmax(y, axis=1)

    # Remove silence timesteps
    labels = labels[labels != silence_idx]

    if len(labels) == 0:
        sample_label = silence_idx
    else:
        # Since each file has 1 instrument, this is safe:
        sample_label = labels[0]

    sample_counts[sample_label] += 1

print("Specs per class:", sample_counts)