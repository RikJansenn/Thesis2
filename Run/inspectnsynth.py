import numpy as np
from pathlib import Path
from collections import Counter

instruments = [
    "bass",
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "organ",
    "reed",
    "string",
    "synth",
    "vocal",
]

folder = Path("D:/Data/nsynth_preprocessed")

counts = Counter()

for file in folder.glob("*.wav"):
    instrument = file.stem.split("_")[0]
    counts[instrument] += 1

print("Instrument counts:")
for instrument in instruments:
    print(f"{instrument:10s}: {counts[instrument]}")

data = np.load("../Data/mel_data_nsynth/mel_train.npz")

y = data["targets"]  # adjust key name if needed

# Take label from first timestep
sample_labels = np.argmax(y[:, 0, :], axis=1)

# Count occurrences per class
class_counts = np.bincount(sample_labels)

print("Class counts:")
for cls, count in enumerate(class_counts):
    print(f"Class {cls}: {count}")