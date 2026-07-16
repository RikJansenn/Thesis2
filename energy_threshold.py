import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from tqdm import tqdm

dataset_path = r"D:/Data/NSynthTrain/nsynth-train.jsonwav.tar/nsynth-train"
json_path = os.path.join(dataset_path, "examples.json")
audio_folder = os.path.join(dataset_path, "audio")

# Load metadata
with open(json_path, "r") as f:
    metadata = json.load(f)

rows = []

for note_id, info in tqdm(metadata.items()):
    wav_path = os.path.join(audio_folder, note_id + ".wav")

    try:
        y, sr = librosa.load(wav_path, sr=None)

        rms = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms + 1e-12)

        rows.append({
            "note_id": note_id,
            "pitch": info["pitch"],
            "velocity": info["velocity"],
            "instrument_family": info["instrument_family_str"],
            "instrument": info["instrument_str"],
            "source": info["instrument_source_str"],
            "rms_db": rms_db
        })

    except Exception as e:
        print(f"Failed: {note_id} ({e})")

df = pd.DataFrame(rows)

print(df.head())
print(df["rms_db"].describe())

plt.figure(figsize=(14, 8))

for family in df["instrument_family"].unique():
    subset = df[df["instrument_family"] == family]

    plt.scatter(
        subset["pitch"],
        subset["rms_db"],
        s=5,
        alpha=0.25,
        label=family
    )

plt.xlabel("MIDI Pitch")
plt.ylabel("RMS (dB)")
plt.title("NSynth RMS vs Pitch by Instrument Family")
plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()