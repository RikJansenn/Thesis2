import os
import json
import random
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf


# -------------------------
# CONFIG
# -------------------------

dataset_path = r"D:/Data/NSynthTrain/nsynth-train.jsonwav.tar/nsynth-train"
json_path = os.path.join(dataset_path, "examples.json")
audio_folder = os.path.join(dataset_path, "audio")

output_folder = r"nsynth_preprocessed"
target_sr = 16000
samples_per_class = 1818
seed = 42


# -------------------------
# FUNCTIONS
# -------------------------

def rms_normalize(audio, eps=1e-8):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms < eps:
        return audio

    return audio / rms


def select_balanced_subset(json_path, samples_per_class=1818, seed=42):
    with open(json_path, "r") as f:
        metadata = json.load(f)

    family_to_files = defaultdict(list)

    for note_id, info in metadata.items():
        family_id = info["instrument_family"]
        filename = f"{note_id}.wav"
        family_to_files[family_id].append(filename)

    selected_files = []

    random.seed(seed)

    for family_id in sorted(family_to_files.keys()):
        files = family_to_files[family_id]

        if len(files) < samples_per_class:
            raise ValueError(
                f"Class {family_id} only has {len(files)} files, "
                f"but you requested {samples_per_class}."
            )

        chosen = random.sample(files, samples_per_class)
        selected_files.extend(chosen)

        print(f"Class {family_id}: selected {len(chosen)} / {len(files)}")

    random.shuffle(selected_files)

    print(f"\nTotal selected: {len(selected_files)}")
    return selected_files, metadata


def preprocess_selected_files(
    selected_files,
    metadata,
    audio_folder,
    output_folder,
    target_sr=8000,
):
    os.makedirs(output_folder, exist_ok=True)

    selected_metadata = {}

    for i, filename in enumerate(selected_files, start=1):
        print(i)
        input_path = os.path.join(audio_folder, filename)

        if not os.path.exists(input_path):
            print(f"Missing file, skipping: {input_path}")
            continue

        note_id = os.path.splitext(filename)[0]

        audio, orig_sr = librosa.load(input_path, sr=None, mono=True)

        if orig_sr != target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr
            )

        audio = rms_normalize(audio)

        output_path = os.path.join(output_folder, filename)

        sf.write(output_path, audio, target_sr)

        selected_metadata[note_id] = metadata[note_id]

        if i % 100 == 0 or i == len(selected_files):
            print(f"Processed {i}/{len(selected_files)} files")

    metadata_output_path = os.path.join(output_folder, "selected_examples.json")

    with open(metadata_output_path, "w") as f:
        json.dump(selected_metadata, f, indent=2)

    print("\nDone!")
    print(f"Saved audio to: {output_folder}")
    print(f"Saved metadata to: {metadata_output_path}")


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    selected_files, metadata = select_balanced_subset(
        json_path=json_path,
        samples_per_class=samples_per_class,
        seed=seed
    )

    preprocess_selected_files(
        selected_files=selected_files,
        metadata=metadata,
        audio_folder=audio_folder,
        output_folder=output_folder,
        target_sr=target_sr
    )