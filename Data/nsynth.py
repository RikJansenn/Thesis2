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

output_folder = r"D:/Data/nsynth_preprocessed_pitch2"
target_sr = 16000
samples_per_class = 1818
min_rms_db = -42
seed = 42


# -------------------------
# FUNCTIONS
# -------------------------

def rms(audio, eps=1e-12):
    return np.sqrt(np.mean(audio ** 2) + eps)


def rms_db(audio):
    return 20 * np.log10(rms(audio))


def rms_normalize(audio, target_db=-20):
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10 ** (target_db / 20)

    gain = target_rms / (rms + 1e-8)
    return audio * gain

def allocate_pitch_counts(common_pitch_capacity, total_samples):
    """
    Allocate total_samples across pitches based on shared pitch availability.
    Every class will use the same number of samples per pitch.
    """
    total_capacity = sum(common_pitch_capacity.values())

    if total_capacity < total_samples:
        raise ValueError(
            f"Not enough shared pitch-balanced samples. "
            f"Can only make {total_capacity}, requested {total_samples}."
        )

    pitches = sorted(common_pitch_capacity.keys())

    raw = {
        pitch: total_samples * common_pitch_capacity[pitch] / total_capacity
        for pitch in pitches
    }

    counts = {pitch: int(np.floor(raw[pitch])) for pitch in pitches}

    remaining = total_samples - sum(counts.values())

    # Give remaining samples to pitches with largest fractional parts
    fractional = sorted(
        pitches,
        key=lambda p: raw[p] - counts[p],
        reverse=True
    )

    for pitch in fractional:
        if remaining == 0:
            break

        if counts[pitch] < common_pitch_capacity[pitch]:
            counts[pitch] += 1
            remaining -= 1

    return counts

def preprocess_balanced_dataset(
    json_path,
    audio_folder,
    output_folder,
    target_sr=16000,
    samples_per_class=1818,
    min_rms_db=-42,
    seed=42
):
    os.makedirs(output_folder, exist_ok=True)

    with open(json_path, "r") as f:
        metadata = json.load(f)

    rng = random.Random(seed)

    # First collect valid files per class and pitch
    family_pitch_to_note_ids = defaultdict(lambda: defaultdict(list))

    print("Checking valid files...")
    i = 0
    for note_id, info in metadata.items():
        family_id = info["instrument_family"]
        pitch = info["pitch"]

        if pitch < 24 or pitch > 89:
            continue

        input_path = os.path.join(audio_folder, f"{note_id}.wav")

        print(f"{i}/{len(metadata.items())}")

        if not os.path.exists(input_path):
            continue

        try:
            audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
            print(orig_sr)
        except Exception:
            continue

        level_db = rms_db(audio)

        if level_db < min_rms_db:
            continue

        family_pitch_to_note_ids[family_id][pitch].append(note_id)

        i += 1

    family_ids = sorted(family_pitch_to_note_ids.keys())

    # Shuffle candidates
    for family_id in family_ids:
        for pitch in family_pitch_to_note_ids[family_id]:
            print(pitch)
            rng.shuffle(family_pitch_to_note_ids[family_id][pitch])

    # Find pitches that exist in every class
    common_pitches = set(family_pitch_to_note_ids[family_ids[0]].keys())

    for family_id in family_ids[1:]:
        print(family_id)
        common_pitches &= set(family_pitch_to_note_ids[family_id].keys())

    common_pitches = sorted(common_pitches)

    print(f"Common pitches across all classes: {len(common_pitches)}")

    # For each pitch, max shared count is limited by the weakest class
    common_pitch_capacity = {}

    for pitch in common_pitches:
        min_available = min(
            len(family_pitch_to_note_ids[family_id][pitch])
            for family_id in family_ids
        )

        if min_available > 0:
            common_pitch_capacity[pitch] = min_available

    target_pitch_counts = allocate_pitch_counts(
        common_pitch_capacity,
        samples_per_class
    )

    print("\nTarget pitch distribution:")
    for pitch, count in target_pitch_counts.items():
        if count > 0:
            print(f"Pitch {pitch}: {count}")

    selected_metadata = {}
    selected_counts = defaultdict(int)

    for family_id in family_ids:
        print(f"\nProcessing class {family_id}")

        selected_note_ids = []

        for pitch, count in target_pitch_counts.items():
            candidates = family_pitch_to_note_ids[family_id][pitch]

            selected_note_ids.extend(candidates[:count])

        rng.shuffle(selected_note_ids)

        for note_id in selected_note_ids:
            input_path = os.path.join(audio_folder, f"{note_id}.wav")
            output_path = os.path.join(output_folder, f"{note_id}.wav")

            try:
                audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
            except Exception as e:
                print(f"Failed loading {note_id}: {e}")
                continue

            if orig_sr != target_sr:
                audio = librosa.resample(
                    audio,
                    orig_sr=orig_sr,
                    target_sr=target_sr
                )

            audio = rms_normalize(audio)

            sf.write(output_path, audio, target_sr)

            selected_metadata[note_id] = metadata[note_id]
            selected_counts[family_id] += 1

            if selected_counts[family_id] % 100 == 0:
                print(
                    f"Class {family_id}: "
                    f"{selected_counts[family_id]}/{samples_per_class}"
                )

        print(f"Class {family_id} complete: {selected_counts[family_id]} files")

    metadata_output_path = os.path.join(output_folder, "selected_examples.json")

    with open(metadata_output_path, "w") as f:
        json.dump(selected_metadata, f, indent=2)

    total_selected = sum(selected_counts.values())

    print("\nDone!")
    print(f"Total selected: {total_selected}")
    print(f"Saved audio to: {output_folder}")
    print(f"Saved metadata to: {metadata_output_path}")


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    preprocess_balanced_dataset(
        json_path=json_path,
        audio_folder=audio_folder,
        output_folder=output_folder,
        target_sr=target_sr,
        samples_per_class=samples_per_class,
        min_rms_db=min_rms_db,
        seed=seed
    )