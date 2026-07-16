import numpy as np
import librosa
import os

folder_path = "D:/Data/nsynth_preprocessed_pitch2"

INSTRUMENT_TO_LABEL = {
    "bass": 0,
    "brass": 1,
    "flute": 2,
    "guitar": 3,
    "keyboard": 4,
    "mallet": 5,
    "organ": 6,
    "reed": 7,
    "string": 8,
    "synth": 9,
    "vocal": 10,
}

NUM_CLASSES = 11


def load_training_data(folder_path):
    mel_samples = []
    targets_mel = []

    for root_dir, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root_dir, filename)
                print(file_path)

                audio, sr = librosa.load(file_path, sr=None)

                S_mel = create_linear_spectrogram(audio, sr)
                mel_samples.append(S_mel)

                label_mel = create_label(S_mel, filename)
                targets_mel.append(label_mel)

    return mel_samples, targets_mel


def get_instrument_from_filename(filename):
    name = os.path.splitext(filename)[0]
    instrument = name.split("_")[0]

    if instrument not in INSTRUMENT_TO_LABEL:
        raise ValueError(f"Unknown instrument '{instrument}' from filename: {filename}")

    return instrument


def create_label(S, filename):
    # Get instrument label
    instrument = get_instrument_from_filename(filename)
    class_index = INSTRUMENT_TO_LABEL[instrument]

    instrument_label = np.eye(NUM_CLASSES)[class_index]

    time_steps = S.shape[0]
    labels_expanded = np.zeros((time_steps, NUM_CLASSES))

    for t in range(time_steps):
        labels_expanded[t] = instrument_label

    return labels_expanded


def create_linear_spectrogram(audio, sr):
    # Spectrogram parameters
    n_fft = 158
    win_length = 158
    hop_length = 256

    # Create Spectrogram, conver to db and transpose to match expected input shape (time_steps, features)
    S = np.abs(librosa.stft(y=audio, win_length=win_length, n_fft=n_fft, hop_length=hop_length))
    S = librosa.amplitude_to_db(S, ref=np.max)
    S = S.T

    # Normalize Spectrogram
    S = (S - S.min()) / (S.max() - S.min())

    print(f"Linear shape: {S.shape}")

    return S


if __name__ == "__main__":
    specs, targets = load_training_data(folder_path)

    os.makedirs("Data/linear_data_nsynth", exist_ok=True)

    np.savez(
        "Data/linear_data_nsynth/specs",
        specs=np.array(specs),
        targets=np.array(targets),
    )