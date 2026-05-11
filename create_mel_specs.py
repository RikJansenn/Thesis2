import numpy as np
import librosa
import os

folder_path = "D:/Data/nsynth_preprocessed"

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
    "synth_lead": 9,
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

                S_mel = create_mel_spectrogram(audio, sr)
                mel_samples.append(S_mel)

                label_mel = create_label(S_mel, filename)
                targets_mel.append(label_mel)

    return mel_samples, targets_mel


def get_instrument_from_filename(filename):
    name = os.path.splitext(filename)[0]

    for instrument in INSTRUMENT_TO_LABEL:
        if name.startswith(instrument + "_"):
            return instrument

    raise ValueError(f"Could not infer instrument from filename: {filename}")


NUM_CLASSES = 11  # 11 instruments + 1 silence


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


def create_mel_spectrogram(audio, sr):
    fixed_length = 1

    n_fft = 512
    win_length = 512
    hop_length = 256
    n_mels = 80

    # audio = trim_or_pad(audio, sr, fixed_length)

    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )

    S = librosa.power_to_db(S, ref=np.max)
    S = S.T

    S = (S - S.min()) / (S.max() - S.min())

    print(f"Mel shape: {S.shape}")
    return S


def trim_or_pad(audio, sr, fixed_length):
    target_len = int(sr * fixed_length)

    if len(audio) < target_len:
        pad_len = target_len - len(audio)

        left = np.random.randint(0, pad_len + 1)
        right = pad_len - left

        audio = np.pad(audio, (left, right), mode="constant")
    else:
        audio = audio[:target_len]

    return audio


if __name__ == "__main__":
    melspecs, targets_mel = load_training_data(folder_path)

    os.makedirs("Data/mel_data_nsynth", exist_ok=True)

    np.savez(
        "Data/mel_data_nsynth/specs",
        specs=np.array(melspecs),
        targets=np.array(targets_mel),
    )