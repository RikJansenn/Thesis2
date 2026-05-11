# So this is just for visualization

# First choose an audio file, create a spectrogram and plot it. Keep track of digit and gender
# We want a zero, a male seven and a female seven, for example

# Then apply IP on the network, and run the specs through the network
# Plot the PDF and stuff

# Should the specs be the preprocessed specs? Or raw?
import reservoirpy as rpy
from reservoirpy.nodes import IPReservoir
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import entropy
import librosa
import soundfile as sf
matplotlib.use('tkAgg')
from models import ShallowNetwork
from utils import *
import pycochleagram.cochleagram as cgram
import pycochleagram.utils as cu

from biological_constraints import apply_ip

# path1 = "C:/Users/rikki/Uni/preprocessed_data/52/7_52_1.wav"
path1 = "C:/Users/rikki/Uni/preprocessed_data/02/0_02_0.wav"
# path1 = "C:/Users/rikki/Uni/preprocessed_data/01/7_01_0.wav"

def dothings(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    sample_factor = 1
    nonlinearity = "db"

    # Length to pad/trim to
    fixed_length = 1
    n = 40

    # Trim/pad to fixed length
    audio = trim_or_pad(audio, sr, fixed_length)

    S = cgram.human_cochleagram(audio,
                                sr,
                                n=n,
                                low_lim=50,
                                high_lim=4000,
                                downsample=downsampler,
                                sample_factor=sample_factor,
                                nonlinearity=nonlinearity,
                                strict=False).T

    plt.subplot(222)
    plt.title('Cochleagram with poly downsampling')
    plt.ylabel('filter #')
    plt.xlabel('time')
    cu.cochshow(np.flipud(S.T), interact=False)
    plt.gca().invert_yaxis()
    plt.show()

    # Normalize Spectrogram
    S = (S - S.min()) / (S.max() - S.min())

    print(f"Coch shape: {S.shape}")
    plot_spec(S)
    return S


def downsampler(envs):
    return cgram.apply_envelope_downsample(
        envs,
        mode='poly',
        audio_sr=8000,
        env_sr=64  # Amount of timesteps
    )

def trim_or_pad(audio, sr, fixed_length):
    target_len = int(sr * fixed_length)
    if len(audio) < target_len:
        pad_len = target_len - len(audio)

        # Choose random split to pad before and after signal
        left = np.random.randint(0, pad_len + 1)
        right = pad_len - left

        audio = np.pad(audio, (left, right), mode='constant')  # pad
    else:
        audio = audio[:target_len]  # trim

    return audio

def plot_spec(spec):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spec,
                             sr=8000,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram of male speaker saying 'zero'", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

S = dothings(path1)

model = ShallowNetwork(N=1000, sr=0.8, lr=0.97, input_scaling=1, sigma=0.1, ridge=1e-7,
                                    input_dim=S.shape[1], input_width=1,
                                    reservoir_width=1, connectivity=1, IP=True)

print("Applying IP")
# a, b = model.apply_ip()
model.create_input_weights()

# Compute KL/entropy
states = model.reservoir.run(S)
plot_pdf(states, 0.1, "PDF of Reservoir States After IP")

