import numpy as np
from Models.ShallowESN import ShallowNetwork
from Models.DeepESN import DeepNetwork
import time
from reservoirpy.observables import effective_spectral_radius
from utils import get_KL_divergence_and_entropy
import pandas as pd
import random
from reservoirpy.observables import effective_spectral_radius
import pickle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib
import librosa
from reservoirpy.observables import spectral_radius, effective_spectral_radius
from sklearn.model_selection import KFold, train_test_split
matplotlib.use('tKagg')


def create_training_data(SPEC):
    if SPEC == "mel":
        data_param = np.load("../Data/mel_data/mel_param.npz")
    if SPEC == "coch":
        data_param = np.oad("../coch_data/coch_param2.npz")

    X = data_param["specs"]
    Y = data_param["targets"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, shuffle=True
    )

    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    IP = True
    TONOTOPIC = False
    SPEC = "mel"

    X_train, X_test, Y_train, Y_test = create_training_data(SPEC)

    N_layers = 14
    Ns = [71]
    sr = 0.8
    lrs = [0.9]
    sigmas = [0.1]
    input_dim = X_train[0].shape[1]
    ip_lrs = [2.9e-4, 2.5e-5, 9.3e-6, 1.4e-5, 1.3e-5, 1.5e-5, 9.3e-6, 1.3e-5, 1.7e-5, 6.8e-6, 1.4e-5, 7.9e-6, 1.2e-5,
              7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6]

    results = []
    for N in Ns:
        for lr in lrs:
            for sigma in sigmas:
                for i in range(20):
                    print("Creating model...")
                    model = DeepNetwork(n_reservoirs=N_layers, N=N, sr=sr, lr=lr, sigma=sigma, ridge=1e-7, input_dim=input_dim,
                                        input_width=0.06, reservoir_width=0.2, connectivity=0.1, ip_lrs=ip_lrs, IP=IP)

                    if IP:
                        print("Applying IP")
                        model.apply_ip()
                        model.create_input_weights()
                    if TONOTOPIC:
                        print("Applying tonotopic mapping")
                        model.create_tonotopic_mapping()

                    model.create_input_weights()
                    model.rescale_reservoirs()

                    print("Training...")
                    model.train(X_train, Y_train)

                    print("Testing...")
                    acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(X_test, Y_test)

                    print(acc)

                    results.append({
                        "N": N,
                        "sr": sr,
                        "lr": lr,
                        "sigma": sigma,
                        "accuracy": acc,
                    })

                    results_df = pd.DataFrame(results)
                    results_df.to_csv(f"results_deep_params_stoch.csv", index=False)
