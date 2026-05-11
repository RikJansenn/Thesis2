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
    IPs = [True]
    TONOTOPIC = False
    SPEC = "mel"

    X_train, X_test, Y_train, Y_test = create_training_data(SPEC)

    N_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Ns = [1000, 1200]
    sr = 0.8
    lrs = [0.94, 0.97]
    sigmas = [0.1, 0.2]
    input_dim = X_train[0].shape[1]
    ip_lrs = [3.16e-4, 7.94e-6, 1.99e-5, 1.26e-5, 7.94e-6, 1.26e-5, 1.26e-5, 7.94e-6, 1e-5, 1.26e-5, 7.94e-6, 1e-5,
              1.26e-5, 3.16e-5, 1.26e-5, 1.26e-5, 5.01e-5, 3.16e-5, 3.16e-5, 7.94e-5]
    # ip_lrs = [3.16e-4, 7.94e-6, 7.94e-6, 1e-5, 1.26e-5, 7.94e-6, 1e-5, 7.94e-6, 1e-5, 1e-5, 1e-5, 7.94e-6, 1.99e-5,
             # 1.26e-5, 1.99e-5, 1e-5, 1.99e-5, 7.94e-6, 1.99e-5, 3.16e-5] # MNIST
    # ip_lrs = [2.9e-4, 2.5e-5, 9.3e-6, 1.4e-5, 1.3e-5, 1.5e-5, 9.3e-6, 1.3e-5, 1.7e-5, 6.8e-6, 1.4e-5, 7.9e-6, 1.2e-5,
    #           7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6]
    # ip_lrs = [2.2e-4, 3.6e-4, 2.1e-4, 2.2e-4, 3.6e-4, 3.6e-4, 3.6e-4, 2.1e-4, 3.6e-4, 2.2e-4, 2.2e-4, 3.6e-4, 3.6e-4,
    #           3.6e-4, 2.2e-4]
    # ip_lrs = [2.2e-4, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4, 1.3e-4, 4.6e-5, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4]


    results = []
    for IP in IPs:
        for lr in lrs:
            for N in Ns:
                for sigma in sigmas:
                    for layer in N_layers:
                        for i in range(5):
                            print("Creating model...")
                            model = DeepNetwork(n_reservoirs=layer, N_total=N, sr=sr, lr=lr, sigma=sigma, ridge=1e-7,
                                                input_dim=input_dim,
                                                input_width=0.06, reservoir_width=0.2, connectivity=0.1, ip_lrs=ip_lrs,
                                                IP=IP)

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
                                "N_layers": layer,
                                "sr": sr,
                                "lr": lr,
                                "sigma": sigma,
                                "accuracy": acc,
                            })

                    results_df = pd.DataFrame(results)
                    results_df.to_csv(f"results_deep_params_{lr}_{sr}_{sigma}_{N}_IP={IP}_nsynth.csv", index=False)
