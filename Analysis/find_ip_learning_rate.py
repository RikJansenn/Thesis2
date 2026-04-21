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
matplotlib.use('tKagg')


def create_training_data(SPEC):
    if SPEC == "mel":
        data_train = np.load("../mel_data/mel_train.npz")
        data_test = np.load("../mel_data/mel_test.npz")
        data_param = np.load("../mel_data/mel_param.npz")
    elif SPEC == "linear":
        data_train = np.load("../linear_data/linear_train.npz")
        data_test = np.load("../linear_data/linear_test.npz")
    elif SPEC == "coch":
        data = np.load("../datasets/dataset_cochs_lowres.npz")

    X_train = data_train["melspecs"]
    Y_train = data_train["targets_mel"]
    X_test = data_test["melspecs"]
    Y_test = data_test["targets_mel"]
    X_param = data_param["melspecs"]
    Y_param = data_param["targets_mel"]

    return X_train, Y_train, X_test, Y_test, X_param, Y_param

if __name__ == "__main__":
    IP = True
    TONOTOPIC = False
    SPEC = "mel"

    X_train, Y_train, X_test, Y_test, X_param, Y_param = create_training_data(SPEC)

    Ns = [100, 100, 200]
    sr = 0.8
    lrs = [0.9]
    sigmas = [0.1]

    results = []
    for N in Ns:
        for lr in lrs:
            for sigma in sigmas:
                print("Creating model...")
                model = DeepNetwork(n_reservoirs=10, N=N, sr=sr, lr=lr, sigma=sigma, ridge=1e-7, input_dim=40,
                                    input_width=0.06, reservoir_width=0.2, connectivity=0.1, IP=IP)

                # model.rescale_reservoirs()

                combined = np.concatenate(X_param[:150], axis=0)
                combined = combined[~np.all(combined == 0, axis=1)]

                n, centroids = model.find_optimal_layers(15, combined, 10, 0.01)

                plt.figure()
                plt.plot(centroids)
                print(n)
                plt.savefig(f"centroids_{N}_{sr}_{lr}_{sigma}_ip={IP}.png")

                # if IP:
                #     print("Applying IP")
                #     model.apply_ip()
                #     model.create_input_weights()
                # if TONOTOPIC:
                #     print("Applying tonotopic mapping")
                #     model.create_tonotopic_mapping()
                # # :
                # # model.create_input_weights()
                #
                # model.rescale_reservoirs()
                #
                # print("Training...")
                # model.train(X_train, Y_train)
                #
                # print("Testing...")
                # acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(X_test, Y_test)
                #
                # print(acc)

                results.append({
                    "N": N,
                    "sr": sr,
                    "lr": lr,
                    "sigma": sigma,
                    "n": n,
                })

                results_df = pd.DataFrame(results)
                #results_df.to_csv(f"results_deep_layers2.csv", index=False)
