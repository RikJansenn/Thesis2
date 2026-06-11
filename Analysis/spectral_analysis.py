import numpy as np
from Models.ShallowESN import ShallowNetwork
from Models.DeepESN import DeepNetwork
from Models.NewDeepESN import NewDeepNetwork
import time
from reservoirpy.observables import effective_spectral_radius
from utils import get_KL_divergence_and_entropy
import pandas as pd
import random
from reservoirpy.observables import effective_spectral_radius
import pickle
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import matplotlib
import librosa
from reservoirpy.observables import spectral_radius, effective_spectral_radius
matplotlib.use('tKagg')


def create_training_data(SPEC):
    # if SPEC == "mel":
    #     data_train = np.load("../Data/mel_data/mel_train.npz")
    #     data_test = np.load("../Data/mel_data/mel_test.npz")
    #     data_param = np.load("../Data/mel_data/mel_param.npz")
    # elif SPEC == "linear":
    #     data_train = np.load("../Data/linear_data/linear_train.npz")
    #     data_test = np.load("../Data/linear_data/linear_test.npz")
    # elif SPEC == "coch":
    #     data = np.load("../datasets/dataset_cochs_lowres.npz")
    #
    # X_train = data_train["specs"]
    # Y_train = data_train["targets"]
    # X_test = data_test["specs"]
    # Y_test = data_test["targets"]
    # X_param = data_param["specs"]
    # Y_param = data_param["targets"]

    data = np.load("../Data/linear_data/linear_train.npz")

    X = data["specs"]
    Y = data["targets"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )

    return X[:3000], Y[:3000]

def impulse_train(period, T):
    x = np.zeros(T)
    x[::period] = 1
    return x

def generate_fbc_signal(T=6000, switch_prob=0.01, seed=0):
    rng = np.random.default_rng(seed)

    # subsequence s1 (periods 3..29)
    s1 = sum(impulse_train(p, T) for p in range(3, 30))

    # subsequence s0 (periods 3..31)
    s0 = sum(impulse_train(p, T) for p in range(3, 32))

    # hidden regime switching
    regime = np.zeros(T)
    regime[0] = rng.integers(0, 2)

    for t in range(1, T):
        if rng.random() < switch_prob:
            regime[t] = 1 - regime[t-1]
        else:
            regime[t] = regime[t-1]

    # build final signal
    s = np.where(regime == 1, s1, s0)

    return s.reshape(-1, 1)

if __name__ == "__main__":
    IP = True
    TONOTOPIC = False
    SPEC = "mel"

    X, Y = create_training_data(SPEC)

    Ns = [50, 100, 200]
    sr = 0.7
    lrs = [0.5]
    sigmas = [0.1]
    ip_lrs = [2.9e-4, 2.5e-5, 9.3e-6, 1.4e-5, 1.3e-5, 1.5e-5, 9.3e-6, 1.3e-5, 1.7e-5, 6.8e-6, 1.4e-5, 7.9e-6, 1.2e-5,
              7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6] # MNIST
    # ip_lrs = [3.16e-4, 7.94e-6, 1.99e-5, 1.26e-5, 7.94e-6, 1.26e-5, 1.26e-5, 7.94e-6, 1e-5, 1.26e-5, 7.94e-6, 1e-5, 1.26e-5, 3.16e-5, 1.26e-5, 1.26e-5, 5.01e-5, 3.16e-5, 3.16e-5, 7.94e-5]
    # ip_lrs = [2.2e-4, 3.6e-4, 2.1e-4, 2.2e-4, 4.6e-4, 3.6e-4, 3.6e-4, 2.1e-4, 3.6e-4, 2.2e-4, 2.2e-4, 3.6e-4, 3.6e-4, 3.6e-4, 2.2e-4]
    # ip_lrs = [2.2e-4, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4, 1.3e-4, 4.6e-5, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4]
    # ip_lrs = [2.2e-4]
    results = []
    for lr in lrs:
        for sigma in sigmas:
            print("Creating model...")
            model = DeepNetwork(n_reservoirs=6, N_total=1000, sr=sr, lr=lr, sigma=sigma, ridge=1e-7, input_dim=40,
                                input_width=0.06, reservoir_width=0.2, connectivity=0.1, ip_lrs=ip_lrs, IP=IP)


            combined = np.concatenate(X[:100], axis=0)
            combined = combined[~np.all(combined == 0, axis=1)]
            print(len(combined))

            n, centroids = model.find_optimal_layers(15, combined, 10, 0.01)

            plt.figure()
            plt.plot(centroids)
            print(n)
            plt.show()
            # plt.savefig(f"centroids_{sr}_{lr}_{sigma}_ip={IP}_nsynth.png")


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
                "sr": sr,
                "lr": lr,
                "sigma": sigma,
            })

            results_df = pd.DataFrame(results)
            #results_df.to_csv(f"results_deep_layers2.csv", index=False)
