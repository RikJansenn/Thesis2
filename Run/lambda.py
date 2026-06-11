import numpy as np
from Models.ShallowESN import ShallowNetwork
from Models.DeepESN import DeepNetwork
from Models.DeepESNInstruments import DeepNetworkInstruments
from Models.NewDeepESN import NewDeepNetwork
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
        #data_param = np.load("../Data/mel_data_nsynth/mel_param.npz")
        data_param = np.load("../Data/mel_data/mel_param.npz")
    if SPEC == "coch":
        data_param = np.oad("../coch_data/coch_param2.npz")

    X = data_param["specs"]
    Y = data_param["targets"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, shuffle=True
    )

    return X_train, X_test, Y_train, Y_test

def load_ridge_data():
    data = np.load("../Data/mel_data/mel_ridge.npz")
    return data["specs"], data["targets"]


def search_ridge_parameter(
    ridge_values,
    X_ridge,
    Y_ridge,
    layer,
    N,
    sr,
    lr,
    sigma,
    input_dim,
    input_width,
    ip_lrs,
    IP,
    TONOTOPIC
):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_ridge = None
    best_mean_acc = -np.inf

    for ridge in ridge_values:
        print(f"\nTesting ridge={ridge}")

        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_ridge), start=1):

            X_train_fold = X_ridge[train_idx]
            X_val_fold = X_ridge[val_idx]

            Y_train_fold = Y_ridge[train_idx]
            Y_val_fold = Y_ridge[val_idx]

            model = DeepNetwork(
                n_reservoirs=layer,
                N_total=N,
                sr=sr,
                lr=lr,
                sigma=sigma,
                ridge=ridge,
                input_dim=input_dim,
                input_width=input_width,
                reservoir_width=0.2,
                connectivity=0.1,
                ip_lrs=ip_lrs,
                IP=IP
            )

            if IP:
                model.apply_ip()

            if TONOTOPIC:
                model.create_tonotopic_mapping()

            model.create_input_weights()
            model.rescale_reservoirs()

            model.train(X_train_fold, Y_train_fold)

            acc, *_ = model.test(X_val_fold, Y_val_fold)

            fold_accuracies.append(acc)

            print(
                f"  Fold {fold}: "
                f"acc={acc:.4f}"
            )

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        print(
            f"ridge={ridge:.2e} | "
            f"mean={mean_acc:.4f} ± {std_acc:.4f}"
        )

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_ridge = ridge

    print(
        f"\nBest ridge: {best_ridge:.2e} "
        f"(mean CV accuracy={best_mean_acc:.4f})"
    )

    return best_ridge

if __name__ == "__main__":
    IPs = [True]
    TONOTOPIC = False
    SPEC = "mel"

    X_train, X_test, Y_train, Y_test = create_training_data(SPEC)

    print(len(X_train))

    N_layers = [6]
    Ns = [1200]
    srs = [0.8]
    lrs = [0.94]
    sigmas = [0.1]
    input_dim = X_train[0].shape[1]
    # ip_lrs = [3.16e-4, 7.94e-6, 7.94e-6, 1e-5, 1.26e-5, 7.94e-6, 1e-5, 7.94e-6, 1e-5, 1e-5, 1e-5, 7.94e-6, 1.99e-5,
    #           1.26e-5, 1.99e-5, 1e-5, 1.99e-5, 7.94e-6, 1.99e-5, 3.16e-5]
    ip_lrs = [2.9e-4, 2.5e-5, 9.3e-6, 1.4e-5, 1.3e-5, 1.5e-5, 9.3e-6, 1.3e-5, 1.7e-5, 6.8e-6, 1.4e-5, 7.9e-6, 1.2e-5,
              7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6]
    # ip_lrs = [2.2e-4, 3.6e-4, 2.1e-4, 2.2e-4, 3.6e-4, 3.6e-4, 3.6e-4, 2.1e-4, 3.6e-4, 2.2e-4, 2.2e-4, 3.6e-4, 3.6e-4,
    #           3.6e-4, 2.2e-4]
    # ip_lrs = [2.2e-4, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4, 1.3e-4, 4.6e-5, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4]

    X_ridge, Y_ridge = load_ridge_data()

    ridge_values = [1e-3, 1e-2]

    results = []
    for IP in IPs:
        for sr in srs:
            for lr in lrs:
                for N in Ns:
                    for sigma in sigmas:
                        for layer in N_layers:
                            #N = 500 * layer
                            input_width = random.uniform(0.01, 0.7)

                            best_ridge = search_ridge_parameter(
                                ridge_values=ridge_values,
                                X_ridge=X_ridge,
                                Y_ridge=Y_ridge,
                                layer=layer,
                                N=N,
                                sr=sr,
                                lr=lr,
                                sigma=sigma,
                                input_dim=input_dim,
                                input_width=input_width,
                                ip_lrs=ip_lrs,
                                IP=IP,
                                TONOTOPIC=TONOTOPIC
                            )

                            # for i in range(4):
                                #print(input_width)
                        #         print(f"Creating model... N_layers = {layer} sr = {sr} lr = {lr} IP = {IP}")
                        #         model = DeepNetwork(n_reservoirs=layer, N_total=N, sr=sr, lr=lr, sigma=sigma, ridge=best_ridge,
                        #                             input_dim=input_dim,
                        #                             input_width=input_width, reservoir_width=0.2, connectivity=0.1, ip_lrs=ip_lrs,
                        #                             IP=IP)
                        #
                        #         if IP:
                        #             print("Applying IP")
                        #             model.apply_ip()
                        #             #model.create_input_weights()
                        #         if TONOTOPIC:
                        #             print("Applying tonotopic mapping")
                        #             model.create_tonotopic_mapping()
                        #
                        #         model.create_input_weights()
                        #         model.rescale_reservoirs()
                        #         #
                        #         # matrix = model.reservoir.W
                        #         # plt.imshow(matrix, cmap='seismic')
                        #         # plt.colorbar()
                        #         #plt.show()
                        #
                        #         print("Training...")
                        #         model.train(X_train, Y_train)
                        #
                        #         print("Testing...")
                        #         acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(X_test, Y_test)
                        #
                        #         print(acc)
                        #
                        #         results.append({
                        #             "layer": layer,
                        #             "accuracy": acc,
                        #         })
                        #
                        #
                        # results_df = pd.DataFrame(results)
                        # results_df.to_csv(f"bigger_reservoirs.csv", index=False)
