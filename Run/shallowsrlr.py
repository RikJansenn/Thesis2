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
        data_param = np.load("../coch_data/coch_param2.npz")

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

    print(len(X_train))

    N_layers = [6]
    Ns = [1200]
    srs = [0.2, 0.4, 0.6, 0.8]
    lrs = [0.9, 0.94, 0.97]
    sigmas = [0.1]
    input_dim = X_train[0].shape[1]
    # ip_lrs = [3.16e-4, 7.94e-6, 7.94e-6, 1e-5, 1.26e-5, 7.94e-6, 1e-5, 7.94e-6, 1e-5, 1e-5, 1e-5, 7.94e-6, 1.99e-5,
    #           1.26e-5, 1.99e-5, 1e-5, 1.99e-5, 7.94e-6, 1.99e-5, 3.16e-5]
    ip_lrs = [2.9e-4, 2.5e-5, 9.3e-6, 1.4e-5, 1.3e-5, 1.5e-5, 9.3e-6, 1.3e-5, 1.7e-5, 6.8e-6, 1.4e-5, 7.9e-6, 1.2e-5,
              7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6, 7.9e-6]
    # ip_lrs = [2.2e-4, 3.6e-4, 2.1e-4, 2.2e-4, 3.6e-4, 3.6e-4, 3.6e-4, 2.1e-4, 3.6e-4, 2.2e-4, 2.2e-4, 3.6e-4, 3.6e-4,
    #           3.6e-4, 2.2e-4]
    # ip_lrs = [2.2e-4, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4, 1.3e-4, 4.6e-5, 1.3e-4, 1.3e-4, 7.7e-5, 1.3e-4, 7.7e-5, 1.3e-4]


    results = []
    for IP in IPs:
        for sr in srs:
            for lr in lrs:
                for N in Ns:
                    for sigma in sigmas:
                        for layer in N_layers:
                            for i in range(5):
                                input_width = random.uniform(0.01, 0.7)
                                # print(input_width)
                                print(f"Creating model... N_layers = {layer} sr = {sr} lr = {lr} IP = {IP}")
                                model = ShallowNetwork(N=N, sr=sr, lr=lr, sigma=sigma, ridge=1e-7,
                                                    input_dim=input_dim,
                                                    input_width=input_width,
                                                    IP=IP)

                                if IP:
                                    print("Applying IP")
                                    model.apply_ip()
                                    #model.create_input_weights()
                                if TONOTOPIC:
                                    print("Applying tonotopic mapping")
                                    model.create_tonotopic_mapping()

                                model.create_input_weights()
                                # model.rescale_reservoirs()
                                #
                                # matrix = model.reservoir.W
                                # plt.imshow(matrix, cmap='seismic')
                                # plt.colorbar()
                                #plt.show()

                                print("Training...")
                                model.train(X_train, Y_train)

                                print("Testing...")
                                acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(X_test, Y_test)

                                print(acc)

                                results.append({
                                    # "layer": layer,
                                    # "N": N,
                                    "sr": sr,
                                    # "sigma"
                                    "lr": lr,
                                    # "input_width": input_width,
                                    "accuracy": acc,
                                })

                                # timesteps_df = pd.DataFrame(timestep_predictions)
                                # timesteps_df.to_csv("timesteps_csv.csv", index=False)
                                #
                                # labels_df = pd.DataFrame(y_per_timestep)
                                # labels_df.to_csv("labels_csv.csv", index=False)


                        results_df = pd.DataFrame(results)
                        results_df.to_csv(f"srlrtest_shallow.csv", index=False)
