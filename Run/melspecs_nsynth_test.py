import numpy as np
from DeepESN import DeepNetworkInstruments
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
        # data_param = np.load("../Data/mel_data_nsynth/mel_param.npz")
        data_param = np.load("../mel_data/mel_data_nsynth/mel_param.npz")

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

    print(X_train[0].shape)

    Ns = [800, 1000, 1200]
    srs = [0.4, 0.6, 0.8]
    lrs = [0.94, 0.97, 1]
    sigmas = [0.1, 0.2, 0.3]
    input_dim = X_train[0].shape[1]
    ip_lrs = [3.16e-4, 7.94e-6, 1.99e-5, 1.26e-5, 7.94e-6, 1.26e-5, 1.26e-5, 7.94e-6, 1e-5, 1.26e-5, 7.94e-6, 1e-5,
              1.26e-5, 3.16e-5, 1.26e-5, 1.26e-5, 5.01e-5, 3.16e-5, 3.16e-5, 7.94e-5]

    results = []
    for IP in IPs:
        for sr in srs:
            for lr in lrs:
                for N in Ns:
                    for sigma in sigmas:
                        for i in range(25):
                            if sr == 0.4 and not IP:
                                layers = 6
                            else:
                                layers = 4

                            print(f"Creating model...  sr = {sr} lr = {lr} IP = {IP}")
                            model = DeepNetworkInstruments(n_reservoirs=layers, N_total=N, sr=sr, lr=lr, sigma=sigma,
                                                           ridge=1e-7,
                                                           input_dim=input_dim,
                                                           input_width=0.06, reservoir_width=0.2, connectivity=0.1,
                                                           ip_lrs=ip_lrs,
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

                            idx = np.random.randint(len(X_test))
                            states = model.forward.run(X_test[idx])
                            states = list(states.values())
                            states = np.concatenate(states, axis=-1)
                            kl, ent = get_KL_divergence_and_entropy(states, sigma)

                            # Compute effective spectral radius
                            esr = 0
                            for reservoir in model.reservoirs:
                                esr += effective_spectral_radius(reservoir.W, lr=lr)
                            esr = esr / model.n_reservoirs

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
                                "IP": IP,
                                "esr": esr,
                                "kl": kl,
                                "accuracy": acc,
                            })

                    results_df = pd.DataFrame(results)
                    results_df.to_csv(f"results_deep_params_{lr}_{sr}_{sigma}_{N}_IP={IP}.csv",
                                      index=False)
