import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from reservoirpy.observables import effective_spectral_radius

from Models.InstrumentNetwork import InstrumentNetwork
from utils import *


def create_training_data(SPEC):
    if SPEC == "mel":
        data = np.load("npz.npz")

    # Combine splits for k-fold cross validation
    X = data["specs"]
    Y = data["targets"]

    return X, Y


def get_results(param_sets, IP, TONOTOPIC, SPEC):
    save_dir = f"../results/instruments_{'ip_' if IP else 'stoch_'}{'tono_' if TONOTOPIC else ''}{SPEC}"

    os.makedirs(f"{save_dir}/trained_models", exist_ok=True)
    os.makedirs(f"{save_dir}/timesteps", exist_ok=True)

    X, Y = create_training_data(SPEC)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    iterations = 10

    for params in param_sets:
        N = params["N"]
        sr = params["sr"]
        lr = params["lr"]
        sigma = params["sigma"]

        results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            for i in range(iterations):
                start_time = time.time()

                print("Creating model...")

                model = InstrumentNetwork(
                    N=N,
                    sr=sr,
                    lr=lr,
                    sigma=sigma,
                    ridge=1e-7,
                    input_dim=X_train[0].shape[1],
                    input_width=0.06,
                    IP=IP,
                )

                if IP:
                    print("Applying IP...")
                    a, b = model.apply_ip()

                if TONOTOPIC:
                    print("Applying tonotopic mapping...")
                    model.create_tonotopic_mapping()
                else:
                    print("Creating random input weights...")
                    model.create_input_weights()

                esr = effective_spectral_radius(
                    model.reservoir.W,
                    lr=lr,
                )

                print("Training model...")
                train_output = model.train(X_train, Y_train)

                if IP:
                    kl, ent = train_output
                else:
                    kl, ent = None, None

                print("Testing model...")
                acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(
                    X_test,
                    Y_test,
                )

                f1_macro = f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )

                f1_weighted = f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )

                print(
                    f"Fold {fold}, run {i} | "
                    f"Accuracy: {acc:.3f} | "
                    f"Macro F1: {f1_macro:.3f}"
                )

                results.append({
                    "N": N,
                    "sr": sr,
                    "lr": lr,
                    "sigma": sigma,
                    "fold": fold,
                    "iteration": i,
                    "kl_divergence": kl,
                    "entropy": ent,
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "esr": esr,
                })

                end_time = time.time()
                print(f"Total runtime: {end_time - start_time:.2f} seconds")

                model.save(
                    f"{save_dir}/trained_models/"
                    f"trained_model_N{N}_sr{sr}_lr{lr}_sigma{sigma}_fold{fold}_run{i}.pkl"
                )

                timestep_df = pd.DataFrame(timestep_predictions)
                timestep_df.to_csv(
                    f"{save_dir}/timesteps/"
                    f"timestep_predictions_N{N}_sr{sr}_lr{lr}_sigma{sigma}_fold{fold}_run{i}.csv",
                    index=False,
                )

                labels_df = pd.DataFrame(y_per_timestep)
                labels_df.to_csv(
                    f"{save_dir}/timesteps/"
                    f"labels_per_timestep_N{N}_sr{sr}_lr{lr}_sigma{sigma}_fold{fold}_run{i}.csv",
                    index=False,
                )

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            f"{save_dir}/results_N{N}_sr{sr}_lr{lr}_sigma{sigma}.csv",
            index=False,
        )


if __name__ == "__main__":
    IPs = [True, False]
    TONOs = [False]
    SPECs = ["mel"]

    for IP in IPs:
        for TONO in TONOs:
            for SPEC in SPECs:
                if IP:
                    param_sets = [
                        {"N": 50, "sr": 0.8, "lr": 0.97, "sigma": 0.1},
                        {"N": 1000, "sr": 0.8, "lr": 0.97, "sigma": 0.1},
                        {"N": 1200, "sr": 0.8, "lr": 0.94, "sigma": 0.1},
                        {"N": 1200, "sr": 0.8, "lr": 0.97, "sigma": 0.1},
                        {"N": 1000, "sr": 0.8, "lr": 0.94, "sigma": 0.2},
                        {"N": 1000, "sr": 0.8, "lr": 0.97, "sigma": 0.2},
                        {"N": 1200, "sr": 0.8, "lr": 0.94, "sigma": 0.2},
                        {"N": 1200, "sr": 0.8, "lr": 0.97, "sigma": 0.2},
                    ]
                else:
                    param_sets = [
                        {"N": 1000, "sr": 0.8, "lr": 0.94, "sigma": 0.1},
                        {"N": 1000, "sr": 0.8, "lr": 0.97, "sigma": 0.1},
                        {"N": 1200, "sr": 0.8, "lr": 0.94, "sigma": 0.1},
                        {"N": 1200, "sr": 0.8, "lr": 0.97, "sigma": 0.1},
                    ]

                get_results(
                    param_sets=param_sets,
                    IP=IP,
                    TONOTOPIC=TONO,
                    SPEC=SPEC,
                )