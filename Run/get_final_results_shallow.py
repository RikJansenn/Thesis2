import numpy as np
import pandas as pd
import time
from reservoirpy.observables import effective_spectral_radius
from sklearn.model_selection import KFold
from Models.ShallowESN import ShallowNetwork

def create_training_data(SPEC):
    """
    Load training data
    :param SPEC: Which type of input data to use
    :return:
        X: Spectrograms
        Y: Labels
    """
    if SPEC == "linear":
        data_train = np.load("../Data/linear_data/linear_train.npz")
        data_test = np.load("../Data/linear_data/linear_train.npz")
    elif SPEC == "mel":
        data_train = np.load("../Data/mel_data/mel_train.npz")
        data_test = np.load("../Data/mel_data/mel_test.npz")
    elif SPEC == "coch":
        data_train = np.load("../Data/coch_data/coch_train2.npz")
        data_test = np.load("../Data/coch_data/coch_test2.npz")

    # I concatenate train + test set here because I changed from having a test set to using k-fold cross validation
    X_train = data_train["specs"]
    Y_train = data_train["targets"]
    X_test = data_test["specs"]
    Y_test = data_test["targets"]

    X = np.concatenate([X_train, X_test], axis=0)
    Y = np.concatenate([Y_train, Y_test], axis=0)

    return X[:3000], Y[:3000]


def get_results(param_sets, IP, TONOTOPIC, SPEC):
    """
    Gets full result from given parameter sets and a given model configuration
    :param param_sets: All parameter sets
    :param IP: Boolean - Apply IP or not
    :param TONOTOPIC: Boolean - Apply tonotopic mapping or not
    :param SPEC: String - Which input type to use
    :return: Saves full results in csv files
    """
    # Directory in which to save results, based on model configuration
    save_dir = f"../results/{'ip_' if IP else 'stoch_'}{'tono_' if TONOTOPIC else ''}{SPEC}"

    # Create training data
    X, Y = create_training_data(SPEC)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

                # Create model
                print("Creating model...")
                model = ShallowNetwork(N=N, sr=sr, lr=lr, input_scaling=1, sigma=sigma, ridge=1e-7,
                                       input_dim=X_train[0].shape[1], input_width=0.06,
                                       reservoir_width=0.2, connectivity=0.1, IP=IP)

                # Apply IP
                if IP:
                    print("Applying IP")
                    a, b = model.apply_ip()

                # Create tonotopic mapping. Create_input_weights is not needed with tonotopic mapping since tonotopic
                # mapping overrides this
                if TONOTOPIC:
                    print("Applying tonotopic mapping")
                    model.create_tonotopic_mapping()
                else:
                    model.create_input_weights()

                # Compute effective spectral radius
                esr = effective_spectral_radius(model.reservoir.W, lr=lr)

                # Train model
                print("Training model...")
                kl, ent = model.train(X_train, Y_train)

                # Test model
                print("Testing model...")
                acc, y_true, y_pred, timestep_predictions, y_per_timestep = model.test(X_test, Y_test)

                print(f"Fold {fold} accuracy: {acc:.3f}")

                results.append({
                    "N": N,
                    "sr": sr,
                    "lr": lr,
                    "sigma": sigma,
                    "fold": fold,
                    "iteration": i,
                    "kl_divergence": kl,
                    "accuracy": acc,
                    "esr": esr,
                })

                end_time = time.time()
                print(f"Total runtime: {end_time - start_time:.2f} seconds")

                # Save trained model
                model.save(
                    f"{save_dir}/trained_models/trained_model_N{N}_sr{sr}_lr{lr}_sigma{sigma}_fold_{fold}_run{i}.pkl")

                # Save timestep predictions + labels
                timestep_df = pd.DataFrame(timestep_predictions)
                timestep_df.to_csv(
                    f"{save_dir}/timesteps/timestep_predictions_N{N}_sr{sr}_lr{lr}_sigma{sigma}_fold_{fold}_run{i}.csv",
                    index=False)

                labels_df = pd.DataFrame(y_per_timestep)
                labels_df.to_csv(
                    f"{save_dir}/timesteps//labels_per_timestep_N{N}_sr{sr}_lr{lr}_sigma{sigma}_fold_{fold}_run{i}.csv",
                    index=False)

        # Save results so far
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{save_dir}/results_N{N}_sr{sr}_lr{lr}_sigma{sigma}.csv",
                          index=False)


if __name__ == "__main__":
    """
    Define model configuration and parameter sets and call get_results for each
    """
    # Define which model configurations to test with
    IPs = [False, True]
    TONOs = [True]
    SPECs = ["linear", "mel", "coch"]

    for IP in IPs:
        for TONO in TONOs:
            for SPEC in SPECs:
                # Define which parameter sets to test with
                if IP:
                    param_sets = [
                        {"N": 1000, "sr": 0.8, "lr": 0.94, "sigma": 0.1},
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
                        {"N": 1200, "sr": 0.8, "lr": 0.94, "sigma": 0.1},
                        {"N": 1000, "sr": 0.8, "lr": 0.97, "sigma": 0.1},
                        {"N": 1200, "sr": 0.8, "lr": 0.94, "sigma": 0.1},
                        {"N": 1200, "sr": 0.8, "lr": 0.97, "sigma": 0.1},
                    ]

                get_results(param_sets=param_sets, IP=IP, TONOTOPIC=TONO, SPEC=SPEC)
