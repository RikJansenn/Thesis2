from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from reservoirpy.datasets import narma
from utils import get_KL_divergence_and_entropy, plot_pdf
from numpy.linalg import eigvals
from functools import reduce
from operator import and_
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tKagg")

class DeepNetwork:
    def __init__(self, n_reservoirs, N_total, sr, lr, sigma, ridge, input_dim, input_width, reservoir_width, connectivity, ip_lrs, IP=True):
        self.n_reservoirs = n_reservoirs
        self.N_total = N_total
        self.sr = sr
        self.lr = lr
        self.sigma = sigma # Consistent across reservoirs?
        self.ridge = ridge # Just one readout layer
        self.input_dim = input_dim # Just one input layer
        self.input_width = input_width # Just one input layer
        self.reservoir_width = reservoir_width
        self.connectivity = connectivity # Consistent across reservoirs?
        self.IP = IP
        self.ip_lrs = ip_lrs
        self.workers = 1

        self._build_model(n_reservoirs)

    def _build_model(self, n_reservoirs):
        self.reservoirs = []
        sr = self.sr
        lr = self.lr
        N = int(round(self.N_total / n_reservoirs))
        for i in range(n_reservoirs):
            if self.IP:
                reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=self.sigma, activation="tanh",
                                        epochs=4,
                                        learning_rate=self.ip_lrs[i], dtype=np.float32, name=f"reservoir_{i}")
            else:
                reservoir = Reservoir(N, sr=sr, lr=lr, dtype=np.float32, name=f"reservoir_{i}")

            self.reservoirs.append(reservoir)
            # sr += 0.05
            # lr -= 0.05

        self.readout = Ridge(ridge=self.ridge, output_dim=11, name="readout")

        self.forward = self.reservoirs[0]
        for reservoir in self.reservoirs[1:]:
            self.forward = self.forward >> reservoir

        # Add readout from all reservoirs
        self.model = self.forward & self.reservoirs >> self.readout

        # Configure forward part to give outputs of all reservoirs after calling run
        self.forward.outputs = self.reservoirs
        self.forward.is_multi_output = True

    # def find_optimal_ip_lr(self, X, previous_lrs, layer, iterations=10):
    #     lrs = 1 * 10 ** np.array([-7, -6, -5, -4, -3], dtype=float)
    #     mean_kls = []
    #
    #     best_lr = None
    #     best_mean_kl = np.inf
    #
    #     for lr in lrs:
    #         kls = []
    #
    #         for i in range(iterations):
    #             self.ip_lrs = previous_lrs + [lr]
    #             self._build_model(layer)
    #             self.apply_ip()
    #             self.create_input_weights()
    #
    #             states = self.forward.run(X, workers=self.workers)
    #
    #             if layer > 1:
    #                 states = list(states.values())
    #                 kl, ent = get_KL_divergence_and_entropy(states[-1], self.sigma)
    #             else:
    #                 kl, ent = get_KL_divergence_and_entropy(states, self.sigma)
    #
    #             kls.append(kl)
    #
    #         mean_kl = np.mean(kls)
    #         mean_kls.append(mean_kl)
    #
    #         print(f"Mean KL for layer {layer} and lr {lr}: {mean_kl}")
    #
    #         if mean_kl < best_mean_kl:
    #             best_mean_kl = mean_kl
    #             best_lr = lr
    #
    #     return best_lr

    import numpy as np

    def _evaluate_ip_lrs(self, X, previous_lrs, layer, lrs, iterations=5):
        best_lr = None
        best_mean_kl = np.inf
        results = []

        for lr in lrs:
            kls = []

            for _ in range(iterations):
                self.ip_lrs = previous_lrs + [lr]
                self._build_model(layer)
                self.apply_ip()
                self.create_input_weights()

                states = self.forward.run(X, workers=self.workers)

                if layer > 1:
                    states = list(states.values())
                    kl, ent = get_KL_divergence_and_entropy(states[-1], self.sigma)
                else:
                    kl, ent = get_KL_divergence_and_entropy(states, self.sigma)

                kls.append(kl)

            mean_kl = np.mean(kls)
            std_kl = np.std(kls)
            results.append((lr, mean_kl, std_kl))

            print(f"Mean KL for layer {layer} and lr {lr:.3e}: {mean_kl:.6f} ± {std_kl:.6f}")

            if mean_kl < best_mean_kl:
                best_mean_kl = mean_kl
                best_lr = lr

        return best_lr, best_mean_kl, results

    def find_optimal_ip_lr(
            self,
            X,
            previous_lrs,
            layer,
            iterations=10,
            coarse_exponents=(-6, -5, -4),
            refine_radius=0.9,
            points_per_round=10,
            min_exp=-10,
            max_exp=0,
    ):
        coarse_lrs = 10 ** np.array(coarse_exponents, dtype=float)

        best_lr, _, _ = self._evaluate_ip_lrs(
            X=X,
            previous_lrs=previous_lrs,
            layer=layer,
            lrs=coarse_lrs,
            iterations=iterations,
        )

        center = np.log10(best_lr)
        low = max(min_exp, center - refine_radius)
        high = min(max_exp, center + refine_radius)

        refine_lrs = 10 ** np.linspace(low, high, points_per_round)

        best_lr, _, _ = self._evaluate_ip_lrs(
            X=X,
            previous_lrs=previous_lrs,
            layer=layer,
            lrs=refine_lrs,
            iterations=iterations,
        )

        return best_lr


    def find_optimal_layers(self, max_layers, X, guesses, eta):
        washout = 100
        centroids = []
        lrs = []
        for n in range(1, max_layers + 1):
            all_last_states = []
            all_secondtolast_states = []
            lr = self.find_optimal_ip_lr(X, lrs, n)
            lrs.append(lr)
        #     for g in range(guesses):
        #         # self.ip_lrs = lrs
        #         self._build_model(n)
        #         if self.IP:
        #             self.apply_ip()
        #         self.create_input_weights()
        #
        #         guess_states = self.forward.run(X, workers=self.workers)
        #
        #         # plt.plot(guess_states[:,0] - np.mean(guess_states[:,0]))
        #         # plt.show()
        #
        #         if n > 1:
        #             guess_states = list(guess_states.values())
        #
        #             # if n == 15:
        #             #     kl, ent = get_KL_divergence_and_entropy(guess_states[-1], self.sigma)
        #             #     print(f"kl: {kl}")
        #             #     plot_pdf(guess_states[-1], self.sigma, f"IP on layer {n}")
        #
        #             all_last_states.append(guess_states[-1][washout:])
        #             all_secondtolast_states.append(guess_states[-2][washout:])
        #         else:
        #             all_last_states.append(guess_states[washout:])
        #
        #             # kl, ent = get_KL_divergence_and_entropy(guess_states, self.sigma)
        #             # print(f"kl: {kl}")
        #             # plot_pdf(guess_states, self.sigma, f"IP on layer {n}")
        #
        #     p, f_norm, f = self.compute_fft(all_last_states)
        #
        #     p_norm = p / np.max(p)
        #     if n == max_layers:
        #         plt.figure(figsize=(10, 5))
        #         markerline, stemlines, baseline = plt.stem(f, p_norm)
        #
        #         plt.setp(markerline, visible=False)
        #         plt.setp(baseline, visible=False)
        #         plt.ylim(bottom=0)
        #
        #         plt.xlabel("Frequency")
        #         plt.ylabel("Normalized Magnitude")
        #         plt.title("Frequency Spectrum")
        #         plt.show()
        #
        #     mu = np.sum(p * f_norm) / np.sum(p)
        #     sigma = np.sqrt(np.sum(p * (f_norm - mu) ** 2) / np.sum(p))
        #
        #     mu_f = np.sum(p * f) / np.sum(p)
        #
        #     centroids.append(mu_f)
        #     print(f"layer {n} centroid: {mu_f:.4f}")
        #
        #     if n > 1:
        #         p2, f2, _ = self.compute_fft(all_secondtolast_states)
        #         mu2 = np.sum(p2 * f2) / np.sum(p2)
        #         sigma2 = np.sqrt(np.sum(p2 * (f2 - mu2) ** 2) / np.sum(p2))
        #
        #         if abs(mu - mu2) <= eta * sigma2:
        #             print(f"Stopping at layer {n}")
        #             # return n, centroids
        #         elif mu > mu2:
        #             print(f"Stopping at layer {n-1}")
        #             # return n-1, centroids
        #
        # return max_layers, centroids

    def compute_fft(self, states):
        comps_g = []
        # Loop through reservoir guesses
        for guess_states in states:
            comps_u = []
            # Loop through individual node signals
            for unit_signal in guess_states.T:
                # Calculate frequency components
                unit_signal = unit_signal# - np.mean(unit_signal)
                comps = np.fft.fft(unit_signal)
                T = len(comps)
                magnitude = np.abs(comps[1:T//2])
                comps_u.append(magnitude)

            # Get average frequency vector for guess (across all sequences)
            comps_u = np.array(comps_u)
            comps_g.append(comps_u.mean(axis=0))

        # Get average frequency vector across all guesses (p)
        comps_g = np.array(comps_g)
        p = comps_g.mean(axis=0)
        # p_std = comps_g.std(axis=0)
        #
        # eps = 1e-12
        # p_cv = p_std / (p + eps)
        # overall_cv = p_cv.mean()
        #
        # print(f"overall_cv: {overall_cv}")
        #
        # n_guesses = comps_g.shape[0]
        # p_sem = p_std / np.sqrt(n_guesses)
        #
        # print(f"p_sem: {p_sem}")
        #
        # overall_consistency = p_std.mean()
        #
        # print(f"overall_consistency: {overall_consistency}")
        #
        # corr_matrix = np.corrcoef(comps_g)
        # n = corr_matrix.shape[0]
        # mean_pairwise_corr = (corr_matrix.sum() - np.trace(corr_matrix)) / (n * (n - 1))
        #
        # print(f"mean_pairwise_corr: {mean_pairwise_corr}")

        f_norm = np.arange(1, T//2 + 1) / T
        f = np.fft.fftfreq(T, d=1 / 8000)[1:T // 2]

        return p, f_norm[1:], f

    def rescale_reservoirs(self):
        for reservoir in self.reservoirs:
            W = np.random.uniform(-1, 1, (reservoir.units, reservoir.units))
            mask = np.random.rand(reservoir.units, reservoir.units) < 0.1
            W *= mask
            reservoir.W = W

            # Identity matrix
            I = np.eye(reservoir.units)

            W_t = (1 - reservoir.lr) * I + reservoir.lr * reservoir.W
            W_t = reservoir.sr * W_t / max(abs(eigvals(W_t)))
            reservoir.W = (W_t - (1 - reservoir.lr) * I)/reservoir.lr

    def create_input_weights(self, p=0.1):
        for i, reservoir in enumerate(self.reservoirs):
            if i == 0:
                input_dim = self.input_dim
                n = 1 / input_dim
                Win = np.random.uniform(0.5*n, n, (reservoir.units, input_dim))
                mask = np.random.rand(reservoir.units, input_dim) < p
                Win *= mask
                reservoir.Win = Win
                reservoir.input_dim = input_dim
            else:
                input_dim = reservoir.units
                Win = np.random.uniform(-1, 1, (reservoir.units, input_dim))
                reservoir.Win = Win
                mask = np.random.rand(reservoir.units, input_dim) < p
                Win *= mask
                reservoir.input_dim = input_dim

    def create_tonotopic_mapping(self):
        neuron_positions = np.linspace(0, 1, self.reservoir.units)
        freq_positions = np.linspace(0, 1, self.input_dim)

        n = 1 / self.input_dim

        ### Create input matrix ###
        W_in = np.zeros((self.reservoir.units, self.input_dim))
        for i, pos in enumerate(neuron_positions):
            W_in[i, :] = np.exp(-0.5 * ((freq_positions - pos) / self.input_width) ** 2)
            W_in[i, :] *= np.random.uniform(0.5, 1.0) * n

        # mask = np.random.randn(self.reservoir.units, self.input_dim) < connectivity
        # W_in *= mask  # Apply sparsity mask to input weights

        ### Create reservoir weight matrix ###
        # First create normal sparse random weights
        mask2 = np.random.randn(self.reservoir.units, self.reservoir.units) < self.connectivity
        W = np.random.uniform(-1, 1, (self.reservoir.units, self.reservoir.units)) * mask2

        # Apply a gaussian weighing based on distance
        distance = np.abs(neuron_positions[:, None] - neuron_positions[None, :])
        locality = np.exp(-0.5 * (distance / self.reservoir_width) ** 2)  # Compute locality weighing
        W *= locality  # Apply weighing to matrix

        # Normalize spectral radius
        eigvals = np.linalg.eigvals(W)
        W *= self.sr / np.max(np.abs(eigvals))

        self.reservoir.Win = W_in
        self.reservoir.W = W
        self.reservoir.input_dim = self.input_dim

    def apply_ip(self, p=0.1):
        # Create input matrix
        self.reservoirs[0].Win = np.random.uniform(0.5, 1, (self.reservoirs[0].units, 1))
        mask = np.random.rand(self.reservoirs[0].units, 1) < p
        self.reservoirs[0].Win *= mask

        # Create narma series
        T = 1000
        _, X_narma = narma(T)

        # Apply IP
        for reservoir in self.reservoirs:
            # Train IP on current input
            reservoir.fit(X_narma, warmup=100)

    def train(self, X, Y):
        states = self.forward.run(X, workers=self.workers)
        if self.n_reservoirs > 1:
            states_list = list(states.values())
            states = np.concatenate(states_list, axis=-1)

        self.readout.fit(states, y=Y, workers=1, warmup=5)

    def test(self, X_test, Y_test):
        states = self.forward.run(X_test, workers=self.workers)

        if self.n_reservoirs > 1:
            states_list = list(states.values())
            states = np.concatenate(states_list, axis=-1)

        predictions_list = self.readout.run(states)

        y_pred = []
        timestep_predictions = []
        y_true = []

        for i, (X_seq, y_seq) in enumerate(zip(X_test, Y_test)):
            predictions = predictions_list[i]                       # Get raw prediction per timestep
            pred_per_timestep = np.argmax(predictions, axis=1)              # Get one-hot winner at each timestep
            timestep_predictions.append(pred_per_timestep)

            non_silence_preds = pred_per_timestep[pred_per_timestep != 10]  # Remove silence as category
            final_pred = mode(non_silence_preds, keepdims=False).mode       # Get winning digit with majority voting

            y_pred.append(final_pred)

            # Get true labels for this sequence
            y_per_timestep = np.argmax(y_seq, axis=1)
            non_silence_true = y_per_timestep[y_per_timestep != 10]
            y_true.append(non_silence_true[0])

        # Convert to arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        timestep_predictions = list(timestep_predictions)
        y_per_timestep = np.array([np.argmax(y_seq, axis=1) for y_seq in Y_test])

        # Compute accuracy
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, y_true, y_pred, timestep_predictions, y_per_timestep

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
