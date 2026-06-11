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
from datetime import datetime
from ShallowESN import ShallowNetwork

matplotlib.use("tKagg")

class NewDeepNetwork:
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
        self.models=[]

        N = int(round(self.N_total / n_reservoirs))

        for i in range(n_reservoirs):
            if i == 0:
                n_input_dim = self.input_dim
            else:
                n_input_dim = N

            model = ShallowNetwork(N, self.sr, self.lr, self.sigma, self.ridge, n_input_dim, self.input_width, self.IP)

            self.models.append()

        self.readout = Ridge(ridge=self.ridge, output_dim=11, name="readout")

    def rescale_reservoirs(self):
        N = int(round(self.N_total / self.n_reservoirs))

        W = self.reservoir.W.copy()

        for i in range(self.n_reservoirs):
            start = i * N
            end = (i + 1) * N

            Wl = self.reservoir.W[start:end, start:end]
            I = np.eye(N, dtype=np.float32)

            W_t = (1 - self.lr) * I + self.lr * Wl
            W_t = self.sr * W_t / max(abs(eigvals(W_t)))
            Wl = ((W_t - (1 - self.lr) * I) / self.lr).astype(np.float32)

            W[start:end, start:end] = Wl

        self.reservoir.W = W.astype(np.float32)


    def create_input_weights(self, p=0.1):
        for i, model in enumerate(self.models):
            if i == 0:
                input_dim = model.input_dim
                n = 1 / input_dim
                Win = np.random.uniform(0.5*n, n, (model.N, input_dim))
                mask = np.random.rand(model.N, input_dim) < p
                Win *= mask
                model.reservoir.Win = Win
                model.reservoir.input_dim = input_dim
            else:
                input_dim = model.N
                Win = np.random.uniform(-1, 1, (model.N, input_dim)).astype(np.float32)
                model.reservoir.Win = Win
                model.reservoir.input_dim = input_dim

    def create_tonotopic_mapping(self):
        neuron_positions = np.linspace(0, 1, self.reservoir.units, dtype=np.float32)
        freq_positions = np.linspace(0, 1, self.input_dim, dtype=np.float32)

        n = 1 / self.input_dim

        ### Create input matrix ###
        W_in = np.zeros((self.reservoir.units, self.input_dim)).astype(np.float32)
        for i, pos in enumerate(neuron_positions):
            W_in[i, :] = np.exp(-0.5 * ((freq_positions - pos) / self.input_width) ** 2).astype(np.float32)
            W_in[i, :] *= np.random.uniform(0.5, 1.0).astype(np.float32) * n

        # mask = np.random.randn(self.reservoir.units, self.input_dim) < connectivity
        # W_in *= mask  # Apply sparsity mask to input weights

        ### Create reservoir weight matrix ###
        # First create normal sparse random weights
        mask2 = np.random.randn(self.reservoir.units, self.reservoir.units).astype(np.float32) < self.connectivity
        W = np.random.uniform(-1, 1, (self.reservoir.units, self.reservoir.units)).astype(np.float32) * mask2

        # Apply a gaussian weighing based on distance
        distance = np.abs(neuron_positions[:, None] - neuron_positions[None, :]).astype(np.float32)
        locality = np.exp(-0.5 * (distance / self.reservoir_width) ** 2).astype(np.float32) # Compute locality weighing
        W *= locality  # Apply weighing to matrix

        # # Normalize spectral radius
        # eigvals = np.linalg.eigvals(W).astype(np.float32)
        # W *= self.sr / np.max(np.abs(eigvals)).astype(np.float32)

        self.reservoir.Win = W_in
        self.reservoir.W = W
        self.reservoir.input_dim = self.input_dim

    def apply_ip(self, p=0.1):
        # Create input matrix
        self.models[0].reservoir.Win = np.random.uniform(0.5, 1, (self.models[0].N, 1))
        mask = np.random.rand(self.reservoirs[0].units, 1) < p
        self.models[0].reservoir.Win *= mask

        # Create narma series
        T = 1000
        _, X_narma = narma(T)
        X_narma = np.asarray(X_narma)

        # Apply IP
        for model in self.models:
            # Train IP on current input
            model.reservoir.fit(X_narma, warmup=100)

    def train(self, X, Y):
        # Run spectrograms to reservoir, and fit readout layer on reservoir states
        states_list = self.reservoir.run(X, workers=self.workers)
        self.readout.fit(states_list, y=Y, workers=1, warmup=3)

    def log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def test(self, X, Y):
        # Run spectrograms through reservoir
        states_list = self.reservoir.run(X, workers=self.workers)

        y_pred = []
        timestep_predictions = []
        y_true = []

        for X_seq, states, y_seq in zip(X, states_list, Y):
            predictions = self.readout.run(states)  # Get raw prediction per timestep
            pred_per_timestep = np.argmax(predictions, axis=1)  # Get one-hot winner at each timestep
            timestep_predictions.append(pred_per_timestep)

            non_silence_preds = pred_per_timestep[pred_per_timestep != 10]  # Remove silence as category
            final_pred = mode(non_silence_preds, keepdims=False).mode  # Get winning digit with majority voting

            y_pred.append(final_pred)

            # Get true labels for this sequence
            y_per_timestep = np.argmax(y_seq, axis=1)
            non_silence_true = y_per_timestep[y_per_timestep != 10]
            y_true.append(non_silence_true[0])

        # Convert to arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        timestep_predictions = list(timestep_predictions)
        y_per_timestep = np.array([np.argmax(y_seq, axis=1) for y_seq in Y])

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
