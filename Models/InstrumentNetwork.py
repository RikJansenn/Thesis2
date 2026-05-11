import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from reservoirpy.datasets import narma
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
from utils import *

class InstrumentNetwork:
    def __init__(self, N, sr, lr, sigma, ridge, input_dim, input_width, IP):
        self.N = N
        self.sr = sr
        self.lr = lr
        self.sigma = sigma
        self.ridge = ridge
        self.input_dim = input_dim
        self.input_width = input_width
        self.IP = IP
        self.workers = 1

        self.num_classes = 12
        self.silence_index = 11

        self.instrument_to_label = {
            "bass": 0,
            "brass": 1,
            "flute": 2,
            "guitar": 3,
            "keyboard": 4,
            "mallet": 5,
            "organ": 6,
            "reed": 7,
            "string": 8,
            "synth_lead": 9,
            "vocal": 10,
        }

        self.label_to_instrument = {
            v: k for k, v in self.instrument_to_label.items()
        }
        self.label_to_instrument[self.silence_index] = "silence"

        if self.IP:
            self.reservoir = IPReservoir(
                N,
                sr=sr,
                lr=lr,
                mu=0.0,
                sigma=sigma,
                activation="tanh",
                epochs=4,
                learning_rate=3e-4,
                dtype=np.float32,
            )
        else:
            self.reservoir = Reservoir(
                N,
                sr=sr,
                lr=lr,
                dtype=np.float32,
            )

        self.readout = Ridge(
            ridge=ridge,
            output_dim=self.num_classes,
        )

    def create_input_weights(self, p=0.1):
        n = 1 / self.input_dim

        Win = np.random.uniform(
            0.5 * n,
            n,
            (self.reservoir.units, self.input_dim),
        )

        mask = np.random.rand(
            self.reservoir.units,
            self.input_dim,
        ) < p

        Win *= mask

        self.reservoir.Win = Win
        self.reservoir.input_dim = self.input_dim

    def create_tonotopic_mapping(self):
        neuron_positions = np.linspace(0, 1, self.reservoir.units)
        freq_positions = np.linspace(0, 1, self.input_dim)

        n = 1 / self.input_dim

        W_in = np.zeros((self.reservoir.units, self.input_dim))

        for i, pos in enumerate(neuron_positions):
            W_in[i, :] = np.exp(
                -0.5 * ((freq_positions - pos) / self.input_width) ** 2
            )
            W_in[i, :] *= np.random.uniform(0.5, 1.0) * n

        self.reservoir.Win = W_in
        self.reservoir.input_dim = self.input_dim

    def apply_ip(self, p=0.1):
        self.reservoir.Win = np.random.uniform(
            0.5,
            1,
            (self.reservoir.units, 1),
        )

        mask = np.random.rand(self.reservoir.units, 1) < p
        self.reservoir.Win *= mask

        T = 1000
        _, X_narma = narma(T)

        a_list = []
        b_list = []

        for epoch in range(4):
            for u in X_narma[100:]:
                post_state = self.reservoir.step(u)
                pre_state = self.reservoir.state["internal"]

                delta_a, delta_b = self.reservoir.gradient(
                    x=pre_state.T,
                    y=post_state.T,
                    a=self.reservoir.a,
                )

                self.reservoir.a += self.reservoir.learning_rate * delta_a
                self.reservoir.b += self.reservoir.learning_rate * delta_b

                a_list.append(self.reservoir.a.copy())
                b_list.append(self.reservoir.b.copy())

        return a_list, b_list

    def train(self, X, Y):
        states_list = self.reservoir.run(X, workers=self.workers)

        self.readout.fit(
            states_list,
            y=Y,
            workers=1,
            warmup=3,
        )

        if self.IP:
            kl, ent = get_KL_divergence_and_entropy(
                states_list,
                self.sigma,
            )
            return kl, ent

        return None

    def test(self, X, Y):
        states_list = self.reservoir.run(X, workers=self.workers)

        y_pred = []
        y_true = []
        timestep_predictions = []

        for states, y_seq in zip(states_list, Y):
            predictions = self.readout.run(states)

            pred_per_timestep = np.argmax(predictions, axis=1)
            timestep_predictions.append(pred_per_timestep)

            non_silence_preds = pred_per_timestep[
                pred_per_timestep != self.silence_index
            ]

            if len(non_silence_preds) == 0:
                final_pred = self.silence_index
            else:
                final_pred = mode(
                    non_silence_preds,
                    keepdims=False,
                ).mode

            y_pred.append(final_pred)

            y_per_timestep = np.argmax(y_seq, axis=1)

            non_silence_true = y_per_timestep[
                y_per_timestep != self.silence_index
            ]

            if len(non_silence_true) == 0:
                true_label = self.silence_index
            else:
                true_label = non_silence_true[0]

            y_true.append(true_label)

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        timestep_predictions = list(timestep_predictions)

        y_per_timestep = np.array([
            np.argmax(y_seq, axis=1) for y_seq in Y
        ])

        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, y_true, y_pred, timestep_predictions, y_per_timestep

    def predict_labels(self, X):
        states_list = self.reservoir.run(X, workers=self.workers)

        predictions = []

        for states in states_list:
            raw_predictions = self.readout.run(states)
            pred_per_timestep = np.argmax(raw_predictions, axis=1)

            non_silence_preds = pred_per_timestep[
                pred_per_timestep != self.silence_index
            ]

            if len(non_silence_preds) == 0:
                final_pred = self.silence_index
            else:
                final_pred = mode(
                    non_silence_preds,
                    keepdims=False,
                ).mode

            predictions.append(final_pred)

        return np.array(predictions)

    def predict_instruments(self, X):
        labels = self.predict_labels(X)

        return [
            self.label_to_instrument[label]
            for label in labels
        ]

    def save(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)