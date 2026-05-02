import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from reservoirpy.datasets import narma
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
from utils import *

class ShallowNetwork:
    def __init__(self, N, sr, lr, sigma, ridge, input_dim, input_width, IP):
        """
        Initialize a shallow reservoir computing network.

        Parameters
        ----------
        N : int
            Number of neurons in the reservoir.
        sr : float
            Spectral radius of the reservoir weight matrix.
        lr : float
            Leaky integration rate of the reservoir neurons.
        sigma : float
            Standard deviation of the target distribution used for IP
        ridge : float
            Regularization coefficient for the ridge regression readout.
        input_dim : int
            Dimensionality of the input features.
        input_width : float
            Width of the frequency neighborhood for tonotopic input mapping.
        IP : bool, optional
            Whether to use intrinsic plasticity.
        """

        self.N = N
        self.sr = sr
        self.lr = lr
        self.sigma = sigma
        self.ridge = ridge
        self.input_dim = input_dim
        self.input_width = input_width
        self.IP = IP
        self.workers = 1  # Amount of workers used for parallelism

        # Create reservoir
        if self.IP:
            self.reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=sigma, activation="tanh", epochs=4,
                                         learning_rate=3e-4, dtype=np.float32)
        else:
            self.reservoir = Reservoir(N, sr=sr, lr=lr, dtype=np.float32)

        # Create readout
        self.readout = Ridge(ridge=ridge, output_dim=11)

    def create_input_weights(self, p=0.1):
        """
        Create the input weight matrix.

        Parameters
        ----------
        p : float
            Connectivity probability
        """
        # Scale input matrix with amount of features per timestep, for applying IP to work
        n = 1 / self.input_dim
        Win = np.random.uniform(0.5 * n, n, (self.reservoir.units, self.input_dim))

        # Create and apply sparsity mask
        mask = np.random.rand(self.reservoir.units, self.input_dim) < p
        Win *= mask

        ### is it actually worste without the scaling :o####
        
        # Set weight matrix and input dimension
        self.reservoir.Win = Win
        self.reservoir.input_dim = self.input_dim

    def create_tonotopic_mapping(self):
        """
        Create tonotopic mapping
        """
        # Creates axes to align neurons with sensitive frequencies
        neuron_positions = np.linspace(0, 1, self.reservoir.units)
        freq_positions = np.linspace(0, 1, self.input_dim)

        n = 1 / self.input_dim

        # === Create input matrix == #
        W_in = np.zeros((self.reservoir.units, self.input_dim))
        for i, pos in enumerate(neuron_positions):
            W_in[i, :] = np.exp(-0.5 * ((freq_positions - pos) / self.input_width) ** 2)
            W_in[i, :] *= np.random.uniform(0.5, 1.0) * n

        # eigvals = np.linalg.eigvals(W)
        # W *= self.sr / np.max(np.abs(eigvals))

        self.reservoir.Win = W_in
        self.reservoir.input_dim = self.input_dim

    def apply_ip(self, p=0.1):
        """
        Apply intrinsic plasticity (IP) to the reservoir.

        Parameters
        ----------
        p : float, optional
            Connectivity probability, by default 0.1.

        Returns
        -------
        a_list : list
            Values of parameter `a` over time.
        b_list : list
            Values of parameter `b` over time.
        """
        # Create input matrix for IP to be consistent with input mapping of the model
        self.reservoir.Win = np.random.uniform(0.5, 1, (self.reservoir.units, 1))
        mask = np.random.rand(self.reservoir.units, 1) < p
        self.reservoir.Win *= mask

        # Create narma series
        T = 1000
        _, X_narma = narma(T)

        # Apply IP
        a_list = []
        b_list = []

        for epoch in range(4):
            for u in X_narma[100:]:
                post_state = self.reservoir.step(u)
                pre_state = self.reservoir.state["internal"]

                delta_a, delta_b = self.reservoir.gradient(x=pre_state.T, y=post_state.T, a=self.reservoir.a)
                self.reservoir.a += self.reservoir.learning_rate * delta_a
                self.reservoir.b += self.reservoir.learning_rate * delta_b

                a_list.append(self.reservoir.a)
                b_list.append(self.reservoir.b)

        return a_list, b_list

    def train(self, X, Y):
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_timesteps, n_features)
            Input features from the training set
        Y : np.ndarray (n_samples, n_timesteps, n_features)
            Corresponding labels for the training set.

        Returns
        -------
        kl : float
            Kullback–Leibler divergence of reservoir states.
        ent : float
            Entropy.
        """
        # Run spectrograms to reservoir, and fit readout layer on reservoir states
        states_list = self.reservoir.run(X, workers=self.workers)
        self.readout.fit(states_list, y=Y, workers=1, warmup=3)

        # Calculate kl-divergence and entropy
        if self.IP:
            kl, ent = get_KL_divergence_and_entropy(states_list, self.sigma)
            return kl, ent


    def test(self, X, Y):
        """
        Evaluate the model on a test dataset.

        Parameters
        ----------
        X : (n_samples, n_timesteps, n_features)
            Input features from the test set.
        Y : (n_samples, n_timesteps, n_features)
            Corresponding labels for the test set.

        Returns
        -------
        accuracy : float
            Overall accuracy on the test set.
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.
        timestep_predictions : array-like
            Model predictions at each timestep.
        y_per_timestep : array-like
            Ground truth labels at each timestep.
        """

        # Run spectrograms through reservoir
        states_list = self.reservoir.run(X, workers=self.workers)

        y_pred = []
        timestep_predictions = []
        y_true = []

        for X_seq, states, y_seq in zip(X, states_list, Y):
            predictions = self.readout.run(states)                          # Get raw prediction per timestep
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
        y_per_timestep = np.array([np.argmax(y_seq, axis=1) for y_seq in Y])

        # Compute accuracy
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, y_true, y_pred, timestep_predictions, y_per_timestep

    def save(self, path):
        """
        Save the model to disk.

        Parameters
        ----------
        path : str
            File path where the model will be saved.

        Returns
        -------
        None
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load a model from disk.

        Parameters
        ----------
        path : str
            File path from which the model will be loaded.

        Returns
        -------
        None
        """
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
