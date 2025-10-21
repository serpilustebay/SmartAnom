import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import numpy as np

from IFModels import eif, gif
from IFModels.FairCutForest import FairCutForest
from IFModels.SciForest import SCiForest


class DeepModels:
    """
    Deep learning-based anomaly detection models:
    Autoencoder, Variational Autoencoder (VAE), and DeepSVDD.
    """

    @staticmethod
    def run_autoencoder(X, y_true=None, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train and evaluate a simple Autoencoder model for anomaly detection.

        Args:
            X (ndarray): Input feature matrix.
            y_true (ndarray, optional): True anomaly labels for adaptive thresholding.
            epochs (int): Training epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate for Adam optimizer.

        Returns:
            tuple: (y_pred, trained_autoencoder)
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(16, activation='relu')(input_layer)
        encoded = layers.Dense(8, activation='relu')(encoded)
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = models.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss='mse')

        autoencoder.fit(X_scaled, X_scaled,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)

        X_pred = autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)

        if y_true is not None and np.any(y_true == 1):
            n_anomalies = np.sum(y_true == 1)
            threshold = np.sort(mse)[-n_anomalies]
        else:
            threshold = np.percentile(mse, 95)

        y_pred = (mse > threshold).astype(int)
        return y_pred, autoencoder

    @staticmethod
    def run_vae(X, y_true=None, epochs=15, batch_size=8, learning_rate=0.001, latent_dim=2):
        """
        Train a Variational Autoencoder (VAE) model for anomaly detection.

        Features:
            - Deterministic sampling for SHAP compatibility.
            - Reconstruction error used as anomaly score.
            - Stable behavior on small datasets.

        Args:
            X (ndarray): Input feature matrix.
            y_true (ndarray, optional): True labels for threshold adjustment.
            epochs (int): Training epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            latent_dim (int): Latent dimension size.

        Returns:
            tuple: (y_pred, trained_vae)
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float32))
        input_dim = X_scaled.shape[1]

        inputs = layers.Input(shape=(input_dim,), name="vae_input")
        h = layers.Dense(16, activation='relu')(inputs)
        h = layers.Dense(8, activation='relu')(h)
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)

        def deterministic_sampling(args):
            z_mean_tensor, z_log_var_tensor = args
            return z_mean_tensor

        z = layers.Lambda(deterministic_sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

        decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
        h_dec = layers.Dense(8, activation='relu')(decoder_input)
        h_dec = layers.Dense(16, activation='relu')(h_dec)
        decoder_output = layers.Dense(input_dim, activation='linear')(h_dec)
        decoder = models.Model(decoder_input, decoder_output, name="decoder")

        outputs = decoder(z)
        vae = models.Model(inputs, outputs, name="vae")

        reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        vae.add_loss(total_loss)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        vae.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

        X_pred = vae.predict(X_scaled, batch_size=batch_size, verbose=0)
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)

        threshold = np.percentile(mse, 95)
        if y_true is not None and np.any(y_true == 1):
            n_anom = np.sum(y_true == 1)
            threshold = np.sort(mse)[-n_anom] if n_anom < len(mse) else threshold

        y_pred = (mse > threshold).astype(int)
        vae.scaler = scaler
        vae.decoder = decoder
        vae.latent_dim = latent_dim

        return y_pred, vae

    @staticmethod
    def run_deepsvdd(X, y_true=None, params=None):
        """
        Train a DeepSVDD model using distance-based anomaly detection.

        Args:
            X (ndarray): Input features.
            y_true (ndarray, optional): True labels.
            params (dict): Hyperparameters such as epochs, learning_rate, hidden_units.

        Returns:
            tuple: (y_pred, trained_model)
        """
        if params is None:
            params = {}
        epochs = params.get("epochs", 50)
        learning_rate = params.get("learning_rate", 0.001)
        hidden_units = params.get("hidden_units", [32, 16, 8])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float32))
        input_dim = X_scaled.shape[1]

        inputs = layers.Input(shape=(input_dim,), name="input")
        h = inputs
        for units in hidden_units:
            h = layers.Dense(units, activation='relu')(h)
        z = layers.Dense(hidden_units[-1], activation=None, name="embedding")(h)
        model = models.Model(inputs, z, name="DeepSVDD")

        c = np.mean(model.predict(X_scaled, verbose=0), axis=0)
        model.c = c

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                z_out = model(X_scaled, training=True)
                dist = tf.reduce_sum((z_out - c) ** 2, axis=1)
                loss = tf.reduce_mean(dist)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        z_out = model.predict(X_scaled, verbose=0)
        dist = np.sum((z_out - c) ** 2, axis=1)

        threshold = np.percentile(dist, 95)
        if y_true is not None and np.any(y_true == 1):
            n_anomalies = np.sum(y_true == 1)
            threshold = np.sort(dist)[-n_anomalies] if n_anomalies > 0 else threshold

        y_pred = (dist > threshold).astype(int)
        model.scaler = scaler
        return y_pred, model


class IFModel:
    """
    Unified interface for Isolation Forest variants (IF, EIF, GIF, SciForest, FairCutForest).
    """

    def __init__(self, model_type, score_method,
                 n_tree, sample_size, contamination, level, k_planes, majority, threshold):
        """Initialize the IF model configuration."""
        self.model_type = model_type
        self.score_method = score_method
        self.n_tree = n_tree
        self.phi = sample_size
        self.contamination = contamination
        self.level = level
        self.k_planes = k_planes
        self.majority = majority
        self.threshold = threshold

    def _predict_score(self, clf, X, fit_needed=False):
        """Compute anomaly scores using MBAS or SBAS scoring."""
        if fit_needed:
            clf.fit(X)
        if self.score_method == "MBAS":
            scores = clf.predict(X)
            return MBAS.predict(scores)
        elif self.score_method == "SBAS":
            path_len = clf.compute_paths_all_tree(X)
            return SBAS.predict(path_len, threshold=self.threshold, majority=self.majority)

    def evaluate_IF(self, X):
        """Evaluate classic Isolation Forest."""
        clf = eif.iForest(X=X, ntrees=self.n_tree, sample_size=self.phi, ExtensionLevel=0)
        return self._predict_score(clf, X)

    def evaluate_EIF(self, X):
        """Evaluate Extended Isolation Forest."""
        clf = eif.iForest(X=X, ntrees=self.n_tree, sample_size=self.phi, ExtensionLevel=1)
        return self._predict_score(clf, X)

    def evaluate_GIF(self, X):
        """Evaluate Generalized Isolation Forest."""
        clf = gif.iForest(X=X, ntrees=self.n_tree, sample_size=self.phi)
        return self._predict_score(clf, X)

    def evaluate_SciForest(self, X):
        """Evaluate SciForest."""
        clf = SCiForest(n_trees=self.n_tree, sample_size=self.phi,
                        k_planes=self.k_planes, extension_level=self.level)
        return self._predict_score(clf, X, fit_needed=True)

    def evaluate_FairCutForest(self, X):
        """Evaluate FairCutForest."""
        clf = FairCutForest(n_trees=self.n_tree, sample_size=self.phi,
                            k_planes=self.k_planes, extension_level=self.level)
        return self._predict_score(clf, X, fit_needed=True)

    def evaluate(self, X, y_true):
        """
        Evaluate selected Isolation Forest variant and compute metrics.

        Returns:
            dict: Performance metrics (accuracy, F1, etc.)
        """
        model_map = {
            "Isolation Forest": self.evaluate_IF,
            "Extended Isolation Forest": self.evaluate_EIF,
            "Generalized Isolation Forest": self.evaluate_GIF,
            "SciForest": self.evaluate_SciForest,
            "FairCutForest": self.evaluate_FairCutForest
        }
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")

        y_pred = model_map[self.model_type](X)
        metrics = compute_metrics(y_true, y_pred)
        return metrics


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from Scores import compute_metrics, MBAS, SBAS
from Models import DeepModels


class BenchmarkModels:
    """
    Benchmark class for classical and deep anomaly detection models.
    """

    @staticmethod
    def run(X, y_true, selected_model, hyperparams):
        """
        Train and evaluate a selected benchmark model.

        Args:
            X (ndarray): Input feature matrix.
            y_true (ndarray): True anomaly labels.
            selected_model (str): Model name.
            hyperparams (dict): Model hyperparameters.

        Returns:
            tuple: (metrics_dict, trained_model)
        """
        y_pred = None
        model_instance = None
        params = hyperparams

        if selected_model == "Sklearn IF":
            model_instance = IsolationForest(
                n_estimators=params.get("n_trees", 100),
                contamination=params.get("contamination", 0.05),
                max_samples=params.get("max_samples", "auto"),
                random_state=42
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        elif selected_model == "One-Class SVM":
            model_instance = OneClassSVM(
                kernel=params.get("kernel", "rbf"),
                nu=params.get("nu", 0.05),
                gamma=params.get("gamma", "scale")
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        elif selected_model == "Local Outlier Factor":
            model_instance = LocalOutlierFactor(
                n_neighbors=params.get("n_neighbors", 20),
                contamination=params.get("contamination", 0.05),
                novelty=True
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        elif selected_model == "Elliptic Envelope":
            model_instance = EllipticEnvelope(
                contamination=params.get("contamination", 0.05),
                random_state=42
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        elif selected_model == "Autoencoder":
            y_pred, model_instance = DeepModels.run_autoencoder(
                X, y_true=y_true,
                epochs=params.get("epochs", 50),
                batch_size=params.get("batch_size", 32),
                learning_rate=params.get("learning_rate", 0.001)
            )

        elif selected_model == "VAE":
            y_pred, model_instance = DeepModels.run_vae(
                X, y_true=y_true,
                epochs=params.get("epochs", 50),
                batch_size=params.get("batch_size", 32),
                learning_rate=params.get("learning_rate", 0.001),
                latent_dim=params.get("latent_dim", 4)
            )

        elif selected_model == "DeepSVDD":
            y_pred, model_instance = DeepModels.run_deepsvdd(X, y_true, params=params)

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

        metrics = compute_metrics(y_true, y_pred)
        return metrics, model_instance
