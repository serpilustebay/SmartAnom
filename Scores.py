import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef


def get_param_grid_for_model(model_name: str):
    grids = {
        # 🌲 Isolation Forest
        "Isolation Forest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🌳 Extended Isolation Forest
        "Extended Isolation Forest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "level": 1,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🔷 Generalized Isolation Forest
        "Generalized Isolation Forest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "k_planes": 2,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🧩 SciForest
        "SciForest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # ⚖️ FairCutForest
        "FairCutForest": {
            "defaults": {
                "n_trees": 100,
                "sample_size": 256,
                "contamination": 0.1,
                "threshold": 0.95,
                "majority": 5
            },
            "grid": {
                "n_trees": [50, 100],
                "sample_size": [128, 256],
                "contamination": [0.05, 0.1],
                "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # 🧠 One-Class SVM
        "One-Class SVM": {
            "defaults": {
                "kernel": "rbf",
                "nu": 0.05,
                "gamma": "scale"
            },
            "grid": {
                "kernel": ["rbf"],
                "nu": [0.01, 0.05],
                "gamma": ["scale"]
            }
        },

        # 🔍 Local Outlier Factor
        "Local Outlier Factor": {
            "defaults": {
                "n_neighbors": 20,
                "leaf_size": 30,
                "metric": "minkowski",
                "contamination": 0.1
            },
            "grid": {
                "n_neighbors": [10, 20, 35],
                "leaf_size": [30, 50, 70],
                "metric": ["minkowski", "euclidean"],
                "contamination": [0.05, 0.1, 0.15]
            }
        },

        # 📈 Elliptic Envelope
        "Elliptic Envelope": {
            "defaults": {
                "contamination": 0.1,
                "support_fraction": 0.8,
                "assume_centered": False
            },
            "grid": {
                "contamination": [0.05, 0.1, 0.15],
                "support_fraction": [0.7, 0.8, 0.9],
                "assume_centered": [False, True]
            }
        },

        # 🤖 Autoencoder
        "Autoencoder": {
            "defaults": {
                "latent_dim": 8,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 50,
                "dropout": 0.2
            },
            "grid": {
                "latent_dim": [4, 8, 16],
                "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128],
                "epochs": [50, 100],
                "dropout": [0.0, 0.2, 0.5]
            }
        },

        # 🧬 Variational Autoencoder (VAE)
        "VAE": {
            "defaults": {
                "latent_dim": 8,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 50,
                "beta": 1.0
            },
            "grid": {
                "latent_dim": [4, 8, 16],
                "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128],
                "epochs": [50, 100],
                "beta": [0.5, 1.0, 2.0]
            }
        },

        # 🧠 DeepSVDD
        "DeepSVDD": {
            "defaults": {
                "latent_dim": 16,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 50,
                "lambda_reg": 0.5
            },
            "grid": {
                "latent_dim": [8, 16, 32],
                "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128],
                "epochs": [50, 100],
                "lambda_reg": [0.1, 0.5, 1.0]
            }
        },
    }

    return grids.get(model_name, {})


class MBAS(object):
    @staticmethod
    def predict(path_lenghts, threshold=0.6):
        """
        Predict anomalies based on individual anomaly scores.

        Parameters
        ----------
        path_lenghts : array-like of shape (n_samples,)
            The anomaly scores or path lengths computed for each sample.

        threshold : float, default=0.6
            The cutoff value used to classify samples.
            If the score > threshold → sample is labeled as anomaly (1),
            otherwise it is considered normal (0).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Binary prediction array where 1 indicates anomaly and 0 indicates normal.
        """

        # Convert input scores to a NumPy array to ensure proper vectorized operations
        scores = np.asarray(path_lenghts)

        # Apply the threshold rule:
        # Samples with score greater than the threshold are labeled as anomalies (1)
        y_pred = (scores > threshold).astype(int)

        # Return binary anomaly predictions
        return y_pred




class SBAS:

    @staticmethod
    def custom_sigmoid(x, k=2):
        """
        Custom sigmoid transformation function.
        It maps input values into the range [0, 1], with x=0 centered at 0.5.
        The 'k' parameter controls the steepness of the curve.
        """
        # The sigmoid function smoothly maps real numbers to (0, 1)
        return 1 / (1 + np.exp(-k * x))



    @staticmethod
    def predict(path_lengths, threshold, majority):
        """
        Predict anomalies based on sigmoid-transformed path lengths.

        Parameters
        ----------
        path_lengths : ndarray of shape (n_samples, n_trees)
            The path lengths or anomaly scores obtained from the isolation-based trees.

        threshold : float
            Sigmoid threshold. Samples with sigmoid score below this value
            are considered potential anomalies.

        majority : int
            The minimum number of trees that must classify a sample as an anomaly
            for it to be labeled as anomalous.

        Returns
        -------
        pred : ndarray of shape (n_samples,)
            Binary prediction array, where 1 indicates anomaly and 0 indicates normal.
        """

        # Apply sigmoid transformation to the path lengths
        scores = SBAS.custom_sigmoid(path_lengths)

        # Mark samples below the threshold as anomalies (1)
        binary = (scores <= threshold).astype(int)

        # Count how many trees voted each sample as anomalous
        total_votes = np.sum(binary, axis=1)

        # If the number of anomaly votes exceeds 'majority', label as anomaly (1)
        pred = (total_votes > majority).astype(int)

        return pred




def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "Accuracy": acc,
        "MCC": mcc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Specificity": specificity,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,

    }

    metrics = {k: round(v, 3) for k, v in metrics.items()}

    return metrics

