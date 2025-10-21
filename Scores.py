"""
Scores.py
----------
Defines metric computation, scoring methods (MBAS, SBAS),
and hyperparameter grids for SmartAnom anomaly detection framework.

"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef
)


def get_param_grid_for_model(model_name: str):
    """
    Retrieve default and grid search parameter sets for each SmartAnom model.

    Args:
        model_name (str): Model name key (e.g., "Isolation Forest", "VAE").

    Returns:
        dict: Contains 'defaults' and 'grid' dictionaries for the model.
    """
    grids = {
        # Isolation Forest
        "Isolation Forest": {
            "defaults": {
                "n_trees": 100, "sample_size": 256, "contamination": 0.1,
                "threshold": 0.95, "majority": 5
            },
            "grid": {
                "n_trees": [50, 100], "sample_size": [128, 256],
                "contamination": [0.05, 0.1], "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # Extended Isolation Forest
        "Extended Isolation Forest": {
            "defaults": {
                "n_trees": 100, "sample_size": 256, "level": 1,
                "contamination": 0.1, "threshold": 0.95, "majority": 5
            },
            "grid": {
                "n_trees": [50, 100], "sample_size": [128, 256],
                "contamination": [0.05, 0.1], "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # Generalized Isolation Forest
        "Generalized Isolation Forest": {
            "defaults": {
                "n_trees": 100, "sample_size": 256, "k_planes": 2,
                "contamination": 0.1, "threshold": 0.95, "majority": 5
            },
            "grid": {
                "n_trees": [50, 100], "sample_size": [128, 256],
                "contamination": [0.05, 0.1], "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # SciForest
        "SciForest": {
            "defaults": {
                "n_trees": 100, "sample_size": 256,
                "contamination": 0.1, "threshold": 0.95, "majority": 5
            },
            "grid": {
                "n_trees": [50, 100], "sample_size": [128, 256],
                "contamination": [0.05, 0.1], "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # FairCutForest
        "FairCutForest": {
            "defaults": {
                "n_trees": 100, "sample_size": 256,
                "contamination": 0.1, "threshold": 0.95, "majority": 5
            },
            "grid": {
                "n_trees": [50, 100], "sample_size": [128, 256],
                "contamination": [0.05, 0.1], "threshold": [0.95, 0.98],
                "majority": [7, 10]
            }
        },

        # One-Class SVM
        "One-Class SVM": {
            "defaults": {"kernel": "rbf", "nu": 0.05, "gamma": "scale"},
            "grid": {"kernel": ["rbf"], "nu": [0.01, 0.05], "gamma": ["scale"]}
        },

        # Local Outlier Factor
        "Local Outlier Factor": {
            "defaults": {
                "n_neighbors": 20, "leaf_size": 30, "metric": "minkowski",
                "contamination": 0.1
            },
            "grid": {
                "n_neighbors": [10, 20, 35], "leaf_size": [30, 50, 70],
                "metric": ["minkowski", "euclidean"],
                "contamination": [0.05, 0.1, 0.15]
            }
        },

        # Elliptic Envelope
        "Elliptic Envelope": {
            "defaults": {
                "contamination": 0.1, "support_fraction": 0.8,
                "assume_centered": False
            },
            "grid": {
                "contamination": [0.05, 0.1, 0.15],
                "support_fraction": [0.7, 0.8, 0.9],
                "assume_centered": [False, True]
            }
        },

        # Autoencoder
        "Autoencoder": {
            "defaults": {
                "latent_dim": 8, "learning_rate": 0.001,
                "batch_size": 64, "epochs": 50, "dropout": 0.2
            },
            "grid": {
                "latent_dim": [4, 8, 16], "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128], "epochs": [50, 100],
                "dropout": [0.0, 0.2, 0.5]
            }
        },

        # Variational Autoencoder
        "VAE": {
            "defaults": {
                "latent_dim": 8, "learning_rate": 0.001,
                "batch_size": 64, "epochs": 50, "beta": 1.0
            },
            "grid": {
                "latent_dim": [4, 8, 16], "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128], "epochs": [50, 100],
                "beta": [0.5, 1.0, 2.0]
            }
        },

        # DeepSVDD
        "DeepSVDD": {
            "defaults": {
                "latent_dim": 16, "learning_rate": 0.001,
                "batch_size": 64, "epochs": 50, "lambda_reg": 0.5
            },
            "grid": {
                "latent_dim": [8, 16, 32], "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64, 128], "epochs": [50, 100],
                "lambda_reg": [0.1, 0.5, 1.0]
            }
        },
    }

    return grids.get(model_name, {})


class MBAS:
    """
    Mean-Based Anomaly Scoring (MBAS):
    Predicts anomalies using a static threshold over raw scores.
    """

    @staticmethod
    def predict(path_lenghts, threshold=0.6):
        """
        Predict anomalies based on individual scores.

        Args:
            path_lenghts (array-like): Anomaly scores or path lengths per sample.
            threshold (float): Cutoff threshold for anomaly classification.

        Returns:
            ndarray: Binary predictions (1 = anomaly, 0 = normal).
        """
        scores = np.asarray(path_lenghts)
        return (scores > threshold).astype(int)


class SBAS:
    """
    Sigmoid-Based Anomaly Scoring (SBAS):
    Applies a sigmoid transformation and majority voting to detect anomalies.
    """

    @staticmethod
    def custom_sigmoid(x, k=2):
        """
        Sigmoid transformation mapping real values to [0, 1].

        Args:
            x (ndarray): Input path lengths.
            k (float): Curve steepness factor.

        Returns:
            ndarray: Sigmoid-transformed values.
        """
        return 1 / (1 + np.exp(-k * x))

    @staticmethod
    def predict(path_lengths, threshold, majority):
        """
        Predict anomalies using sigmoid transformation and majority voting.

        Args:
            path_lengths (ndarray): Path lengths (n_samples, n_trees).
            threshold (float): Sigmoid cutoff threshold.
            majority (int): Minimum anomaly votes required to mark as anomaly.

        Returns:
            ndarray: Binary anomaly predictions.
        """
        scores = SBAS.custom_sigmoid(path_lengths)
        binary = (scores <= threshold).astype(int)
        total_votes = np.sum(binary, axis=1)
        return (total_votes > majority).astype(int)


def compute_metrics(y_true, y_pred):
    """
    Compute standard performance metrics for anomaly detection.

    Args:
        y_true (ndarray): Ground truth labels (0/1).
        y_pred (ndarray): Predicted labels (0/1).

    Returns:
        dict: Dictionary containing Accuracy, MCC, F1, Precision, Recall, Specificity, FPR, and FNR.
    """
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

    return {k: round(v, 3) for k, v in metrics.items()}
