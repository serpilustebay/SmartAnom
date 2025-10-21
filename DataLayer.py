import pandas as pd
from tkinter import messagebox
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs

# =============================================
# 🔹 Define global parameters
# =============================================
n_points = 800
seed = 343

# =============================================
# 🔹 Define anomaly points for each dataset type
# These will be appended to synthetic datasets to simulate anomalies (outliers)
# =============================================

anomalies_moons = np.array([
    [3.5, -6], [-3, 6],
    [5, 4], [-4, -4],
    [0, 6], [0, -6],
    [0, 0], [-5, -5]
])

anomalies_circles = np.array([
    [0, 0],                 # inside the inner circle
    [6.5, 0], [-6.5, 0],
    [0, 6.5], [0, -6.5],
    [4.5, 4.5], [-4.5, -4.5],
    [7, 3]
])

anomalies_blobs = np.array([
    [7, 7.5], [0, 0],
    [0, 8], [0, -8],
    [8, -2], [-8, 2],
    [5, -7], [-5, 7]
])

anomalies_spiral = np.array([
    [6, 5], [-6, -5],
    [7, -3], [-7, 3],
    [3, -7], [-3, 7],
    [0, 0], [0, -7]
])

anomalies_sin = np.array([
    [0, 8], [1, -8],
    [2, 7], [-2, -7],
    [0, 0], [-4, 8],
    [6, 6], [-6, -7]
])

anomalies_helix = np.array([
    [0, 8], [0, 0],
    [3, 7], [-3, -7],
    [6, -6], [-6, 7],
    [5, 8], [-5, -8]
])

# =============================================
# 🔹 Utility function to add anomalies
# =============================================
def append_anomalies(X, anomalies):
    """
    Append synthetic anomaly points to the dataset.

    Parameters:
        X : ndarray
            Original dataset (normal samples)
        anomalies : ndarray
            Points to be added as anomalies

    Returns:
        X_new : ndarray
            Dataset containing both normal and anomalous samples
        y_new : ndarray
            Label array (0 = normal, 1 = anomaly)
    """
    X_new = np.vstack([X, anomalies])
    y_new = np.hstack([np.zeros(len(X)), np.ones(len(anomalies))])
    return X_new, y_new


# =============================================
# 🔹 Spiral dataset generator
# =============================================
def generate_spiral(n_points=n_points, noise=0.1, revolutions=6):
    """
    Generate a 2D spiral dataset with noise.

    Parameters:
        n_points : int
            Number of data points
        noise : float
            Standard deviation of Gaussian noise
        revolutions : int
            Number of spiral turns

    Returns:
        ndarray of shape (n_points, 2)
    """
    theta = np.linspace(0, 2 * np.pi * revolutions, n_points)
    scale = 0.5
    r = theta * scale

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Add Gaussian noise
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)

    # Normalize values between -10 and 10
    def scale_to_range(arr, new_min, new_max):
        arr_min, arr_max = arr.min(), arr.max()
        return (arr - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min

    x_scaled = scale_to_range(x, -10, 10)
    y_scaled = scale_to_range(y, -10, 10)

    return np.stack([x_scaled, y_scaled], axis=1)


# =============================================
# 🔹 Dataset Generators
# =============================================

def createMoons():
    """Generate 'moons' dataset and append anomaly points."""
    X_moons, _ = make_moons(n_samples=n_points, noise=0.15, random_state=seed)
    X_moons = (X_moons * 5) - 2  # scale and shift
    X_moons_all, y_moons_all = append_anomalies(X_moons, anomalies_moons)
    return X_moons_all, y_moons_all


def createCircles():
    """Generate 'circles' dataset and append anomaly points."""
    X_circles, _ = make_circles(n_samples=n_points, noise=0.05, factor=0.5, random_state=seed)
    X_circles_scaled = X_circles * 8
    X_circles_all, y_circles_all = append_anomalies(X_circles_scaled, anomalies_circles)
    return X_circles_all, y_circles_all


def createBlobs():
    """Generate multi-cluster (blobs) dataset and append anomaly points."""
    centers = np.array([[-6, 4], [6, 6], [0, -4]])
    X_blobs, _ = make_blobs(n_samples=n_points, centers=centers, cluster_std=1.0, random_state=seed)
    X_blobs_all, y_blobs_all = append_anomalies(X_blobs, anomalies_blobs)
    return X_blobs_all, y_blobs_all


def createSpiral():
    """Generate spiral dataset and append anomaly points."""
    X_spiral = generate_spiral()
    X_spiral_all, y_spiral_all = append_anomalies(X_spiral, anomalies_spiral)
    return X_spiral_all, y_spiral_all


def createSinusoidalData(num_points=n_points, noise_std=0.5, frequency=3, amplitude=2):
    """
    Generate sinusoidal dataset with Gaussian noise and append anomaly points.
    """
    x_vals = np.linspace(-9, 9, num_points)
    raw_y_vals = amplitude * np.sin(frequency * x_vals) + np.random.normal(0, noise_std, size=num_points)

    # Normalize y to [-9, 9]
    y_min, y_max = raw_y_vals.min(), raw_y_vals.max()
    scaled_y_vals = (raw_y_vals - y_min) / (y_max - y_min)
    y_vals = scaled_y_vals * 18 - 9

    data_sin = np.column_stack((x_vals, y_vals))
    labels_sin = np.zeros(num_points, dtype=int)
    data_sin_all, y_sin_all = append_anomalies(data_sin, anomalies_sin)
    return data_sin_all, y_sin_all


def createHelix(num_points=n_points, noise_std=0.2, frequency=2, amplitude=6):
    """
    Generate a 2D double helix dataset (two intertwined sine waves).
    """
    if seed is not None:
        np.random.seed(seed)

    x_vals = np.linspace(-10, 10, num_points)
    y1 = amplitude * np.sin(frequency * x_vals) + np.random.normal(0, noise_std, size=num_points)
    y2 = -amplitude * np.sin(frequency * x_vals) + np.random.normal(0, noise_std, size=num_points)

    data_helix = np.concatenate([
        np.column_stack((x_vals, y1)),
        np.column_stack((x_vals, y2))
    ])

    labels_helix = np.zeros(data_helix.shape[0], dtype=int)
    data_helix_all, labels_helix_all = append_anomalies(data_helix, anomalies_helix)
    return data_helix_all, labels_helix_all


# =============================================
# 🔹 DataLoader class
# =============================================
class DataLoader:
    """
    Handles data loading from files and synthetic dataset generation.
    """

    @staticmethod
    def load_data(file_path):
        """
        Load dataset from CSV, Excel, or JSON file.

        The file must contain a column named 'outlier' (0 = normal, 1 = anomaly).
        """
        if file_path.endswith(".csv"):
            dataset = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            dataset = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            dataset = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")

        if "outlier" not in dataset.columns:
            raise ValueError("Dataset must contain 'outlier' column for true labels")

        y = dataset["outlier"].values
        X = dataset.drop(['outlier'], axis=1).values
        return X, y, dataset

    @staticmethod
    def create_SyntheticData(choice):
        """
        Generate a synthetic dataset based on user selection.
        Supported types: moons, circle, blobs, spiral, sin, helix
        """
        print("---------")
        X, y = None, None

        if choice == "moons":
            X, y = createMoons()
        elif choice == "circle":
            X, y = createCircles()
        elif choice == "blobs":
            X, y = createBlobs()
        elif choice == "spiral":
            X, y = createSpiral()
        elif choice == "sin":
            X, y = createSinusoidalData()
        elif choice == "helix":
            X, y = createHelix()
        else:
            messagebox.showerror("Error", "Unknown dataset type selected.")

        return X, y
