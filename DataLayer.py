import pandas as pd
from tkinter import messagebox

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs



# Spiral Dataset
n_points = 800
seed=343

import numpy as np

anomalies_moons = np.array([
    #[4.5, 3.5], [-4.5, -3.5],
    #[5.5, -2], [-5.5, 2],
    [3.5, -6], [-3, 6],
    [5, 4], [-4, -4],
    [0, 6], [0, -6],
    [0, 0], [-5, -5]
])

anomalies_circles = np.array([
    [0, 0],  # iç boşluğun ortasında
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


def append_anomalies(X, anomalies):
    X_new = np.vstack([X, anomalies])
    y_new = np.hstack([np.zeros(len(X)), np.ones(len(anomalies))])
    return X_new, y_new


def generate_spiral(n_points=n_points, noise=0.1, revolutions=6):
    theta = np.linspace(0, 2 * np.pi * revolutions, n_points)

    scale = 0.5  # r'nin büyüme katsayısı, küçültüp spiral sıklaştırıyoruz
    r = theta * scale

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)

    # Normalize et -10 ile 10 arasına sığdır
    def scale_to_range(arr, new_min, new_max):
        arr_min, arr_max = arr.min(), arr.max()
        return (arr - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min

    x_scaled = scale_to_range(x, -10, 10)
    y_scaled = scale_to_range(y, -10, 10)

    return np.stack([x_scaled, y_scaled], axis=1)




def createMoons():
    # Dataset 1: Moons
    X_moons, _ = make_moons(n_samples=n_points, noise=0.15, random_state=seed)
    X_moons = (X_moons * 5) - 2
    X_moons_all, y_moons_all = append_anomalies(X_moons, anomalies_moons)
    return X_moons_all, y_moons_all


# Dataset 2: Circles
def createCircles():
    X_circles, _ = make_circles(n_samples=n_points, noise=0.05, factor=0.5, random_state=seed)
    X_circles_scaled = X_circles * 8
    X_circles_all, y_circles_all = append_anomalies(X_circles_scaled, anomalies_circles)
    return X_circles_all, y_circles_all


# Dataset 3: Blobs
def createBlobs():
    # Küme merkezlerini daha yakın yapalım, mesela -1.5, 0, 1.5 aralığında
    centers = np.array([
        [-6, 4],
        [6, 6],
        [0, -4]
    ])
    # Küme standart sapması aynı kalsın
    X_blobs, _ = make_blobs(n_samples=n_points, centers=centers, cluster_std=1.0, random_state=seed)
    X_blobs_all, y_blobs_all = append_anomalies(X_blobs, anomalies_blobs)
    return X_blobs_all, y_blobs_all


# Dataset 4: Spiral
def createSpiral():
    X_spiral = generate_spiral()
    X_spiral_all, y_spiral_all = append_anomalies(X_spiral, anomalies_spiral)
    return X_spiral_all, y_spiral_all


#Dataset 5: Sin
def createSinusoidalData(num_points=n_points, noise_std=0.5, frequency=3, amplitude=2):
    # X aralığını -9 ile 9 arasına ayarla
    x_vals = np.linspace(-9, 9, num_points)

    # Sinüs eğrisini oluştur
    raw_y_vals = amplitude * np.sin(frequency * x_vals) + np.random.normal(0, noise_std, size=num_points)

    # Normalize edip -9 ile 9 arasına taşı
    y_min, y_max = raw_y_vals.min(), raw_y_vals.max()
    scaled_y_vals = (raw_y_vals - y_min) / (y_max - y_min)  # 0-1 aralığına getir
    y_vals = scaled_y_vals * 18 - 9  # -9 ile 9 arasına ölçekle

    data_sin = np.column_stack((x_vals, y_vals))
    labels_sin = np.zeros(num_points, dtype=int)

    # Anomalileri ekle
    data_sin_all, y_sin_all = append_anomalies(data_sin, anomalies_sin)
    return data_sin_all, y_sin_all

#HELİX dataset
def createHelix(num_points=n_points, noise_std=0.2, frequency=2, amplitude=6):
    """
    İki sarmallı (double helix) veri seti üretir. x ekseni -10 ile 10 arasına yayılır.
    """
    if seed is not None:
        np.random.seed(seed)

    # x [-10, 10] aralığında eşit dağılmış
    x_vals = np.linspace(-10, 10, num_points)

    # İki sarmal için y değerleri (biri pozitif sinüs, diğeri negatif sinüs)
    y1 = amplitude * np.sin(frequency * x_vals) + np.random.normal(0, noise_std, size=num_points)
    y2 = -amplitude * np.sin(frequency * x_vals) + np.random.normal(0, noise_std, size=num_points)

    # Her iki sarmalı birleştir
    data_helix = np.concatenate([
        np.column_stack((x_vals, y1)),
        np.column_stack((x_vals, y2))
    ])

    labels_helix = np.zeros(data_helix.shape[0], dtype=int)  # Tüm noktalar "normal"
    data_helix_all, labels_helix_all = append_anomalies(data_helix, anomalies_helix)

    return data_helix_all, labels_helix_all



class DataLoader:

    @staticmethod
    def load_data(file_path):
        # Identify file format and load accordingly
        if file_path.endswith(".csv"):
            dataset = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            dataset = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            dataset = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")

        # Ensure the dataset contains a column with ground truth labels
        if "outlier" not in dataset.columns:
            raise ValueError("Dataset must contain 'outlier' column for true labels")

        # Separate features (X) and labels (y)
        y = dataset["outlier"].values
        X = dataset.drop(['outlier'], axis=1).values

        # Return preview of the dataset
        return X, y, dataset

    @staticmethod
    def create_SyntheticData(choice):
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
