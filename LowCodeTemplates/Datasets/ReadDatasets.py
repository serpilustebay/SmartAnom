from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# -----------------------------
# Dataset yolları
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Bu dosyanın bulunduğu dizin

dataset_paths = {
    "ALOI": os.path.join(BASE_DIR, "Datasets", "ALOI", "ALOI_withoutdupl.arff"),
    "Arrhythmia": os.path.join(BASE_DIR, "Datasets", "Arrhythmia", "Arrhythmia_withoutdupl_10_v01.arff"),
    "Glass": os.path.join(BASE_DIR, "Datasets", "Glass", "Glass_withoutdupl_norm.arff"),
    "HeartDisease": os.path.join(BASE_DIR, "Datasets", "HeartDisease", "HeartDisease_withoutdupl_10_v01.arff"),
    "Hepatitis": os.path.join(BASE_DIR, "Datasets", "Hepatitis", "Hepatitis_withoutdupl_10_v01.arff"),
    "Ionosphere": os.path.join(BASE_DIR, "Datasets", "Ionosphere", "Ionosphere_withoutdupl_norm.arff"),
    "InternetAds": os.path.join(BASE_DIR, "Datasets", "InternetAds", "InternetAds_norm_10_v01.arff"),
    "KDDCup": os.path.join(BASE_DIR, "Datasets", "KDDCup99", "KDDCup99_original.arff"),
    "Lymphography": os.path.join(BASE_DIR, "Datasets", "Lymphography", "Lymphography_original.arff"),
    "PageBlocks": os.path.join(BASE_DIR, "Datasets", "PageBlocks", "PageBlocks_10.arff"),
    "Pima": os.path.join(BASE_DIR, "Datasets", "Pima", "Pima_withoutdupl_10_v01.arff"),
    "PWBC": os.path.join(BASE_DIR, "Datasets", "WPBC", "WPBC_withoutdupl_norm.arff"),
    "Shuttle": os.path.join(BASE_DIR, "Datasets", "Shuttle", "Shuttle_withoutdupl_v01.arff"),
    "Wilt": os.path.join(BASE_DIR, "Datasets", "Wilt", "Wilt_withoutdupl_05.arff")
}

def read_arff(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Byte tipindeki stringleri decode et
    for col in df.select_dtypes([object]).columns:
        if df[col].dropna().apply(lambda x: isinstance(x, (bytes, bytearray))).any():
            df[col] = df[col].apply(lambda v: v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v)
    return df


def load_dataset(name, categorical_cols=None):
    """
    Genel veri seti okuma fonksiyonu.

    Args:
        name: dataset_paths sözlüğünde anahtar
        categorical_cols: Liste halinde kategorik kolonlar (LabelEncoder uygulanacak)
    Returns:
        X, y: numpy array
    """
    df = read_arff(dataset_paths[name])
    df = df.drop(columns=['id'], errors='ignore')
    df['outlier'] = df['outlier'].map({"yes": 1, "no": 0})

    if categorical_cols:
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

    y = df['outlier'].values
    X = df.drop(columns=['outlier']).values
    return X, y

# Sample Usage:
# X, y = load_dataset("ALOI")
# X, y = load_dataset("KDDCup", categorical_cols=["protocol_type", "service", "flag"])
# X, y = load_dataset("Lymphography", categorical_cols=['Lymphatics', 'Block_of_affere', ...])
