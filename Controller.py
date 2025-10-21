"""
AnomalyController.py
--------------------
Central control layer for SmartAnom framework.
Handles dataset loading, model execution, metric computation,
and SHAP/XAI-based explainability for supported algorithms.

"""

import Scores
from DataLayer import DataLoader
from ExplainabilityModels import ExplainabilityModels
from Models import IFModel, BenchmarkModels


class AnomalyController:
    """
    The controller layer orchestrates all SmartAnom operations:
    - Dataset loading and preprocessing.
    - Model execution (IF-family, benchmark models).
    - Metric computation and tracking.
    - Explainability via SHAP/XAI.

    Attributes:
        X (ndarray): Feature matrix.
        y (ndarray): Labels (0 = normal, 1 = anomaly).
        dataset (pd.DataFrame): Original dataset reference.
        last_trained_model (object): Last fitted model instance.
        last_model_name (str): Name of last executed model.
        last_metrics (dict): Evaluation metrics of the last run.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.dataset = None
        self.last_trained_model = None
        self.last_model_name = None
        self.last_metrics = None

    # ============================================================= #
    # 🧩 DATA OPERATIONS
    # ============================================================= #
    def load_dataset(self, file_path):
        """
        Load dataset from CSV, Excel, or JSON file and split into X, y.

        Args:
            file_path (str): Path to the dataset file.

        Returns:
            pd.DataFrame: Preview of first 10 rows.
        """
        self.X, self.y, self.dataset = DataLoader.load_data(file_path)
        return self.dataset.head(10)

    def load_synthetic_data(self, choice):
        """
        Generate synthetic dataset (e.g., moons, spiral, blobs).

        Args:
            choice (str): Type of synthetic dataset.

        Returns:
            tuple: (X, y) arrays.
        """
        self.X, self.y = DataLoader.create_SyntheticData(choice)
        self.dataset = None
        return self.X, self.y

    # ============================================================= #
    # 🧠 MODEL TRAINING & EVALUATION
    # ============================================================= #
    def run_if_model(self, model_type, score_method, params):
        """
        Run Isolation Forest family models:
        IF, EIF, GIF, SciForest, FairCutForest.

        Args:
            model_type (str): Model name.
            score_method (str): Scoring method ("MBAS" or "SBAS").
            params (dict): Custom hyperparameters.

        Returns:
            dict: Evaluation metrics.
        """
        if self.X is None or self.y is None:
            raise ValueError("Dataset not loaded.")

        model_data = Scores.get_param_grid_for_model(model_type)
        defaults = model_data.get("defaults", {})
        full_params = defaults.copy()
        full_params.update(params)

        n_trees = full_params.get("n_trees", 100)
        sample_size = min(full_params.get("sample_size", 256), self.X.shape[0])
        contamination = full_params.get("contamination", 0.1)
        level = full_params.get("level", 1)
        k_planes = full_params.get("k_planes", 2)
        threshold = full_params.get("threshold", 0.95)
        majority = full_params.get("majority", 5)

        model = IFModel(
            model_type=model_type,
            score_method=score_method,
            n_tree=n_trees,
            sample_size=sample_size,
            contamination=contamination,
            level=level,
            k_planes=k_planes,
            threshold=threshold,
            majority=majority
        )

        metrics = model.evaluate(self.X, self.y)
        self.last_trained_model = model
        self.last_model_name = model_type
        self.last_metrics = metrics
        return metrics

    def run_benchmark_model(self, selected_model, hyperparams):
        """
        Run benchmark models:
        Autoencoder, VAE, DeepSVDD, OCSVM, LOF, Elliptic Envelope.

        Args:
            selected_model (str): Model name.
            hyperparams (dict): Hyperparameter dictionary.

        Returns:
            dict: Evaluation metrics.
        """
        if self.X is None or self.y is None:
            raise ValueError("Dataset not loaded.")

        metrics, model_instance = BenchmarkModels.run(
            self.X,
            self.y,
            selected_model,
            hyperparams
        )

        self.last_trained_model = model_instance
        self.last_model_name = selected_model
        self.last_metrics = metrics
        return metrics

    # ============================================================= #
    # 🧩 EXPLAINABILITY (SHAP & XAI)
    # ============================================================= #
    def run_explainability(self, model_name=None):
        """
        Run SHAP-based explainability analysis for the last trained model.

        Args:
            model_name (str, optional): Target model name.
                If None, uses last executed model.
        """
        if self.X is None:
            raise ValueError("Dataset not loaded.")
        if self.last_trained_model is None:
            raise ValueError("No trained model found.")

        model_name = model_name or self.last_model_name
        model_obj = self.last_trained_model
        feature_names = getattr(self.dataset, "columns", None)

        explain_map = {
            "Sklearn IF": lambda m, X: ExplainabilityModels.explain_isolation_forest(m, X, feature_names=feature_names),
            "Autoencoder": lambda m, X: ExplainabilityModels.explain_autoencoder(m, X, feature_names=feature_names),
            "VAE": lambda m, X: ExplainabilityModels.explain_vae(m, X, feature_names=feature_names),
            "DeepSVDD": lambda m, X: ExplainabilityModels.explain_deepsvdd(m, X, feature_names=feature_names),
            "One-Class SVM": lambda m, X: ExplainabilityModels.explain_ocsvm(m, X, feature_names=feature_names),
            "Local Outlier Factor": lambda m, X: ExplainabilityModels.explain_lof(m, X, feature_names=feature_names),
            "Elliptic Envelope": lambda m, X: ExplainabilityModels.explain_elliptic(m, X, feature_names=feature_names)
        }

        if model_name not in explain_map:
            raise ValueError(f"No explainability defined for: {model_name}")

        print(f"[Controller] Running SHAP explainability for {model_name}")
        explain_map[model_name](model_obj, self.X)
        print(f"[Controller] Explainability completed for {model_name}")

    # ============================================================= #
    # 🧩 BATCH EXECUTION (ALL MODELS)
    # ============================================================= #
    def run_all_models(self):
        """
        Run all available anomaly detection models sequentially.

        Returns:
            list[dict]: [{"Model": model_name, "Accuracy": acc}, ...]
        """
        if self.X is None or self.y is None:
            raise ValueError("Dataset not loaded.")

        model_list = [
            "Isolation Forest", "Extended Isolation Forest", "Generalized Isolation Forest",
            "SciForest", "FairCutForest",
            "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope",
            "Autoencoder", "VAE", "DeepSVDD"
        ]

        results = []
        for model in model_list:
            try:
                if model in ["Isolation Forest", "Extended Isolation Forest",
                             "Generalized Isolation Forest", "SciForest", "FairCutForest"]:
                    metrics = self.run_if_model(model, "SBAS", {})
                else:
                    metrics = self.run_benchmark_model(model, {})

                acc = metrics.get("Accuracy", 0)
                results.append({"Model": model, "Accuracy": acc})
                print(f"[Controller] {model} → Accuracy: {acc:.3f}")
            except Exception as e:
                print(f"[Controller] ⚠️ {model} failed: {e}")
                results.append({"Model": model, "Accuracy": 0})

        return results
