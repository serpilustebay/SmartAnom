"""
HyperparameterSearch.py
-----------------------
Performs grid search optimization for SmartAnom models
and integrates with GUI progress & reporting.

Author: Serpil Üstebay
"""

import itertools
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datetime import datetime


class HyperparameterSearch:
    """
    Performs Grid Search optimization for SmartAnom models.
    Supports: IF-family, SciForest, FairCutForest, OCSVM, LOF, EE, AE, VAE, DeepSVDD
    """

    @staticmethod
    def grid_search_IF(controller, X, y, model_name, metric, param_grid):
        """
        Generic Grid Search for all SmartAnom models.

        Parameters
        ----------
        controller : object
            AnomalyController instance.
        X, y : ndarray
            Dataset features and labels.
        model_name : str
            Model name (e.g. "Isolation Forest", "Autoencoder").
        metric : str
            Optimization metric ("f1", "accuracy", "precision", "recall").
        param_grid : dict
            Dictionary of parameter lists for grid search.

        Returns
        -------
        best_params : dict
            Parameters achieving the best score.
        best_score : float
            Best metric value.
        results : list[dict]
            All combinations and their metric scores.
        """


        # --- Hazırlık
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        best_score = -np.inf
        best_params = None
        results = []

        total = len(all_combinations)
        print(f"🔎 Starting Grid Search for {model_name} ({total} combinations)")

        # --- Arama Döngüsü
        for i, params in enumerate(all_combinations, start=1):
            try:
                print(params)
                # Modeli çalıştır
                if model_name in ["Isolation Forest", "Extended Isolation Forest",
                                  "Generalized Isolation Forest", "SciForest", "FairCutForest"]:
                    performance_metrics = controller.run_if_model(model_name, score_method="SBAS", params=params)
                else:
                    performance_metrics = controller.run_benchmark_model(model_name, params)

                # Geçerli metrik skorunu al
                score = HyperparameterSearch.extract_metric(performance_metrics, metric)
                results.append({"params": params, metric: score})

                # En iyiyi güncelle
                if score > best_score:
                    best_score = score
                    best_params = params

                print(f"[{i}/{total}] {model_name} | {metric.upper()} = {score:.4f}")

            except Exception as e:
                print(f"⚠️ Error at {params}: {e}")
                results.append({"params": params, metric: None})
                continue

        # --- Sonuçları dosyaya kaydet
        HyperparameterSearch.save_results(model_name, metric, results)

        # --- En iyi parametreleri JSON olarak kaydet
        HyperparameterSearch.save_best_params(model_name, best_params, best_score, metric)

        # --- En iyi modeli otomatik eğit
        try:
            print("🧠 Training best model automatically...")
            if model_name in ["Isolation Forest", "Extended Isolation Forest",
                              "Generalized Isolation Forest", "SciForest", "FairCutForest"]:
                controller.run_if_model(model_name, score_method="SBAS", **best_params)
            else:
                controller.run_benchmark_model(model_name, best_params)
            print("✅ Best model retrained successfully.")
        except Exception as e:
            print(f"⚠️ Could not retrain best model: {e}")

        print(f"🏁 Grid Search completed for {model_name}. Best {metric}: {best_score:.4f}")
        return best_params, best_score, results

    # ==========================================================
    # 🎯 Extract chosen metric
    # ==========================================================
    @staticmethod
    def extract_metric(performance_metrics, metric):
        """
        Extracts a metric value from controller output.
        """
        try:
            if metric in performance_metrics:
                return float(performance_metrics[metric])
        except Exception:
            return 0.0

    # ==========================================================
    # 💾 Save all results (CSV)
    # ==========================================================
    @staticmethod
    def save_results(model_name, metric, results):
        """
        Saves all parameter combinations and their metric scores to CSV.
        """
        try:
            os.makedirs("../SmartAnom/Optimization/results", exist_ok=True)
            df = pd.DataFrame([{
                **r["params"],
                metric: r.get(metric, None)
            } for r in results])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Optimization/results/{model_name.replace(' ', '_')}_grid_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"📁 Results saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    # ==========================================================
    # 💾 Save best parameters (JSON)
    # ==========================================================
    @staticmethod
    def save_best_params(model_name, best_params, best_score, metric):
        """
        Saves the best parameter set as JSON for later reuse.
        """
        try:
            os.makedirs("../SmartAnom/Optimization/best", exist_ok=True)
            filename = f"Optimization/best/{model_name.replace(' ', '_')}_best.json"
            data = {
                "model": model_name,
                "best_params": best_params,
                "best_score": best_score,
                "metric": metric
            }
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            print(f"💾 Best params saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save best parameters: {e}")
