"""
HyperparameterSearch.py
-----------------------
Performs grid search optimization for SmartAnom models
and integrates with GUI progress & reporting.
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
    Grid Search optimization engine for SmartAnom.
    Supports tree-based models (IF, EIF, GIF, SciForest, FairCutForest)
    and benchmark models (OCSVM, LOF, EE, AE, VAE, DeepSVDD).
    """

    @staticmethod
    def grid_search_IF(controller, X, y, model_name, metric, param_grid):
        """
        Execute grid search for a given SmartAnom model.

        Args:
            controller (object): SmartAnom controller instance.
            X (ndarray): Feature matrix.
            y (ndarray): True labels (0/1).
            model_name (str): Target model name.
            metric (str): Metric to optimize ("f1", "accuracy", "precision", "recall").
            param_grid (dict): Parameter grid {param_name: [values]}.

        Returns:
            tuple: (best_params, best_score, results)
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        best_score = -np.inf
        best_params = None
        results = []

        total = len(all_combinations)
        print(f"Starting Grid Search for {model_name} ({total} combinations)")

        for i, params in enumerate(all_combinations, start=1):
            try:
                print(params)
                if model_name in ["Isolation Forest", "Extended Isolation Forest",
                                  "Generalized Isolation Forest", "SciForest", "FairCutForest"]:
                    performance_metrics = controller.run_if_model(model_name, score_method="SBAS", params=params)
                else:
                    performance_metrics = controller.run_benchmark_model(model_name, params)

                score = HyperparameterSearch.extract_metric(performance_metrics, metric)
                results.append({"params": params, metric: score})

                if score > best_score:
                    best_score = score
                    best_params = params

                print(f"[{i}/{total}] {model_name} | {metric.upper()} = {score:.4f}")

            except Exception as e:
                print(f"Error at {params}: {e}")
                results.append({"params": params, metric: None})
                continue

        HyperparameterSearch.save_results(model_name, metric, results)
        HyperparameterSearch.save_best_params(model_name, best_params, best_score, metric)

        try:
            print("Training best model automatically...")
            if model_name in ["Isolation Forest", "Extended Isolation Forest",
                              "Generalized Isolation Forest", "SciForest", "FairCutForest"]:
                controller.run_if_model(model_name, score_method="SBAS", **best_params)
            else:
                controller.run_benchmark_model(model_name, best_params)
            print("Best model retrained successfully.")
        except Exception as e:
            print(f"⚠️ Could not retrain best model: {e}")

        print(f"🏁 Grid Search completed for {model_name}. Best {metric}: {best_score:.4f}")
        return best_params, best_score, results

    @staticmethod
    def extract_metric(performance_metrics, metric):
        """
        Extract the target metric value from controller output.

        Args:
            performance_metrics (dict): Metrics dictionary.
            metric (str): Metric key.

        Returns:
            float: Extracted metric value.
        """
        try:
            if metric in performance_metrics:
                return float(performance_metrics[metric])
        except Exception:
            return 0.0

    @staticmethod
    def save_results(model_name, metric, results):
        """
        Save all parameter combinations and their metric scores as CSV.

        Args:
            model_name (str): Model name.
            metric (str): Optimized metric.
            results (list[dict]): Search results.
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

    @staticmethod
    def save_best_params(model_name, best_params, best_score, metric):
        """
        Save the best parameter configuration as JSON.

        Args:
            model_name (str): Model name.
            best_params (dict): Best parameter set.
            best_score (float): Best achieved score.
            metric (str): Optimization metric.
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
            print(f"Best params saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save best parameters: {e}")
