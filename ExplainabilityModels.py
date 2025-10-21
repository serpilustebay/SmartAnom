import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import shap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.tree import DecisionTreeRegressor


def get_sample_slices(X, max_background=200, max_explain=100):
    """Select background and explanation subsets for SHAP based on dataset size."""
    n_samples = X.shape[0]
    bg_size = min(max_background, max(10, int(n_samples * 0.5)))
    ex_size = min(max_explain, max(5, int(n_samples * 0.25)))
    if n_samples <= 20:
        bg_size = ex_size = n_samples
    return X[:bg_size], X[:ex_size]


class ExplainabilityModels:
    """Provide SHAP-based explainability for models used in SmartAnom."""

    @staticmethod
    def _show_progress_popup(message="Processing, please wait..."):
        """Show popup window with an indeterminate progress bar."""
        popup = tk.Toplevel()
        popup.title("Processing")
        popup.geometry("340x140")
        popup.resizable(False, False)
        popup.configure(bg="#f8f9fa")

        tk.Label(popup, text=message, font=("Arial", 11, "bold"),
                 bg="#f8f9fa", fg="#333").pack(pady=15)

        style = ttk.Style(popup)
        style.theme_use("clam")
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor="#e0e0e0",
                        background="#4CAF50",
                        thickness=20)

        pb = ttk.Progressbar(popup, mode="indeterminate",
                             length=260,
                             style="Custom.Horizontal.TProgressbar")
        pb.pack(pady=10)
        pb.start(10)

        tk.Label(popup, text="This may take a few moments...",
                 bg="#f8f9fa", fg="#555").pack()
        popup.update()
        return popup, pb

    @staticmethod
    def _show_plot_in_window(fig, title="SHAP Summary"):
        """Display a Matplotlib figure inside a Tkinter popup with save/close options."""
        win = tk.Toplevel()
        win.title(title)
        win.geometry("900x650")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def save_figure():
            """Save the displayed SHAP figure to PNG file."""
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png")],
                    title="Save Plot As"
                )
                if file_path:
                    fig.savefig(file_path, dpi=300, bbox_inches="tight")
                    messagebox.showinfo("Success",
                                        f"Plot saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot:\n{str(e)}")

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Save as PNG", command=save_figure,
                  bg="#009688", fg="white", width=15).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Close", command=win.destroy,
                  bg="#E91E63", fg="white", width=15).pack(side="left", padx=10)

    @staticmethod
    def explain_isolation_forest(model: IsolationForest, X, feature_names=None):
        """Generate SHAP summary plot for Isolation Forest model."""
        popup, pb = ExplainabilityModels._show_progress_popup("Explaining Isolation Forest...")
        try:
            X_bg, X_ex = get_sample_slices(X)
            explainer = shap.Explainer(model.predict, X_bg)
            shap_values = explainer(X_ex)
            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
            pb.stop(); popup.destroy()
            ExplainabilityModels._show_plot_in_window(fig, "Isolation Forest SHAP Summary")
        except Exception as e:
            pb.stop(); popup.destroy()
            print(f"[Explainability] IsolationForest SHAP failed: {e}")

    @staticmethod
    def explain_autoencoder(model, X, feature_names=None):
        """Generate SHAP summary plot for Autoencoder reconstruction error."""
        popup, pb = ExplainabilityModels._show_progress_popup("Explaining Autoencoder...")
        try:
            X_bg, X_ex = get_sample_slices(X)
            predict_fn = lambda x: np.mean((x - model.predict(x, verbose=0)) ** 2, axis=1)
            explainer = shap.Explainer(predict_fn, X_bg)
            shap_values = explainer(X_ex)
            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
            pb.stop(); popup.destroy()
            ExplainabilityModels._show_plot_in_window(fig, "Autoencoder SHAP Summary")
        except Exception as e:
            pb.stop(); popup.destroy()
            print(f"[Explainability] Autoencoder SHAP failed: {e}")

    @staticmethod
    def explain_vae(model, X, feature_names=None):
        """Generate SHAP summary plot for Variational Autoencoder reconstruction error."""
        popup, pb = ExplainabilityModels._show_progress_popup("Explaining Variational Autoencoder...")
        try:
            X_bg, X_ex = get_sample_slices(X)
            X_bg, X_ex = np.array(X_bg, np.float32), np.array(X_ex, np.float32)
            predict_fn = lambda x: np.mean((x - model.predict(x, verbose=0)) ** 2, axis=1)
            explainer = shap.Explainer(predict_fn, X_bg)
            shap_values = explainer(X_ex)
            if shap_values is None or np.all(np.abs(shap_values.values) < 1e-9):
                pb.stop(); popup.destroy()
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No meaningful SHAP values",
                        ha="center", va="center", color="gray")
                ax.axis("off")
                ExplainabilityModels._show_plot_in_window(fig, "VAE SHAP (Empty)")
                return
            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
            pb.stop(); popup.destroy()
            ExplainabilityModels._show_plot_in_window(fig, "VAE SHAP Summary")
        except Exception as e:
            pb.stop(); popup.destroy()
            print(f"[Explainability] VAE SHAP failed: {e}")

    @staticmethod
    def explain_deepsvdd(model, X, feature_names=None, center_attr=None):
        """Generate SHAP summary plot for DeepSVDD using distance to center."""
        popup, pb = ExplainabilityModels._show_progress_popup("Explaining DeepSVDD model...")
        try:
            X_bg, X_ex = get_sample_slices(X)
            X_bg, X_ex = np.asarray(X_bg, np.float32), np.asarray(X_ex, np.float32)
            c = getattr(model, center_attr, None) if center_attr else None
            if c is None:
                c = np.mean(model.predict(X_bg, verbose=0), axis=0)
            def distance_score(x): return np.sum((model.predict(x, verbose=0) - c) ** 2, axis=1)
            explainer = shap.Explainer(distance_score, X_bg)
            shap_values = explainer(X_ex)
            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
            pb.stop(); popup.destroy()
            ExplainabilityModels._show_plot_in_window(fig, "DeepSVDD SHAP Summary")
        except Exception as e:
            pb.stop(); popup.destroy()
            print(f"[Explainability] DeepSVDD SHAP failed: {e}")

    @staticmethod
    def explain_ocsvm(model: OneClassSVM, X, feature_names=None):
        """Generate SHAP summary plot for One-Class SVM."""
        popup, pb = ExplainabilityModels._show_progress_popup("Explaining One-Class SVM...")
        try:
            X_bg, X_ex = get_sample_slices(X)
            explainer = shap.Explainer(model.decision_function, X_bg)
            shap_values = explainer(X_ex)
            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
            pb.stop(); popup.destroy()
            ExplainabilityModels._show_plot_in_window(fig, "One-Class SVM SHAP Summary")
        except Exception as e:
            pb.stop(); popup.destroy()
            print(f"[Explainability] One-Class SVM SHAP failed: {e}")

    @staticmethod
    def explain_lof(model: LocalOutlierFactor, X, feature_names=None):
        """Generate SHAP explanation for Local Outlier Factor using surrogate tree."""
        popup, pb = ExplainabilityModels._show_progress_popup("Explaining Local Outlier Factor...")
        try:
            lof_scores = -model.negative_outlier_factor_
            tree = DecisionTreeRegressor(max_depth=4, random_state=42)
            tree.fit(X, lof_scores)
            explainer = shap.TreeExplainer(tree)
            shap_values = explainer.shap_values(X)
            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            pb.stop(); popup.destroy()
            ExplainabilityModels._show_plot_in_window(fig, "LOF SHAP (Surrogate Tree)")
        except Exception as e:
            pb.stop(); popup.destroy()
            print(f"[Explainability] LOF SHAP (surrogate) failed: {e}")

    @staticmethod
    def explain_elliptic(model: EllipticEnvelope, X, feature_names=None):
        """Plot Mahalanobis distance distribution for Elliptic Envelope model."""
        fig, ax = plt.subplots()
        dist = model.mahalanobis(X)
        ax.hist(dist, bins=30, color='salmon', edgecolor='black')
        ax.set_title("Mahalanobis Distance Distribution (Elliptic Envelope)")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Frequency")
        ExplainabilityModels._show_plot_in_window(fig, "Elliptic Envelope Distribution")
