"""
main.py
-------
Main GUI module for the SmartAnom framework.
Provides a Tkinter-based light-themed graphical interface
for dataset handling, model training, hyperparameter optimization,
and SHAP explainability.

Author: Serpil Üstebay
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Controller import AnomalyController
from Scores import get_param_grid_for_model

# Internal MVC modules
import VisualizeWindow
import HyperSearchWindow


# ============================================================
# 🌈 Light Modern Theme
# ============================================================
def apply_modern_style(root: tk.Tk):
    """
    Apply a modern light UI theme to the SmartAnom interface.

    Args:
        root (tk.Tk): Root Tkinter window.
    """
    style = ttk.Style(root)
    style.theme_use("clam")

    base_bg = "#ffffff"
    card_bg = "#F8FAFF"
    text_fg = "#1A1A1A"
    accent = "#5cc6a0"
    accent_hover = "#05afca"
    secondary = "#95eefc"
    warning = "#FFB300"
    danger = "#E85C5C"

    root.configure(bg=base_bg)
    style.configure(".", background=base_bg, foreground=text_fg, font=("Segoe UI", 10))

    # Buttons
    style.configure("TButton", font=("Segoe UI", 10, "bold"),
                    background=accent, foreground="#000000",
                    borderwidth=0, padding=8)
    style.map("TButton", background=[("active", accent_hover)])

    style.configure("Accent.TButton", background=secondary, foreground="#000000", font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", background=[("active", accent_hover)])

    style.configure("Warning.TButton", background=warning, foreground="#000000", font=("Segoe UI", 10, "bold"))
    style.map("Warning.TButton", background=[("active", "#FFB300")])

    style.configure("Danger.TButton", background=danger, foreground="#000000", font=("Segoe UI", 10, "bold"))
    style.map("Danger.TButton", background=[("active", "#EF5350")])

    style.configure("Treeview", background="white", foreground="#000000",
                    fieldbackground="white", rowheight=26, font=("Segoe UI", 9))
    style.configure("Treeview.Heading", background=accent, foreground="#000000",
                    font=("Segoe UI", 10, "bold"))
    style.configure("Header.TLabel", background=card_bg, foreground="#000000",
                    font=("Segoe UI", 11, "bold"))


# ============================================================
# 🧩 SmartAnom Main GUI
# ============================================================
class AnomalyGUI:
    """
    Main SmartAnom GUI class.
    Handles:
        - Dataset loading & visualization
        - Model selection & training
        - Metric reporting
        - SHAP explainability
        - Hyperparameter optimization
    """

    def __init__(self, root):
        """Initialize GUI layout, controller, and interface components."""
        self.root = root
        self.root.title("SmartAnom")
        self.root.geometry("1200x800+0+0")
        self.root.minsize(1000, 700)
        self.controller = AnomalyController()
        self.hyperparams = {}

        # Main scrollable layout
        main_frame = tk.Frame(root, bg="#f8f9fa")
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        container = self.scrollable_frame

        # ======================================================
        # 📁 Dataset Upload & Synthetic Data Generation
        # ======================================================
        top_frame = tk.Frame(container, bg="#f8f9fa")
        top_frame.pack(pady=10, padx=15, fill="x")

        # Upload panel
        upload_frame = tk.Frame(top_frame, bg="#ffffff", bd=1, relief="solid", padx=15, pady=10)
        upload_frame.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(upload_frame, text="📁 Upload Dataset", style="Header.TLabel").pack(anchor="w", pady=3)
        ttk.Button(upload_frame, text="Choose File", style="Accent.TButton", command=self.upload_file).pack(side="left", padx=5, pady=5)
        self.file_path_label = ttk.Label(upload_frame, text="No file selected", foreground="#000000")
        self.file_path_label.pack(side="left", padx=10)

        # Synthetic panel
        synth_frame = tk.Frame(top_frame, bg="#ffffff", bd=1, relief="solid", padx=15, pady=10)
        synth_frame.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(synth_frame, text="🧬 Create Synthetic Data", style="Header.TLabel").pack(anchor="w", pady=3)
        ttk.Button(synth_frame, text="Generate Synthetic Data", style="Accent.TButton",
                   command=self.open_synthetic_window).pack(side="left", padx=5, pady=5)

        # ======================================================
        # 📊 Dataset Preview
        # ======================================================
        self.table_frame = tk.Frame(container, bg="#f8f9fa")
        self.table_frame.pack(pady=10, fill="both", expand=True)

        # ======================================================
        # 🧩 Model Panels (IF, Benchmarks, Metrics)
        # ======================================================
        # (remaining initialization unchanged for brevity)

    # ----------------------------------------------------------
    # Dataset, Model, and Explainability Handlers
    # ----------------------------------------------------------
    def upload_file(self):
        """Open file dialog and load selected dataset."""
        file_path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("JSON", "*.json")])
        if file_path:
            self.file_path_label.config(text=file_path)
            try:
                df_head = self.controller.load_dataset(file_path)
                self.show_table(df_head)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def open_synthetic_window(self):
        """Open popup to select and create synthetic dataset."""
        win = tk.Toplevel(self.root)
        win.title("Generate Synthetic Data")
        win.geometry("260x320")
        win.configure(bg="#ffffff")
        ttk.Label(win, text="Select Synthetic Dataset", style="Header.TLabel").pack(pady=10)
        self.synth_choice = tk.StringVar(value="moons")
        for name, val in [("Moons", "moons"), ("Circles", "circle"), ("Blobs", "blobs"),
                          ("Spiral", "spiral"), ("Sinusoidal", "sin"), ("Helix", "helix")]:
            ttk.Radiobutton(win, text=name, variable=self.synth_choice, value=val).pack(anchor="w", padx=20)
        ttk.Button(win, text="Create", style="TButton",
                   command=lambda: [win.destroy(), self.create_synthetic_data()]).pack(pady=15, fill="x")

    def create_synthetic_data(self):
        """Generate selected synthetic dataset and visualize."""
        choice = self.synth_choice.get()
        X, y = self.controller.load_synthetic_data(choice)
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ['#2196F3' if label == 0 else '#F44336' for label in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=30, edgecolor="k")
        ax.set_title(f"Synthetic Data: {choice.capitalize()}", color="#000000")
        ax.set_aspect('equal', adjustable='box')

        self.canvas = FigureCanvasTkAgg(fig, master=self.table_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_if_model(self):
        """Run selected Isolation Forest family model."""
        try:
            model = self.if_model.get()
            score = self.score_method.get()
            params = self.hyperparams.get(model, {})
            metrics = self.controller.run_if_model(model, score, params)
            self.update_metrics(metrics, title=f"{model} ({score}) Results")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_benchmarks(self):
        """Run selected benchmark model (AE, VAE, DeepSVDD, etc.)."""
        model = self.benchmark_model.get()
        try:
            params = self.hyperparams.get(model, {})
            metrics = self.controller.run_benchmark_model(model, params)
            self.update_metrics(metrics, title=f"{model} Results")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_shap_explainability(self):
        """Perform SHAP-based explainability for last trained model."""
        try:
            messagebox.showinfo("Explainability", "Starting SHAP explainability analysis...")
            self.controller.run_explainability()
            messagebox.showinfo("Explainability", "SHAP analysis completed. Check SHAP plot window.")
        except Exception as e:
            messagebox.showerror("Error", f"SHAP analysis failed:\n{str(e)}")


# ============================================================
# Run App
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    apply_modern_style(root)
    app = AnomalyGUI(root)
    root.mainloop()
