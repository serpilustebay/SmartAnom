import tkinter as tk
from tkinter import ttk, messagebox
from Optimization import HyperparameterSearch
from Scores import get_param_grid_for_model


class HyperSearchWindow(tk.Toplevel):
    """SmartAnom – Unified Hyperparameter Search Window (Single Instance)."""

    def __init__(self, parent, model_name, controller):
        """
        Initialize the hyperparameter optimization window.

        Args:
            parent (tk.Widget): Parent window.
            model_name (str): Selected model name.
            controller: Main controller object containing dataset and methods.
        """
        super().__init__(parent)
        self.title("🔍 Hyperparameter Optimization")
        self.geometry("540x680")
        self.resizable(False, False)
        self.configure(bg="#f8f9fa")
        self.controller = controller
        self.model_name = model_name
        self.parent = parent
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        container = ttk.Frame(self, padding=15)
        container.pack(fill="both", expand=True)

        ttk.Label(container,
                  text="SmartAnom Hyperparameter Optimization",
                  font=("Segoe UI", 13, "bold")).pack(pady=(0, 10))
        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=5)

        # Model selection
        model_frame = ttk.Frame(container)
        model_frame.pack(fill="x", pady=5)
        ttk.Label(model_frame, text="Select Model:", font=("Segoe UI", 10)).pack(anchor="w")

        self.model_var = tk.StringVar(value=model_name)
        all_models = [
            "Isolation Forest", "Extended Isolation Forest", "Generalized Isolation Forest",
            "SciForest", "FairCutForest",
            "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope",
            "Autoencoder", "VAE", "DeepSVDD"
        ]
        model_box = ttk.Combobox(model_frame, textvariable=self.model_var,
                                 values=all_models, width=32, state="readonly")
        model_box.pack(anchor="w", pady=3)
        model_box.bind("<<ComboboxSelected>>", lambda e: self.load_param_grid())

        # Evaluation metric
        metric_frame = ttk.Frame(container)
        metric_frame.pack(fill="x", pady=5)
        ttk.Label(metric_frame, text="Evaluation Metric:", font=("Segoe UI", 10)).pack(anchor="w")
        self.metric_var = tk.StringVar(value="Accuracy")
        metric_box = ttk.Combobox(metric_frame, textvariable=self.metric_var,
                                  values=["Accuracy", "F1 Score", "Precision", "Recall"],
                                  width=18, state="readonly")
        metric_box.pack(anchor="w", pady=3)

        # Search type
        search_frame = ttk.Frame(container)
        search_frame.pack(fill="x", pady=5)
        ttk.Label(search_frame, text="Search Type:", font=("Segoe UI", 10)).pack(anchor="w")
        self.search_type = tk.StringVar(value="Grid Search")
        search_box = ttk.Combobox(search_frame, textvariable=self.search_type,
                                  values=["grid"], width=18, state="readonly")
        search_box.pack(anchor="w", pady=3)

        # Scrollable parameter frame
        ttk.Label(container, text="Parameter Ranges:",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 4))
        param_frame = ttk.Frame(container)
        param_frame.pack(fill="both", expand=True, pady=5)
        canvas = tk.Canvas(param_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(param_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.entries = {}
        self.load_param_grid()

        # Buttons
        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=10)
        ttk.Button(button_frame, text="Run Search", style="Accent.TButton",
                   command=self.run_search).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close", command=self.on_close).pack(side="right", padx=5)

        # Progress bar
        self.progress_frame = ttk.Frame(container)
        self.progress_frame.pack(fill="x", pady=(5, 10))
        self.progress = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=400)
        self.progress.pack(pady=5)
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack()

        # Results section
        result_frame = ttk.LabelFrame(container, text="Results", padding=10)
        result_frame.pack(fill="both", expand=False, pady=5)
        self.result_label = ttk.Label(result_frame, text="", font=("Segoe UI", 10), justify="left")
        self.result_label.pack(anchor="w")

    def load_param_grid(self):
        """Load the parameter grid dynamically for the selected model."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        model = self.model_var.get()
        model_data = get_param_grid_for_model(model)

        if not model_data or "grid" not in model_data:
            ttk.Label(self.scrollable_frame,
                      text="⚠️ No grid parameters defined for this model.",
                      foreground="#d32f2f",
                      font=("Segoe UI", 10, "italic")).pack(pady=10)
            return

        param_grid = model_data["grid"]
        for key, values in param_grid.items():
            box = ttk.LabelFrame(self.scrollable_frame, text=key, padding=5)
            box.pack(fill="x", pady=4)
            entry = ttk.Entry(box, width=45)
            entry.insert(0, str(values))
            entry.pack(padx=4, pady=3)
            self.entries[key] = entry

    def run_search(self):
        """Run grid-based hyperparameter search with progress feedback."""
        model = self.model_var.get()
        metric = self.metric_var.get()

        param_grid = {}
        for k, entry in self.entries.items():
            try:
                param_grid[k] = eval(entry.get())
                if not isinstance(param_grid[k], (list, tuple)):
                    raise ValueError
            except Exception:
                messagebox.showerror("Invalid Input",
                                     f"Parameter {k} must be a list of values, e.g. [50, 100, 200].")
                return

        self.progress_label.config(text="Running search... Please wait ⏳")
        self.progress.start(10)
        self.update_idletasks()

        try:
            best_params, best_score, all_results = HyperparameterSearch.grid_search_IF(
                controller=self.controller,
                X=self.controller.X,
                y=self.controller.y,
                model_name=model,
                metric=metric,
                param_grid=param_grid
            )

            self.progress.stop()
            self.progress_label.config(text="Search complete")

            result_text = f"Best {metric.replace('_',' ').title()}: {best_score:.3f}\n\n📊 Best Parameters:\n"
            for k, v in best_params.items():
                result_text += f" - {k}: {v}\n"

            self.result_label.config(text=result_text)
            self.controller.best_model = (model, best_params)

            messagebox.showinfo("Optimization Complete",
                                f"Best {metric.replace('_',' ').title()}: {best_score:.3f}\n\nBest Parameters:\n{best_params}")

        except Exception as e:
            self.progress.stop()
            self.progress_label.config(text="Error occurred")
            messagebox.showerror("Error", f"Hyperparameter search failed:\n\n{str(e)}")

    def on_close(self):
        """Close the window and reset singleton reference."""
        self.destroy()
        if hasattr(self.parent, "hyper_window"):
            self.parent.hyper_window = None
