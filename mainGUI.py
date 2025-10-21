import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Controller import AnomalyController
from Scores import get_param_grid_for_model

# Internal MVC modules
import  VisualizeWindow
import HyperSearchWindow


# ============================================================
# 🌈 Light Modern Theme
# ============================================================
def apply_modern_style(root: tk.Tk):
    """Light modern style with black text and bold headings."""
    style = ttk.Style(root)
    style.theme_use("clam")

    base_bg = "#ffffff"  # açık gri-mavi zemin
    card_bg = "#F8FAFF"  # beyazımsı kart
    text_fg = "#1A1A1A"  # koyu gri (siyah yerine daha yumuşak)
    accent = "#5cc6a0"  # derin mavi (ana renk)
    accent_hover = "#05afca"  # hover rengi
    secondary = "#95eefc"  # açık mavi (butonlar)
    warning = "#FFB300"  # amber sarısı
    danger = "#E85C5C"  # sıcak kırmızı

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

    # Treeview
    style.configure("Treeview", background="white", foreground="#000000",
                    fieldbackground="white", rowheight=26, font=("Segoe UI", 9))
    style.configure("Treeview.Heading", background=accent, foreground="#000000",
                    font=("Segoe UI", 10, "bold"))

    # Labels
    style.configure("Header.TLabel", background=card_bg, foreground="#000000",
                    font=("Segoe UI", 11, "bold"))


# ============================================================
# 🧩 SmartAnom Main GUI
# ============================================================
class AnomalyGUI:
    """SmartAnom GUI with light theme and aligned panels."""

    def __init__(self, root):
        self.root = root
        self.root.title("SmartAnom")
        self.root.geometry("1200x800+0+0")
        self.root.minsize(1000, 700)
        self.controller = AnomalyController()
        self.hyperparams={}


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

        def _on_mouse_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

        container = self.scrollable_frame

        # ======================================================
        # 📁 Load & Synthetic (Aligned Top)
        # ======================================================
        top_frame = tk.Frame(container, bg="#f8f9fa")
        top_frame.pack(pady=10, padx=15, fill="x")

        upload_frame = tk.Frame(top_frame, bg="#ffffff", bd=1, relief="solid", padx=15, pady=10)
        upload_frame.pack(side="left", fill="x", expand=True, padx=5)

        ttk.Label(upload_frame, text="📁 Upload Dataset", style="Header.TLabel").pack(anchor="w", pady=3)
        ttk.Button(upload_frame, text="Choose File", style="Accent.TButton",
                   command=self.upload_file).pack(side="left", padx=5, pady=5)
        self.file_path_label = ttk.Label(upload_frame, text="No file selected", foreground="#000000")
        self.file_path_label.pack(side="left", padx=10)

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
        # 🧩 3 Aligned Model Panels (shortened height)
        # ======================================================
        bottom_frame = tk.Frame(container, bg="#f8f9fa")
        bottom_frame.pack(pady=5, padx=10, fill="x")

        # Daha kısa ve kompakt paneller
        frame_opts = dict(bg="#ffffff", padx=10, pady=8, relief="solid", bd=1, width=350, height=260)
        if_frame = tk.LabelFrame(bottom_frame, text="Isolation Forest Models",
                                 font=("Segoe UI", 11, "bold"), **frame_opts)
        bench_frame = tk.LabelFrame(bottom_frame, text="Benchmark Models",
                                    font=("Segoe UI", 11, "bold"), **frame_opts)
        metrics_frame = tk.LabelFrame(bottom_frame, text="Performance Metrics",
                                      font=("Segoe UI", 11, "bold"), **frame_opts)

        if_frame.pack(side="left", padx=8, pady=5, fill="both", expand=True)
        bench_frame.pack(side="left", padx=8, pady=5, fill="both", expand=True)
        metrics_frame.pack(side="left", padx=8, pady=5, fill="both", expand=True)

        # --- IF Panel --- #
        self.if_model = tk.StringVar(value="Select")
        for model in ["Isolation Forest", "Extended Isolation Forest", "Generalized Isolation Forest", "SciForest", "FairCutForest"]:
            ttk.Radiobutton(if_frame, text=model, variable=self.if_model, value=model).pack(anchor="w", pady=1)

        ttk.Label(if_frame, text="Score Method", style="Header.TLabel").pack(anchor="w", pady=3)
        self.score_method = tk.StringVar(value="MBAS")
        for score in ["MBAS", "SBAS"]:
            ttk.Radiobutton(if_frame, text=score, variable=self.score_method, value=score).pack(anchor="w", pady=1)

        ttk.Button(if_frame, text="Configure Hyperparameters", style="Accent.TButton",
                   command=self.open_if_hyperparam_window).pack(pady=3, fill="x")
        ttk.Button(if_frame, text="Run IF Model", style="TButton",
                   command=self.run_if_model).pack(pady=3, fill="x")

        # --- Benchmark Panel --- #
        self.benchmark_model = tk.StringVar(value="")
        models = ["Sklearn IF", "Autoencoder", "VAE", "DeepSVDD",
                  "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope"]
        for model in models:
            ttk.Radiobutton(bench_frame, text=model, variable=self.benchmark_model, value=model).pack(anchor="w",
                                                                                                      pady=1)

        ttk.Button(bench_frame, text="Configure Hyperparameters", style="Accent.TButton",
                   command=self.open_benchmark_hyperparam_window).pack(pady=3, fill="x")
        ttk.Button(bench_frame, text="Run Benchmark Model", style="TButton",
                   command=self.run_benchmarks).pack(pady=3, fill="x")

        # --- Metrics Panel --- #
        self.metrics_table = ttk.Treeview(metrics_frame, columns=["Metric", "Value"], show="headings", height=8)
        self.metrics_table.heading("Metric", text="Metric")
        self.metrics_table.heading("Value", text="Value")
        self.metrics_table.pack(fill="both", expand=True, pady=3)
        ttk.Button(metrics_frame, text="Save Results", style="Warning.TButton",
                   command=self.save_results).pack(pady=3, fill="x")

        # ======================================================
        # 🧰 Advanced Tools
        # ======================================================
        advanced_frame = tk.LabelFrame(container, text="Advanced Tools", font=("Segoe UI", 11, "bold"),
                                       bg="#ffffff", padx=15, pady=10, relief="solid", bd=1)
        advanced_frame.pack(pady=10, fill="x", padx=10)

        ttk.Button(advanced_frame, text="Perform Hyperparameter Optimization", style="Accent.TButton",
                   command=self.open_hyper_search_window).pack(side="left", padx=10)
        ttk.Button(advanced_frame, text="Execute Benchmark Evaluation Suite", style="TButton",
                   command=self.visualize_metrics).pack(side="left", padx=10)
        ttk.Button(advanced_frame, text="Generate Explainability Analysis (SHAP)", style="Danger.TButton",
                   command=self.run_shap_explainability).pack(side="left", padx=10)

    # ============================================================
    # 📁 Dataset Handling (unchanged)
    # ============================================================
    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("JSON", "*.json")])
        if file_path:
            self.file_path_label.config(text=file_path)
            try:
                df_head = self.controller.load_dataset(file_path)
                self.show_table(df_head)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def open_synthetic_window(self):
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

    def show_table(self, df):
        # Önce eski tabloyu temizle
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Ana çerçeve
        frame = tk.Frame(self.table_frame, bg="#f8f9fa")
        frame.pack(fill="both", expand=True)

        # Scrollbar’lar
        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")

        # Treeview (tablolu görünüm)
        tree = ttk.Treeview(
            frame,
            columns=list(df.columns),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )

        # Scrollbar bağlantıları
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        # Grid yerleşimi
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Frame içinde grid genişlemeleri
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Sütun başlıkları ve veriler
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=110)

        for _, row in enumerate(df.itertuples(index=False)):
            tree.insert("", "end", values=row)

    # ============================================================
    # Remaining methods (unchanged)
    # ============================================================
    def open_if_hyperparam_window(self):
        model = self.if_model.get()
        model_params = get_param_grid_for_model(model)
        defaults = model_params.get("defaults", {})

        win = tk.Toplevel(self.root)
        win.title(f"{model} Hyperparameters")
        win.geometry("350x500")
        win.configure(bg="#ffffff")

        entries = {}
        for key, val in defaults.items():
            ttk.Label(win, text=key).pack(pady=3)
            e = ttk.Entry(win)
            e.insert(0, str(self.hyperparams.get(model, {}).get(key, val)))
            e.pack()
            entries[key] = e

        ttk.Button(win, text="Save", style="TButton",
                   command=lambda: [self._save_hyperparams(model, entries), win.destroy()]).pack(pady=15, fill="x")

    def _save_hyperparams(self, model, entries):
        self.hyperparams[model] = {k: float(v.get()) if '.' in v.get() else int(v.get()) for k, v in entries.items()}

    def run_if_model(self):
        try:
            model = self.if_model.get()
            score = self.score_method.get()
            params = self.hyperparams.get(model, {})
            metrics = self.controller.run_if_model(model, score,params)
            self.update_metrics(metrics, title=f"{model} ({score}) Results")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def open_benchmark_hyperparam_window(self):
        model = self.benchmark_model.get()
        if not model:
            messagebox.showwarning("No Selection", "Please select a benchmark model first!")
            return

        model_data = get_param_grid_for_model(model)
        defaults = model_data.get("defaults", {})  # <-- buradan alınıyor

        win = tk.Toplevel(self.root)
        win.title(f"{model} Hyperparameters")
        win.geometry("350x520")
        win.configure(bg="#ffffff")

        entries = {}

        for key, val in defaults.items():
            ttk.Label(win, text=key).pack(pady=3)
            e = ttk.Entry(win)
            e.insert(0, str(self.hyperparams.get(model, {}).get(key, val)))
            e.pack()
            entries[key] = e

        ttk.Button(
            win,
            text="Save",
            style="TButton",
            command=lambda: [self._save_benchmark_params(model, entries), win.destroy()]
        ).pack(pady=15, fill="x")


    def _save_benchmark_params(self, model, entries):
        hp = {}
        for key, entry in entries.items():
            val = entry.get()
            if key == "hidden_units":
                hp[key] = [int(x.strip()) for x in val.split(",")]
            elif key in ["learning_rate", "contamination"]:
                hp[key] = float(val)
            elif key in ["epochs", "batch_size", "n_neighbors", "n_trees", "max_samples"]:
                hp[key] = int(val)
            else:
                hp[key] = val
        self.hyperparams[model] = hp

    def run_benchmarks(self):
        model = self.benchmark_model.get()
        try:
            params = self.hyperparams.get(model, {})
            metrics = self.controller.run_benchmark_model(model, params)
            self.update_metrics(metrics, title=f"{model} Results")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_metrics(self, metrics: dict, title: str = "Results"):
        for row in self.metrics_table.get_children():
            self.metrics_table.delete(row)
        self.metrics_table.heading("Metric", text=title)
        for k, v in metrics.items():
            self.metrics_table.insert("", "end", values=[k, str(v)])

    def open_hyper_search_windows(self):
        model = self.if_model.get()
        if not model:
            messagebox.showwarning("Model Selection", "Please select a model first.")
            return

        # popup çağır
        HyperSearchWindow(self.root, model_name=model, controller=self.controller)

    def open_hyper_search_window(self):
        model = self.if_model.get()
        if not model:
            messagebox.showwarning("Model Selection", "Please select a model first.")
            return

        # 🔒 Eğer pencere zaten açıksa yeniden açma
        if hasattr(self, "hyper_window") and self.hyper_window is not None:
            try:
                # Eğer pencere hala açık ise sadece öne getir
                self.hyper_window.lift()
                self.hyper_window.focus()
                return
            except tk.TclError:
                # Pencere kapatılmış ama referans silinmemiş olabilir
                self.hyper_window = None

        # 🚀 Yeni pencere oluştur (ilk kez)
        self.hyper_window = HyperSearchWindow(self.root, model_name=model, controller=self.controller)

    def visualize_metrics(self):
        """Opens model selection popup for visualization."""
        if self.controller.X is None or self.controller.y is None:
            messagebox.showwarning("No Dataset", "Please load or generate a dataset first.")
            return

        # Singleton mantığı (birden fazla popup açılmasın)
        if hasattr(self, "metrics_window") and self.metrics_window is not None:
            try:
                self.metrics_window.lift()
                self.metrics_window.focus()
                return
            except tk.TclError:
                self.metrics_window = None

        self.metrics_window = VisualizeWindow(self.root, self.controller)

    def run_shap_explainability(self):
        try:
            messagebox.showinfo("Explainability", "Starting SHAP explainability analysis...")
            self.controller.run_explainability()
            messagebox.showinfo("Explainability", "SHAP analysis completed. Check the SHAP plot window.")
        except Exception as e:
            messagebox.showerror("Error", f"SHAP analysis failed:\n{str(e)}")

    def save_results(self, file_path: str = None):
        if not file_path:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                for iid in self.metrics_table.get_children():
                    row = self.metrics_table.item(iid)["values"]
                    writer.writerow(row)
            messagebox.showinfo("Success", f"Results saved successfully to: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")


# ============================================================
# Run App
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    apply_modern_style(root)
    app = AnomalyGUI(root)
    root.mainloop()
