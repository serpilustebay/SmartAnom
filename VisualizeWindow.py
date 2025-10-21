import tkinter as tk
from tkinter import ttk, messagebox


class VisualizeWindow(tk.Toplevel):
    """Popup: select models to run and visualize accuracy chart."""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.title("📊 Model Selection for Visualization")
        self.geometry("400x500")
        self.configure(bg="#f8f9fa")
        self.controller = controller
        self.parent = parent

        ttk.Label(self, text="Select Models to Run", font=("Segoe UI", 12, "bold")).pack(pady=10)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, pady=5)

        # Model listesi
        self.model_vars = {}
        model_list = [
            "Isolation Forest", "Extended Isolation Forest", "Generalized Isolation Forest",
            "SciForest", "FairCutForest",
            "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope",
            "Autoencoder", "VAE", "DeepSVDD"
        ]
        for model in model_list:
            var = tk.BooleanVar(value=True)
            self.model_vars[model] = var
            tk.Checkbutton(frame, text=model, variable=var, bg="#f8f9fa",
                           font=("Segoe UI", 10)).pack(anchor="w", padx=25, pady=2)

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(self, text="Run & Visualize", style="Accent.TButton",
                   command=self.run_selected_models).pack(pady=10)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=300)
        self.progress.pack(pady=5)
        self.status_label = ttk.Label(self, text="")
        self.status_label.pack()

    def run_selected_models(self):
        """Seçilen modelleri çalıştırır, Accuracy grafiğini gösterir ve tüm metrikleri Excel + PNG olarak kaydeder."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import pandas as pd
        import os

        selected_models = [m for m, v in self.model_vars.items() if v.get()]
        if not selected_models:
            messagebox.showwarning("No Selection", "Please select at least one model.")
            return

        if self.controller.X is None or self.controller.y is None:
            messagebox.showwarning("No Dataset", "Please load or generate a dataset first.")
            return

        # Progress başlat
        self.progress.start(10)
        self.status_label.config(text="Running selected models... Please wait ⏳")
        self.update_idletasks()

        all_results = []
        try:
            # --- 1️⃣ Modelleri sırayla çalıştır
            for model in selected_models:
                try:
                    print(f"[Visualize] Running {model}...")
                    if model in ["Isolation Forest", "Extended Isolation Forest",
                                 "Generalized Isolation Forest", "SciForest", "FairCutForest"]:
                        metrics = self.controller.run_if_model(model, "SBAS", {})
                    else:
                        metrics = self.controller.run_benchmark_model(model, {})

                    result = {
                        "Model": model,
                        "Accuracy": metrics.get("Accuracy", 0),
                        "Precision": metrics.get("Precision", 0),
                        "Recall": metrics.get("Recall", 0),
                        "F1 Score": metrics.get("F1 Score", 0)
                    }
                    all_results.append(result)
                except Exception as e:
                    print(f"[Visualize] ⚠️ {model} failed: {e}")
                    all_results.append({
                        "Model": model,
                        "Accuracy": 0, "Precision": 0, "Recall": 0, "F1 Score": 0
                    })

            # --- 2️⃣ DataFrame oluştur
            df = pd.DataFrame(all_results)
            print(df)

            # --- 3️⃣ Kaydetme klasörünü oluştur
            os.makedirs("../SmartAnom/Results", exist_ok=True)
            excel_path = os.path.join("../SmartAnom/Results", "Model_Metrics_Results.xlsx")
            plot_path = os.path.join("../SmartAnom/Results", "Model_Accuracy_Chart.png")

            # --- 4️⃣ Excel olarak kaydet (tüm metriklerle)
            df.to_excel(excel_path, index=False)
            print(f"📁 Metrics saved to {excel_path}")

            # --- 5️⃣ Bar Chart (Accuracy)
            models = df["Model"].tolist()
            accuracies = df["Accuracy"].tolist()

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(models, accuracies, color="#2196F3", edgecolor="black")
            ax.set_title("Model Accuracy Comparison", fontsize=13, weight="bold", color="#000000")
            ax.set_xlabel("Models", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_ylim(0, 1.05)
            plt.xticks(rotation=45, ha="right")

            # Barların üstüne değerleri yaz
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{acc:.2f}", ha="center", va="bottom", fontsize=9, color="#000000")

            plt.tight_layout()

            # --- 6️⃣ Grafiği PNG olarak kaydet
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"🖼️ Plot saved to {plot_path}")

            # --- 7️⃣ Tkinter popup içinde göster
            chart_frame = ttk.Frame(self)
            chart_frame.pack(fill="both", expand=True, pady=10)
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # --- 8️⃣ Progress durdur, bilgi mesajı
            self.progress.stop()
            self.status_label.config(text=f"✅ Completed! Saved to:\n{excel_path}\n{plot_path}")
            messagebox.showinfo(
                "Visualization Complete",
                f"Results saved to:\n\n{excel_path}\n{plot_path}"
            )

        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="❌ Error occurred.")
            messagebox.showerror("Error", f"Visualization failed:\n{str(e)}")


