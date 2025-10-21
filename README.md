# SmartAnom: Sigmoid-Based Anomaly Detection in Low-Code/No-Code Frameworks

SmartAnom is an open-source, low-code / no-code anomaly detection framework that integrates classical, tree-based, and deep learning models under a modular and reproducible architecture.  
It introduces the **Sigmoid-Based Anomaly Scoring (SBAS)** mechanism — a novel approach that transforms Isolation Forest path lengths into binary anomaly indicators using sigmoid functions and majority voting.

---

## Key Features

- **Low-Code/No-Code GUI:**  
  Built with Tkinter and ttk to enable dataset upload, model selection, parameter tuning, and performance visualization without coding.

- **Integrated Algorithms:**  
  - Tree-based: Isolation Forest (IF), Extended IF (EIF), Generalized IF (GIF), SciForest, FairCutForest  
  - Classical: One-Class SVM, Local Outlier Factor (LOF), Elliptic Envelope  
  - Deep Models: Autoencoder (AE), Variational Autoencoder (VAE), DeepSVDD

- **Custom Scoring Mechanisms:**  
  - `Classic`: Mean-based anomaly scoring  
  - `SBAS`: Sigmoid-Based Anomaly Scoring (proposed method)  
  - `MBAS`: Mean-Based Anomaly Scoring (baseline comparison)

- **Explainability:**  
  SHAP-based visual explanations for feature importance and model interpretability.

- **Optimization Module:**  
  Grid/Random Search support for automatic hyperparameter tuning.

---

## 🧠 Architecture Overview

```
SmartAnom/

├── mainGUI.py              # GUI controller
├── Controller.py           #Controller LAaer
├── Views/                  # All GUI components (Tkinter/ttk)
├── Model/                  # IF, EIF, GIF, and deep and classical models
├── DataLayer/              # Dataset loading and synthetic data generation
├── Optimization/           # Hyperparameter search (Grid )
├── ExplainabilityModels/   # Explainability module (SHAP)

```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or newer
- Required libraries:
  ```bash
  pip install -r requirements.txt
  ```
  (If `requirements.txt` not provided yet, install manually:)
  ```bash
  pip install numpy pandas scikit-learn matplotlib shap tensorflow
  ```

---

## 🚀 Usage

Run SmartAnom from the main script:

```bash
python mainGUI.py
```

Through the GUI, you can:
- Upload datasets (CSV/Excel)
- Generate synthetic datasets (`moons`, `blobs`, `spiral`, etc.)
- Choose models and scoring methods
- Run optimization
- Visualize F1, Accuracy, Precision, Recall, and SHAP plots

---

## 📊 Example Visualization

| Model | Score Type | Accuracy | F1 | Precision | Recall |
|:------|:-----------|:---------|:--|:-----------|:--------|
| IF | Classic | 0.91 | 0.83 | 0.79 | 0.87 |
| EIF | SBAS | **0.95** | **0.90** | **0.88** | **0.92** |
| GIF | MBAS | 0.93 | 0.85 | 0.84 | 0.86 |

---

## 🧾 Citation

If you use SmartAnom in your research, please cite:

> Üstebay, S. (2025). *SmartAnom: Sigmoid-Based Anomaly Detection in Low-Code/No-Code Frameworks.*  
> Zenodo DOI: [https://doi.org/10.5281/zenodo.17376652](https://doi.org/10.5281/zenodo.17376652)

---

## Reproducibility

All experiments, datasets, and results are publicly available on Zenodo and GitHub.  
SmartAnom follows FAIR principles to ensure that experiments are **Findable**, **Accessible**, **Interoperable**, and **Reusable**.

- 💾 [Zenodo Archive (DOI)](https://doi.org/10.5281/zenodo.17376652)  
- 🧮 [GitHub Repository](https://github.com/serpilustebay/SmartAnom)  
- 📁 Reproducible capsule with datasets and scripts in `Low-Code Templates/` directory.

---

## 🪪 License

SmartAnom is released under the **MIT License** — free for academic and commercial use.  
See the [LICENSE](LICENSE) file for details.

---

## 🧭 Contact

**Serpil Üstebay**  
Computer Engineering Department, Istanbul Medeniyet University  
📧 serpil.ustebay@medeniyet.edu.tr 
🌐 [https://github.com/serpilustebay/SmartAnom](https://github.com/serpilustebay/SmartAnom)
