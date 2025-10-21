# 🧠 SmartAnom: Sigmoid-Based Anomaly Detection Framework

**SmartAnom** is a modular, low-code / no-code framework for anomaly detection that integrates Isolation Forest (IF) family models, classical algorithms, and deep learning–based methods.  
It introduces **Sigmoid-Based Anomaly Scoring (SBAS)** — a novel scoring approach that improves detection accuracy and robustness compared to traditional mean-based anomaly scoring (MBAS).

---

## 📌 Key Features

- **Unified Interface:** Low-code GUI built with Tkinter/ttk for dataset handling, training, and explainability.
- **Multiple Algorithms:**
  - IF Variants: Isolation Forest, Extended IF, Generalized IF, SciForest, FairCutForest  
  - Classical: One-Class SVM, Local Outlier Factor, Elliptic Envelope  
  - Deep Models: Autoencoder, Variational Autoencoder (VAE), DeepSVDD
- **Novel Scoring:**  
  - `SBAS` (Sigmoid-Based Anomaly Scoring)  
  - `MBAS` (Mean-Based Anomaly Scoring)
- **Hyperparameter Optimization:** Built-in grid search for reproducibility.
- **Explainability:** SHAP-based visualization of model decisions.
- **Reproducibility:** FAIR principles (Findable, Accessible, Interoperable, Reusable).

---

## 🚀 Quick Start

```bash
git clone https://github.com/serpil-ustebay/SmartAnom.git
cd SmartAnom
pip install -r requirements.txt
python main.py
```

---

## 📊 Reproducibility

All experiments, datasets, and results are archived and publicly available.

- 💾 [Zenodo Archive (DOI)](https://doi.org/10.5281/zenodo.17406251)  
- 🧮 [GitHub Repository](https://github.com/serpil-ustebay/SmartAnom)

SmartAnom follows **FAIR** principles to ensure: *Findable • Accessible • Interoperable • Reusable*

---


---

## 📜 License

Released under the **MIT License**.

---

**Author:** Serpil Üstebay  
**Institution:** Istanbul Medeniyet University  
**DOI:** [10.5281/zenodo.17406251](https://doi.org/10.5281/zenodo.17406251)
