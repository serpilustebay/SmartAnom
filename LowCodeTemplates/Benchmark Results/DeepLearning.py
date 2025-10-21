import pandas as pd
from  Models import BenchmarkModels
from LowCodeTemplates.Datasets.ReadDatasets import load_dataset, dataset_paths



excel_file = "Deep_results.xlsx"
all_results_total = []

datasets = {name: (lambda n=name: load_dataset(n)) for name in dataset_paths.keys()}


# --- Loop over datasets and models --- #
for ds_name, func in datasets.items():
    X_train, y_train = func()
    print(f"{ds_name} dataset shape: {X_train.shape}, anomalies: {sum(y_train)}")

    # List of models to run
    models_to_run = ["One-Class SVM", "Local Outlier Factor", "Elliptic Envelope", "Autoencoder", "VAE", "DeeSVDD"]

    for model_name in models_to_run:
        # Run the model
        metrics = BenchmarkModels.run(X_train, y_train, model_name)
        # metrics should be a dict with keys: "Accuracy", "MCC", "Precision", etc.

        # Append results to the list
        all_results_total.append([
            model_name,
            ds_name,
            metrics.get("Accuracy", None),
            metrics.get("MCC", None),
            metrics.get("Precision", None),
            metrics.get("Recall", None),
            metrics.get("F1 Score", None),
            metrics.get("Specificity", None),
            metrics.get("False Positive Rate", None),
            metrics.get("False Negative Rate", None)
        ])

# --- Create DataFrame --- #
columns = ["Algorithm", "Dataset", "Accuracy", "MCC", "Precision", "Recall", "F1 Score", "Specificity",
           "False Positive Rate", "False Negative Rate"]
df_results = pd.DataFrame(all_results_total, columns=columns)

# --- Write to Excel --- #
df_results.to_excel(excel_file, index=False)
print(f"Results saved to: {excel_file}")


