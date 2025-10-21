import pandas as pd
import matplotlib.pyplot as plt
from IFModels import gif, eif
from IFModels import FairCutForest
from IFModels.SciForest import SCiForest
from LowCodeTemplates.Datasets.ReadDatasets import load_dataset, dataset_paths
from Scores import SBAS, compute_metrics


def evaluate_sbas(model, X, y, threshold, majority):
    path_len = model.compute_paths_all_tree(X)
    y_pred = SBAS.predict(path_len, threshold=threshold, majority=majority)
    metrics = compute_metrics(y, y_pred)
    df_metrics = pd.DataFrame([metrics])
    df_metrics['Threshold'] = threshold
    df_metrics['Majority'] = majority
    return df_metrics

def gridSearchSBAS(model, X, y, metric='Accuracy'):
    all_results_list = []
    for threshold in thresholds:
        for majority in [3,5,10]:
            result = evaluate_sbas(model, X, y, threshold, majority)
            all_results_list.append(result)
    all_results_df = pd.concat(all_results_list, ignore_index=True)
    all_results_df = all_results_df.sort_values(by=metric, ascending=False)
    best_result = all_results_df.iloc[[0]].copy()
    return best_result


if __name__ == '__main__':
    thresholds = [0.998, 0.99, 0.98, 0.95]
    n_repeats = 1
    sample_size = 256
    nTree = 100
    datasets = {name: (lambda n=name: load_dataset(n)) for name in dataset_paths.keys()}
    results_file = "SBAS_Bench_results.xlsx"
    all_results_total = []

    for ds_name, func in datasets.items():
        X_train, y_train = func()
        print(f"{ds_name} dataset shape: {X_train.shape}, anomalies: {sum(y_train)}")
        n, d = X_train.shape
        phi = min(sample_size, n)

        for repeat in range(n_repeats):
            print(f"  Repeat {repeat + 1}/{n_repeats}")
            clf_IF = eif.iForest(X_train, ntrees=nTree, sample_size=phi, ExtensionLevel=0)
            clf_EIF = eif.iForest(X=X_train, ntrees=nTree, sample_size=phi, ExtensionLevel=1)
            clf_GIF = gif.iForest(X=X_train, ntrees=nTree, sample_size=phi)
            clf_Sci = SCiForest(n_trees=nTree, sample_size=phi, k_planes=1, extension_level="full")
            clf_Sci.fit(X_train)
            clf_fair = FairCutForest(n_trees=nTree, sample_size=phi, k_planes=1, extension_level="full")
            clf_fair.fit(X_train)

            print("Grid searching")
            if_sb = gridSearchSBAS(clf_IF, X_train, y_train)
            eif_sb = gridSearchSBAS(clf_EIF, X_train, y_train)
            gif_sb = gridSearchSBAS(clf_GIF, X_train, y_train)
            sci_sb = gridSearchSBAS(clf_Sci, X_train, y_train)
            fair_sb = gridSearchSBAS(clf_fair, X_train, y_train)

            for df, alg in zip([if_sb, eif_sb, gif_sb, sci_sb, fair_sb],
                               ["iForest", "EIF", "GIF", "SCiForest", "FairCutForest"]):
                df["Algorithm"] = alg
                df["Dataset"] = ds_name
                df["Repeat"] = repeat + 1
                all_results_total.append(df)

    all_results_df = pd.concat(all_results_total, ignore_index=True)

    # Ortalama ve std
    summary_df = all_results_df.groupby(["Dataset", "Algorithm"]).agg(['mean', 'std']).reset_index()
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]

    with pd.ExcelWriter(results_file) as writer:
        all_results_df.to_excel(writer, sheet_name="All_Repeats", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Results and summary saved to {results_file}")


# ================================
# Plot Accuracy
# ================================


if "Accuracy_mean" in summary_df.columns:
    plt.figure(figsize=(10, 6))

    # Benzersiz dataset ve algoritmaları bul
    datasets = summary_df["Dataset"].unique()
    algorithms = summary_df["Algorithm"].unique()

    # Her veri seti için çubuk çiz
    for ds in datasets:
        subset = summary_df[summary_df["Dataset"] == ds]
        accuracies = subset["Accuracy_mean"].values
        plt.bar([f"{ds}-{alg}" for alg in subset["Algorithm"]], accuracies, label=ds)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy (Mean)")
    plt.title("Mean Accuracy Across Datasets and Algorithms (MBAS)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"sbas_accuracy.png")
    plt.show()

else:
    print(" 'accuracy_mean' column not found in summary_df. Check metric names in PerformanceMetrics.")

