"""
Wine market segmentation using unsupervised machine learning.
Code evidence for Task 3: train/test split, baseline vs tuned models,
hyperparameter tuning, visual outputs and performance metrics.
"""

import os

# Limit numerical library threads so the script runs reliably in marking environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# Set a random state for reproducibility and define output directories for figures and data.
RANDOM_STATE = 42
OUTPUT_DIR = "mlvda_outputs"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Define a function to evaluate clustering performance, handling DBSCAN noise points safely.
def evaluate_clustering(X_values, labels, y_true=None, density_based=False):
    """Return clustering metrics, handling DBSCAN noise points safely."""
    labels = np.asarray(labels)
    n_noise = int(np.sum(labels == -1)) if density_based else 0
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    coverage = ((len(labels) - n_noise) / len(labels)) * 100
    noise_pct = (n_noise / len(labels)) * 100

# Initialize metrics with NaN for those that may not be computable due to noise or single cluster issues.
    metrics = {
        "Number of Clusters": n_clusters,
        "Number of Noise Points": n_noise,
        "Coverage (%)": coverage,
        "Noise (%)": noise_pct,
        "Silhouette Score": np.nan,
        "Davies-Bouldin Index": np.nan,
        "Adjusted Rand Index": np.nan,
        "Normalized Mutual Information": np.nan,
    }
# For density-based models, exclude noise points from metric calculations; for others, include all points.
    mask = labels != -1 if density_based else np.ones(len(labels), dtype=bool)
    X_eval = X_values[mask]
    labels_eval = labels[mask]
    y_eval = np.asarray(y_true)[mask] if y_true is not None else None

# Silhouette Score and Davies-Bouldin Index require at least 2 clusters; ARI and NMI also require at least 2 true classes.
    if len(labels_eval) > 1 and len(set(labels_eval)) >= 2:
        metrics["Silhouette Score"] = silhouette_score(X_eval, labels_eval)
        metrics["Davies-Bouldin Index"] = davies_bouldin_score(X_eval, labels_eval)
        if y_eval is not None and len(set(y_eval)) >= 2:
            metrics["Adjusted Rand Index"] = adjusted_rand_score(y_eval, labels_eval)
            metrics["Normalized Mutual Information"] = normalized_mutual_info_score(y_eval, labels_eval)

    return metrics

# Define a helper function to save figures and display them in interactive environments, while ensuring compatibility with headless environments.
def save_and_show(fig, filename):
    """Save figures used in the report and display them in interactive environments."""
    path = os.path.join(FIGURE_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    # In Colab/Jupyter this displays the graph; in headless/marking environments it is safely skipped.
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)

# Define a function to create scatter plots of PCA results, colouring by specified labels, and including variance explained in axis labels.
def scatter_pca(data, hue_column, title, filename, legend_title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(data[hue_column].unique(), key=lambda value: str(value)):
        subset = data[data[hue_column] == label]
        ax.scatter(subset["PC1"], subset["PC2"], label=str(label), s=45, alpha=0.85) # Use sorted labels for consistent legend ordering and convert to string for better display.
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pc1_var * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_var * 100:.2f}% variance)")
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    save_and_show(fig, filename) # Save the figure with the specified filename in the figures directory.


# 1. Load and document dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="target")
target_names = pd.Series([wine.target_names[i] for i in y], name="target_name") # Map numeric labels to actual wine class names for better interpretability in outputs.

wine_dataset = X.copy()
wine_dataset["target"] = y
wine_dataset["target_name"] = target_names
wine_dataset.to_csv(os.path.join(DATA_DIR, "wine_dataset_used.csv"), index=False)

# Print dataset overview, including shape, feature count, missing values, and descriptive statistics.
print("\nDATASET OVERVIEW")
print("Dataset shape:", wine_dataset.shape)
print("Feature count:", X.shape[1])
print("\nMissing values per column:")
print(wine_dataset.isnull().sum())
print("\nDescriptive statistics:")
print(wine_dataset.describe().round(3))

# 2. Preprocess features and create a train/test split for robustness checking
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Print the shapes of the training and testing sets, and clarify that labels are only used for stratification and post-hoc validation, not for clustering training.
print("\nTRAIN/TEST SPLIT")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Note: labels are used only for stratification and post-hoc validation, not for clustering training.")

# 3. Principal Component Analysis (PCA) for visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pc1_var = pca.explained_variance_ratio_[0]
pc2_var = pca.explained_variance_ratio_[1]
total_var = pc1_var + pc2_var

# Create a DataFrame for PCA results, including the actual wine class labels and names for better interpretability in visualisations.
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Actual_Wine_Label"] = y
pca_df["Actual_Wine_Name"] = target_names

# Print the explained variance for the first two principal components to provide context for the visualisations and interpretability of the PCA plots.
print("\nPCA EXPLAINED VARIANCE")
print(f"PC1: {pc1_var:.4f}")
print(f"PC2: {pc2_var:.4f}")
print(f"Total explained variance: {total_var:.4f}")

# Visualise the PCA results coloured by actual wine class labels and names to provide an intuitive understanding of the dataset's structure and the separability of the classes in the PCA space.
scatter_pca(
    pca_df,
    "Actual_Wine_Name",
    "Figure 1: PCA plot coloured by actual wine class",
    "figure_1_pca_actual_wine_classes.png",
    "Actual class",
)

# 4. K-Means hyperparameter tuning
k_values = list(range(2, 11))
inertia_values = []
kmeans_silhouette_values = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia_values.append(model.inertia_)
    kmeans_silhouette_values.append(silhouette_score(X_scaled, labels))

# Save K-Means tuning results to a CSV for reporting and print the results in a readable format.
kmeans_tuning_df = pd.DataFrame({
    "K": k_values,
    "Inertia": inertia_values,
    "Silhouette Score": kmeans_silhouette_values,
})
kmeans_tuning_df.to_csv(os.path.join(DATA_DIR, "kmeans_hyperparameter_tuning.csv"), index=False)
print("\nK-MEANS HYPERPARAMETER TUNING")
print(kmeans_tuning_df.round(4)) # Print the K-Means tuning results with rounded values for better readability in the console output.

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, inertia_values, marker="o")
ax.set_title("Figure 2: K-Means Elbow Method")
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Inertia")
ax.set_xticks(k_values)
ax.grid(True, alpha=0.3)
save_and_show(fig, "figure_2_kmeans_elbow_method.png") # Save the elbow method plot with the specified filename in the figures directory.

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, kmeans_silhouette_values, marker="o")
ax.set_title("Figure 3: K-Means Silhouette Scores Across K Values")
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.set_xticks(k_values)
ax.grid(True, alpha=0.3)
save_and_show(fig, "figure_3_kmeans_silhouette_scores.png") # Save the silhouette scores plot with the specified filename in the figures directory.

# 5. Baseline vs tuned models
model_specs = [
    ("K-Means", "Baseline", {"n_clusters": 2, "n_init": 10}),
    ("K-Means", "Tuned", {"n_clusters": 3, "n_init": 10}),
    ("Agglomerative Clustering", "Baseline", {"n_clusters": 2, "linkage": "ward"}),
    ("Agglomerative Clustering", "Tuned", {"n_clusters": 3, "linkage": "ward"}),
    ("DBSCAN", "Baseline", {"eps": 1.5, "min_samples": 5}),
    ("DBSCAN", "Tuned", {"eps": 2.25, "min_samples": 10}),
]
# We will store the results for the full dataset comparison and the train/test split comparison in separate lists of dictionaries, which will then be converted to DataFrames for saving and reporting.
full_comparison_rows = []
split_comparison_rows = []

# Loop through each model specification, fit the model to the full dataset and the train/test splits, evaluate the clustering performance using the defined metrics, and store the results in the respective lists for later conversion to DataFrames and saving as CSV files.
for algorithm, version, params in model_specs:
    density_based = algorithm == "DBSCAN"

# Fit the specified clustering model to the full dataset and the train/test splits, and predict cluster labels for each. For K-Means, we fit separate models for the full dataset and the train/test splits to ensure that the clustering is based solely on the training data for the split evaluation, while for Agglomerative Clustering and DBSCAN we can fit directly since they do not rely on centroids.
    if algorithm == "K-Means":
        full_model = KMeans(random_state=RANDOM_STATE, **params)
        full_labels = full_model.fit_predict(X_scaled)
        train_model = KMeans(random_state=RANDOM_STATE, **params)
        train_labels = train_model.fit_predict(X_train)
        test_labels = train_model.predict(X_test)

        # For K-Means, we fit a separate model on the training data and use it to predict the test labels to ensure that the clustering is based solely on the training data for the split evaluation, while for Agglomerative Clustering and DBSCAN we can fit directly since they do not rely on centroids.
    elif algorithm == "Agglomerative Clustering":
        full_labels = AgglomerativeClustering(**params).fit_predict(X_scaled)
        train_labels = AgglomerativeClustering(**params).fit_predict(X_train)
        test_labels = AgglomerativeClustering(**params).fit_predict(X_test)

        # For Agglomerative Clustering, we can fit directly on the full dataset and the train/test splits since it does not rely on centroids, and it will produce consistent cluster labels based on the hierarchical structure of the data.
    else:
        full_labels = DBSCAN(**params).fit_predict(X_scaled)
        train_labels = DBSCAN(**params).fit_predict(X_train)
        test_labels = DBSCAN(**params).fit_predict(X_test)

    full_metrics = evaluate_clustering(X_scaled, full_labels, y, density_based=density_based)
    train_metrics = evaluate_clustering(X_train, train_labels, y_train, density_based=density_based)
    test_metrics = evaluate_clustering(X_test, test_labels, y_test, density_based=density_based)

    if algorithm == "K-Means" and version == "Tuned":
        pca_df["KMeans_Cluster"] = full_labels
    if algorithm == "Agglomerative Clustering" and version == "Tuned":
        pca_df["Agglomerative_Cluster"] = full_labels
    if algorithm == "DBSCAN" and version == "Tuned":
        pca_df["DBSCAN_Cluster"] = full_labels

# Create a text representation of the parameters used for better readability in the output tables, and store the results in the full comparison and split comparison lists for later conversion to DataFrames and saving as CSV files.
    param_text = ", ".join([f"{key}={value}" for key, value in params.items()])
    full_comparison_rows.append({
        "Algorithm": algorithm,
        "Version": version,
        "Parameters Used": param_text,
        **full_metrics,
    })
    split_comparison_rows.extend([
        {"Algorithm": algorithm, "Version": version, "Split": "Training", **train_metrics},
        {"Algorithm": algorithm, "Version": version, "Split": "Testing", **test_metrics},
    ])

# Convert the lists of dictionaries to DataFrames, round the metric values for better readability, save them as CSV files for reporting, and print the comparison tables in a readable format.
full_comparison_df = pd.DataFrame(full_comparison_rows).round(4)
split_comparison_df = pd.DataFrame(split_comparison_rows).round(4)
full_comparison_df.to_csv(os.path.join(DATA_DIR, "baseline_vs_tuned_full_dataset_comparison.csv"), index=False)
split_comparison_df.to_csv(os.path.join(DATA_DIR, "train_test_baseline_tuned_comparison.csv"), index=False)

# Print the comparison tables for the full dataset and the train/test splits, showing the performance metrics for each model version in a readable format.
print("\nBASELINE VS TUNED MODEL COMPARISON - FULL DATASET")
print(full_comparison_df)
print("\nTRAINING/TESTING SPLIT COMPARISON")
print(split_comparison_df)

# 6. Visualise baseline vs tuned and train/test outputs
valid_full = full_comparison_df.dropna(subset=["Silhouette Score"]).copy()
valid_full["Model Version"] = valid_full["Algorithm"] + "\n" + valid_full["Version"]

# Create bar charts to compare the silhouette scores of the baseline vs tuned models for the full dataset, and to compare the silhouette scores across the training and testing splits for each model version. The charts are saved with specified filenames in the figures directory.
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(valid_full["Model Version"], valid_full["Silhouette Score"])
ax.set_title("Figure 3a: Baseline vs Tuned Silhouette Score Comparison")
ax.set_xlabel("Model version")
ax.set_ylabel("Silhouette Score")
ax.tick_params(axis="x", rotation=30)
ax.grid(axis="y", alpha=0.3)
save_and_show(fig, "figure_3a_baseline_vs_tuned_silhouette.png")

# For the train/test split comparison, we create a new column that combines the algorithm, version, and split information for better readability on the x-axis, and then create a bar chart to compare the silhouette scores across the training and testing splits for each model version. The chart is saved with a specified filename in the figures directory.
valid_split = split_comparison_df.dropna(subset=["Silhouette Score"]).copy()
valid_split["Model Version Split"] = (
    valid_split["Algorithm"] + "\n" + valid_split["Version"] + " - " + valid_split["Split"]
)
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(valid_split["Model Version Split"], valid_split["Silhouette Score"])
ax.set_title("Figure 3b: Training and Testing Split Silhouette Comparison")
ax.set_xlabel("Model version and split")
ax.set_ylabel("Silhouette Score")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.3)
save_and_show(fig, "figure_3b_train_test_silhouette_comparison.png") # Save the train/test silhouette comparison plot with the specified filename in the figures directory.

# 7. Final model visualisations
scatter_pca(
    pca_df,
    "KMeans_Cluster",
    "Figure 4: PCA plot coloured by tuned K-Means clusters",
    "figure_4_pca_kmeans_clusters.png",
    "K-Means cluster",
)

# Create scatter plots of the PCA results coloured by the tuned Agglomerative Clustering and DBSCAN cluster labels, including the variance explained in the axis labels, and save the figures with specified filenames in the figures directory.
scatter_pca(
    pca_df,
    "Agglomerative_Cluster",
    "Figure 5: PCA plot coloured by tuned Agglomerative clusters",
    "figure_5_pca_agglomerative_clusters.png",
    "Agglomerative cluster",
)

scatter_pca(
    pca_df,
    "DBSCAN_Cluster",
    "Figure 6: PCA plot coloured by tuned DBSCAN clusters",
    "figure_6_pca_dbscan_clusters.png",
    "DBSCAN cluster (-1 = noise)",
)

final_model_comparison = full_comparison_df[
    (full_comparison_df["Version"] == "Tuned")
].copy()
final_model_comparison.to_csv(os.path.join(DATA_DIR, "final_tuned_model_comparison.csv"), index=False)
pca_df.to_csv(os.path.join(DATA_DIR, "wine_pca_cluster_results.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(final_model_comparison["Algorithm"], final_model_comparison["Silhouette Score"])
ax.set_title("Figure 7: Tuned Model Silhouette Score Comparison")
ax.set_ylabel("Silhouette Score")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
save_and_show(fig, "figure_7_tuned_model_silhouette_comparison.png")

# Create a bar chart to compare the Davies-Bouldin Index of the tuned models for the full dataset, and save the figure with a specified filename in the figures directory.
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(final_model_comparison["Algorithm"], final_model_comparison["Davies-Bouldin Index"])
ax.set_title("Figure 8: Tuned Model Davies-Bouldin Index Comparison")
ax.set_ylabel("Davies-Bouldin Index")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
save_and_show(fig, "figure_8_tuned_model_davies_bouldin_comparison.png")

# Create a bar chart to compare the coverage percentage of the tuned models for the full dataset, and save the figure with a specified filename in the figures directory.
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(final_model_comparison["Algorithm"], final_model_comparison["Coverage (%)"])
ax.set_title("Figure 9: Tuned Model Coverage Comparison")
ax.set_ylabel("Coverage (%)")
ax.set_ylim(0, 110)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
save_and_show(fig, "figure_9_tuned_model_coverage_comparison.png") # Save the coverage comparison plot with the specified filename in the figures directory.

# Print the final comparison of the tuned models for the full dataset, showing the performance metrics for each model version in a readable format.
print("\nFINAL TUNED MODEL COMPARISON")
print(final_model_comparison.round(4))
print("\nAll figures and CSV outputs have been saved in the mlvda_outputs folder.")
