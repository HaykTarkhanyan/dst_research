import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, \
    precision_score, recall_score, \
    silhouette_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import pandas as pd
import os
from .config import *

logger = logging.getLogger(__name__)

def get_distance(data, model=None, alg="kmeans", density_radius=None, penalty_rate=penalty_rate):
    labels = data[:, -1]
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2
    if alg == "kmeans":
        distances = - np.min(
            np.linalg.norm(data[:, np.newaxis] - model.cluster_centers_, axis=2), axis=1)
    elif alg == "means_no_clustering":
        coordinates = data[:, :-1]
        midpoints = np.array([coordinates[labels == label].mean(axis=0) for label in unique_labels])
        label_to_midpoint = dict(zip(unique_labels, midpoints))

        distances = np.array([
            np.linalg.norm(coord - label_to_midpoint[label]) for coord, label in zip(coordinates, labels)
        ])
    elif alg == "density_no_clustering":
        distances= calculate_adjusted_density(data, labels, radius=density_radius,
                                              penalty_rate=penalty_rate,
                                              remove_outliers=False, normalize=False)

    elif alg == "dbscan":
        distances= calculate_adjusted_density(data, model.labels_, radius=density_radius,
                                              penalty_rate=penalty_rate,
                                              remove_outliers=False, normalize=False)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    return distances

def remove_outliers_and_normalize(df, distance_column="distance", label_column="labels"):
    outliers = detect_outliers_z_score(df["distance"])
    outlier_set = set(outliers)
    df["outlier"] = df[distance_column].isin(outlier_set)
    df_no_outliers = df[~df["outlier"]]

    # min max scale the df. dont use outliers
    min_val = df_no_outliers.groupby(label_column)[distance_column].min()
    max_val = df_no_outliers.groupby(label_column)[distance_column].max()

    label_min = df[label_column].map(min_val)
    label_max = df[label_column].map(max_val)
    range_val = label_max - label_min

    dist_norm = np.where(
        range_val > 0,
        (df[distance_column].values - label_min.values) / range_val.values,
        0.0
    )
    dist_norm = np.clip(dist_norm, -0.01, 1.01)

    return dist_norm


def evaluate_classifier(*, y_actual, y_clust, dataset="train", print_results=False,
                        purpose="kmeans_eval"):
    TP = sum((y_actual == 1) & (y_clust == 1))
    TN = sum((y_actual == 0) & (y_clust == 0))
    accuracy = (TP + TN) / len(y_actual)
    if purpose == "kmeans_eval":
        if accuracy < 0.5: # swap 1's and 0's
            y_clust = np.array([1 if label == 0 else 0 for label in y_clust])

    TP = sum((y_actual == 1) & (y_clust == 1))
    TN = sum((y_actual == 0) & (y_clust == 0))
    FP = sum((y_actual == 0) & (y_clust == 1))
    FN = sum((y_actual == 1) & (y_clust == 0))

    accuracy = (TP + TN) / len(y_actual)
    denom = TP + 0.5 * (FP + FN)
    f1 = TP / denom if denom > 0 else 0.0
    conf_matrix = np.array([[TN, FP], [FN, TP]])

    if print_results:
        logger.debug(f"Evaluation on {dataset}")
        logger.debug(f"\tAccuracy:  {accuracy:.2f}")
        logger.debug(f"\tF1 Score: {f1:.2f}")
        logger.debug(f"\tConfusion Matrix: \n{conf_matrix}")

    return {"accuracy": accuracy, "f1": f1, "confusion_matrix": conf_matrix}


def evaluate_clustering(df, labels, model=None, alg="kmeans", round_digits=3,
                        print_results=False, dataset="train"):
    silhouette = silhouette_score(df, labels).round(round_digits)
    calinski_harabasz = calinski_harabasz_score(df, labels).round(round_digits)
    inertia = None
    if alg == "kmeans" and dataset=="train":
        inertia = round(model.inertia_,round_digits)


    if print_results:
        logger.debug(f"Evaluation on {dataset}")
        logger.debug(f"\t{silhouette = }")
        logger.debug(f"\t{calinski_harabasz = }")
        if alg == "kmeans" and dataset=="train":
            logger.debug(f"\t{inertia = }")

    return {"silhouette": silhouette, "calinski_harabasz": calinski_harabasz, "inertia": inertia}

def dbscan_predict(model, X):
    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


def calculate_adjusted_density(data, labels, radius, penalty_rate=0.5,
                               remove_outliers=False, normalize=False):
    """
    Calculate the adjusted density of each point in the dataset based on the number of points within a specified radius.
    The density score is penalized if neighboring points belong to a different class.

    Args:
    - data (numpy array): The dataset where each row is a point in space.
    - labels (numpy array): Class labels corresponding to each data point.
    - radius (float): The radius of the ball within which to count neighboring points.
    - penalty_rate (float): Penalty multiplier for points from different classes within the radius.
    - remove_outliers (bool): Whether to remove outliers from the dataset before calculating densities.

    Returns:
    - densities (numpy array): An array where each element is the adjusted density of the corresponding point in 'data'.
    """
    tree = cKDTree(data)
    densities = np.zeros(data.shape[0])

    for i, point in enumerate(data):
        indices = tree.query_ball_point(point, r=radius)
        class_counts = np.sum(labels[indices] != labels[i])
        # Calculate density with penalty for different class points
        densities[i] = len(indices) - 1 - (class_counts * penalty_rate)

    if remove_outliers:
        outliers = detect_outliers_z_score(densities)
        outlier_set = set(outliers)
        densities = np.array([d for d in densities if d not in outlier_set])

    if normalize:
        densities = (densities - densities.min()) / (densities.max() - densities.min())

    if sum(densities) == 0:
        msg = "All densities are zero. Consider changing the radius"
        logger.error(msg)
        raise ValueError(msg)

    return densities

def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    if k is None:
        k = 2 * X.shape[-1] - 1
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()

def run_dbscan(X_scaled, target_clusters=TARGET_CLUSTERS, eps=EPS, min_samples=MIN_SAMPLES, step=STEP, max_eps=MAX_EPS):
    current_clusters = 0
    while current_clusters != target_clusters and eps <= max_eps:
        # DBSCAN with current eps and min_samples
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_

        # Exclude noise labels and count unique clusters
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        current_clusters = len(unique_labels)

        # Check if we have the desired number of clusters
        if current_clusters == target_clusters:
            logger.info(f"Found the desired number of clusters: {current_clusters} at eps={eps}")
            break
        else:
            eps += step

    # If the loop completes without breaking
    if current_clusters != target_clusters:
        logger.warning("Could not find the exact number of desired clusters with the given parameters.")
    else:
        return db

def detect_outliers_z_score(data, threshold=OUTLIER_THRESHOLD_NUM_STD):
    data = np.asarray(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0:
        return []
    z_scores = np.abs((data - mean) / std_dev)
    return data[z_scores > threshold].tolist()

def report_results(y_train, y_train_pred, y_test, y_test_pred, epoch=None, dt=None, losses=None, method=None, dataset=None,
                   name=None, save_results=False, save_path=None, print_results=True,
                   breaks=3, mult_rules=False, clustering_alg=None, label_for_dist=None, plot_folder="plots"):
    #todo add reportin on train
    fig = None
    plot_save_path = None
    if epoch and dt and losses:
        if print_results:
            if not os.path.exists(os.path.join(plot_folder, dataset)):
                os.makedirs(os.path.join(plot_folder, dataset))

            logger.debug(f"Training Time: {dt:.2f}s")
            logger.debug(f"Epochs: {epoch+1}")
            logger.debug(f"Min Loss: {losses[-1]:.3f}")
            fig = plt.figure()
            plt.style.use('ggplot')
            plt.plot(list(range(epoch+1)), losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            name_title = f"{dataset} | clst={clustering_alg} | maf_method={method}"
            plt.title(name_title)
            plot_save_path = os.path.join(plot_folder, dataset, f"{name}.png")
            plt.savefig(plot_save_path)
            plt.close(fig)
            # save to csv

    def eval_classification(y_actual, y_pred, subset="train"):
        accuracy = accuracy_score(y_actual, y_pred)
        precision = precision_score(y_actual, y_pred)
        recall = recall_score(y_actual, y_pred)
        f1 = f1_score(y_actual, y_pred)
        roc_auc = roc_auc_score(y_actual, y_pred)
        avg_precision = average_precision_score(y_actual, y_pred)
        conf_matrix = confusion_matrix(y_actual, y_pred)


        if print_results:
            logger.debug(f"Accuracy:  {accuracy:.2f}")
            logger.debug(f"Precision: {precision:.2f}")
            logger.debug(f"Recall: {recall:.2f}")
            logger.debug(f"F1 Score: {f1:.2f}")
            logger.debug(f"ROC AUC: {roc_auc:.2f}")
            logger.debug(f"Average Precision: {avg_precision:.2f}")
            logger.debug(f"Confusion Matrix: {conf_matrix}")
        return {f"{subset}_accuracy": accuracy, f"{subset}_precision": precision,
                f"{subset}_recall": recall, f"{subset}_f1": f1, f"{subset}_roc_auc": roc_auc,
                f"{subset}_avg_precision": avg_precision}

    res_train = eval_classification(y_train, y_train_pred, subset="train")
    res_test = eval_classification(y_test, y_test_pred, subset="test")

    if save_results:
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        if save_path is None:
            save_path = f"experiments.csv"
        if name is None:
            name = "No name"
        res_row = {"datetime": now, "name": name, "MAF method": method, "dataset": dataset,
                    "breaks": breaks, "mult_rules": mult_rules,
                    "training_time": dt, "epochs": epoch+1,"min_loss": losses[-1],
                    "all_losses": losses, "clustering_alg": clustering_alg,
                    "label_for_dist": label_for_dist}

        res_row.update(res_train)
        res_row.update(res_test)

        res_df = pd.read_csv(save_path) if os.path.exists(save_path) else pd.DataFrame()
        res_df = pd.concat([res_df, pd.DataFrame([res_row])], ignore_index=True)
        res_df.to_csv(save_path, index=False)
        res_row["plot"] = fig
        res_row["plot_save_path"] = plot_save_path
        return res_row


def filter_by_rule(df, rule_lambda, lower_confidence_by_proportion=LOWER_CONFIDENCE_BY_PROPORTION,
                   print_results=False, label_column=LABEL_COL_FOR_DIST):
    """
    Filters a DataFrame based on a given rule lambda function and calculates the confidence score.

    Note:
        The confidence score is calculated as the average distance to the centroid of the most common cluster.
        If the data points belong to the same cluster, the confidence score is the average distance to the centroid.
        If the data points belong to different clusters, the confidence score is the average distance to the centroid of the most common cluster.
        If the confidence score is lowered by proportion, the confidence score is multiplied by the proportion of data points in the most common cluster.

        Since the closer the data points are to the centroid, the better, the confidence score is calculated as 1 - ...
    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        rule_lambda (function): A lambda function that defines the filtering rule.
        lower_confidence_by_proportion (bool, optional): Whether to lower the confidence score by proportion.
            Defaults to True.

    Returns:
        tuple: A tuple containing the confidence scores for each class and uncertainty.

    Example:
        filter_by_rule(df, lambda row: row["x"]>0.5 and row["y"]>0.5)


    """
    required_columns = [label_column, "distance_norm"]
    for column in required_columns:
        assert column in df.columns, f"{column} column not found in DataFrame"
    assert df["distance_norm"].isna().sum() == 0, "distance_norm column contains NaN values"
    assert np.isinf(df["distance_norm"]).sum() == 0, "distance_norm column contains inf values"

    # example is lambda row: row["x"]>0.5 and row["y"]>0.5
    df["rule_applies"] = df.apply(rule_lambda, axis=1)

    df_rule = df[df["rule_applies"]]

    if df_rule.empty:
        return 0, 0, 1 # full uncertainty (this is the identity in the (MAF, dst rule) group)

    num_labels = df_rule[label_column].nunique()
    if print_results:
        logger.debug(f"Number of data points left after filtering: {len(df_rule)}")
        logger.debug(f"Number of clusters left after filtering: {num_labels}")

    most_common_cluster = df_rule[label_column].mode().values[0]
    if print_results: logger.debug(f"Most common cluster: {most_common_cluster}")


    if df_rule[label_column].nunique() == 1:

        if print_results: logger.debug("All data points belong to the same cluster")
        confidence = df_rule["distance_norm"].mean()

        if print_results: logger.debug(f"Confidence: {confidence}")
    else:
        if print_results: logger.debug("Data points belong to different clusters")
        confidence = df_rule[df_rule[label_column] == most_common_cluster]["distance_norm"].mean()
        if print_results: logger.debug(f"Confidence: {confidence}")

        if lower_confidence_by_proportion:
            num_points = len(df_rule[df_rule[label_column] == most_common_cluster])
            if num_points == 0:
                if print_results: logger.debug("No data points in most common cluster")
                raise ValueError("No data points in most common cluster")
            proportion = num_points / len(df_rule)

            confidence = confidence * proportion
            if print_results: logger.debug(f"Confidence after lowering based on proportion: {confidence}")

    assert not np.isnan(confidence), "Confidence is NaN"

    if most_common_cluster == 0:
        return confidence, (1-confidence)/2, (1-confidence)/2
    elif most_common_cluster == 1:
        return (1-confidence)/2, confidence, (1-confidence)/2
    else:
        raise ValueError(f"Most common cluster is {most_common_cluster}, but should be 0 or 1")


def is_categorical(arr, max_cat = 6):
    return len(np.unique(arr[~np.isnan(arr)])) <= max_cat


def normalize(a, b, c):
    a = 0 if a < 0 else 1 if a > 1 else a
    b = 0 if b < 0 else 1 if b > 1 else b
    c = 0 if c < 0 else 1 if c > 1 else c
    n = float(a + b + c)
    return a/n, b/n, c/n


def one_hot(n, k):
    a = np.zeros((len(n), k))
    for i in range(len(n)):
        a[i, int(n[i])] = 1
    return a


def h_center(z):
    return np.exp(- z * z)


def h_right(z):
    return (1 + np.tanh(z - 1))/2


def h_left(z):
    return (1 - np.tanh(z + 1))/2
