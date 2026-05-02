"""Ablation studies for clustering validation."""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from config.settings import N_CLUSTERS, RANDOM_STATE, SAMPLE_SIZE


def euclidean_dist(x, y):
    """Euclidean distance between two arrays."""
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))


def dtw_distance(x, y):
    """DTW distance using fastdtw."""
    d, _ = fastdtw(x, y, dist=euclidean_dist)
    return d


def compute_euclidean_kmeans(
    X, n_clusters=N_CLUSTERS, max_iter=50, random_state=RANDOM_STATE
):
    """Euclidean K-Means for ablation comparison."""
    from sklearn.cluster import KMeans

    # Pad sequences to same length for Euclidean distance
    max_len = max(len(x) for x in X)
    X_padded = np.array(
        [np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in X]
    )
    X_filled = np.nan_to_num(X_padded, nan=np.nanmean(X_padded))

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init=10
    )
    labels = kmeans.fit_predict(X_filled)

    return labels, kmeans.cluster_centers_


def compute_dtw_kmeans_simple(
    X, n_clusters=N_CLUSTERS, max_iter=50, random_state=RANDOM_STATE
):
    """Simple DTW K-Means for ablation comparison."""
    np.random.seed(random_state)
    idx = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)

    for it in range(max_iter):
        distances = np.zeros((len(X), n_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(centers):
                distances[i, j] = dtw_distance(x, c)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = [
            np.mean([X[i] for i in range(len(X)) if labels[i] == k], axis=0)
            for k in range(n_clusters)
        ]

    return labels, np.array(centers)


def compute_silhouette_dtw(X, labels):
    """Compute silhouette score using DTW distance."""
    n = len(X)
    if n > 1000:
        # Subsample for efficiency
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(n, 1000, replace=False)
        X_sub = [X[i] for i in idx]
        labels_sub = labels[idx]
    else:
        X_sub = X
        labels_sub = labels

    n_sub = len(X_sub)
    silhouette_vals = np.zeros(n_sub)

    for i in range(n_sub):
        own_cluster = labels_sub == labels_sub[i]
        other_clusters = ~own_cluster

        if np.sum(own_cluster) <= 1 or np.sum(other_clusters) == 0:
            silhouette_vals[i] = 0
            continue

        a_i = np.mean(
            [
                dtw_distance(X_sub[i], X_sub[j])
                for j in range(n_sub)
                if own_cluster[j] and j != i
            ]
        )
        b_i = min(
            [
                np.mean(
                    [
                        dtw_distance(X_sub[i], X_sub[j])
                        for j in range(n_sub)
                        if other_clusters[j] and labels_sub[j] == k
                    ]
                )
                for k in np.unique(labels_sub)
                if k != labels_sub[i]
            ]
        )

        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    return np.mean(silhouette_vals)


def run_k_ablation(X_sample, k_values=[3, 4, 5, 6]):
    """Ablation study for K value selection."""
    print("\n" + "=" * 60)
    print("Ablation: K Value Selection")
    print("=" * 60)

    results = {}
    for k in k_values:
        print(f"\n  Testing K={k}...")
        labels, centers = compute_dtw_kmeans_simple(X_sample, n_clusters=k, max_iter=20)

        # Compute metrics
        max_len = max(len(x) for x in X_sample)
        X_padded = np.array(
            [np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in X_sample]
        )
        X_filled = np.nan_to_num(X_padded, nan=np.nanmean(X_padded))

        silhouette = silhouette_score(X_filled, labels, metric="euclidean")
        calinski = calinski_harabasz_score(X_filled, labels)

        results[k] = {
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "labels": labels,
        }
        print(f"    Silhouette: {silhouette:.4f}")
        print(f"    Calinski-Harabasz: {calinski:.2f}")

    best_k = max(results.keys(), key=lambda k: results[k]["silhouette"])
    print(f"\n  Best K by Silhouette: {best_k}")

    return results, best_k


def run_distance_ablation(X_sample):
    """Ablation study for distance metric comparison."""
    print("\n" + "=" * 60)
    print("Ablation: Distance Metric Comparison")
    print("=" * 60)

    max_len = max(len(x) for x in X_sample)
    X_padded = np.array(
        [np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in X_sample]
    )
    X_filled = np.nan_to_num(X_padded, nan=np.nanmean(X_padded))

    # Euclidean K-Means
    print("\n  Running Euclidean K-Means...")
    labels_euc, centers_euc = compute_euclidean_kmeans(X_sample)
    silhouette_euc = silhouette_score(X_filled, labels_euc, metric="euclidean")
    calinski_euc = calinski_harabasz_score(X_filled, labels_euc)

    # DTW K-Means
    print("\n  Running DTW K-Means...")
    labels_dtw, centers_dtw = compute_dtw_kmeans_simple(X_sample)
    silhouette_dtw = compute_silhouette_dtw(X_sample, labels_dtw)

    results = {
        "euclidean": {
            "silhouette": silhouette_euc,
            "calinski_harabasz": calinski_euc,
            "labels": labels_euc,
        },
        "dtw": {
            "silhouette": silhouette_dtw,
            "labels": labels_dtw,
        },
    }

    print(f"\n  Euclidean K-Means:")
    print(f"    Silhouette: {silhouette_euc:.4f}")
    print(f"    Calinski-Harabasz: {calinski_euc:.2f}")
    print(f"\n  DTW K-Means:")
    print(f"    Silhouette (DTW): {silhouette_dtw:.4f}")

    return results


def run_bootstrap_stability(X_sample, n_clusters=N_CLUSTERS, n_bootstrap=10):
    """Bootstrap stability analysis for clustering validation."""
    print("\n" + "=" * 60)
    print("Bootstrap Stability Analysis")
    print("=" * 60)

    n_samples = len(X_sample)
    agreement_matrix = np.zeros((n_samples, n_samples))

    for b in range(n_bootstrap):
        print(f"  Bootstrap iteration {b+1}/{n_bootstrap}...")
        np.random.seed(RANDOM_STATE + b)

        # Sample with replacement
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_sample[bootstrap_idx]

        # Run clustering
        labels_boot, _ = compute_dtw_kmeans_simple(
            X_boot, n_clusters=n_clusters, max_iter=20
        )

        # Update agreement matrix
        for i in range(n_samples):
            for j in range(n_samples):
                if (
                    bootstrap_idx[i] in bootstrap_idx
                    and bootstrap_idx[j] in bootstrap_idx
                ):
                    idx_i = np.where(bootstrap_idx == i)[0][0]
                    idx_j = np.where(bootstrap_idx == j)[0][0]
                    if labels_boot[idx_i] == labels_boot[idx_j]:
                        agreement_matrix[i, j] += 1

    # Compute stability score
    agreement_matrix /= n_bootstrap
    stability_score = np.mean(agreement_matrix)

    print(f"\n  Bootstrap stability score: {stability_score:.4f}")
    print(f"  (1.0 = perfectly stable, 0.0 = completely unstable)")

    return stability_score, agreement_matrix
