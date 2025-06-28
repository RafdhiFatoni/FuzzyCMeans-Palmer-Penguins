import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skfuzzy.cluster import cmeans as FCM
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from scipy.stats import mode

def select_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Mengambil hanya kolom fitur yang diinginkan dari DataFrame.

    Args:
        df (pd.DataFrame): DataFrame input.
        feature_columns (list): Daftar nama kolom fitur yang diinginkan.

    Returns:
        pd.DataFrame: DataFrame hanya dengan kolom fitur yang dipilih.
    """
    return df[feature_columns].copy()

def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan standarisasi pada setiap kolom DataFrame menggunakan StandardScaler (mean=0, std=1).
    Args:
        df (pd.DataFrame): DataFrame input.
    Returns:
        pd.DataFrame: DataFrame dengan fitur yang sudah distandarisasi.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

def fuzzy_c_means(X, n_clusters=3, m=2, max_iter=150, error=1e-5, random_state=None):
    """
    Melakukan clustering menggunakan library Fuzzy C-Means (FCM).

    Args:
        X (np.ndarray or pd.DataFrame): Data input (n_samples, n_features).
        n_clusters (int): Jumlah cluster.
        m (float): Fuzziness parameter (>1).
        max_iter (int): Maksimal iterasi.
        error (float): Threshold konvergensi.
        random_state (int, optional): Seed random.

    Returns:
        centers (np.ndarray): Pusat cluster (n_clusters, n_features).
        u (np.ndarray): Matriks keanggotaan (n_samples, n_clusters).
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    cntr, u, u0, d, jm, p, fpc = FCM(X, c=n_clusters, m=m, maxiter=max_iter, error=error, seed=random_state)
    return cntr, u

def plot_fcm_clusters_2d(X, cluster_labels, centers, x_col, y_col):
    """
    Menampilkan visualisasi kluster FCM 2D beserta centroidnya.

    Args:
        X (pd.DataFrame or np.ndarray): Data input (n_samples, n_features).
        u (np.ndarray): Matriks keanggotaan (n_samples, n_clusters).
        centers (np.ndarray): Pusat cluster (n_clusters, n_features).
        x_col (str or int): Nama kolom (jika DataFrame) atau indeks fitur (jika ndarray) untuk sumbu x.
        y_col (str or int): Nama kolom (jika DataFrame) atau indeks fitur (jika ndarray) untuk sumbu y.
    """
    if isinstance(X, pd.DataFrame):
        x = X[x_col].values
        y = X[y_col].values
        x_idx = X.columns.get_loc(x_col)
        y_idx = X.columns.get_loc(y_col)
    else:
        x = X[:, x_col]
        y = X[:, y_col]
        x_idx = x_col
        y_idx = y_col

    n_clusters = centers.shape[0]
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', n_clusters)

    for i in range(n_clusters):
        plt.scatter(
            x[cluster_labels == i],
            y[cluster_labels == i],
            s=40,
            color=colors(i),
            label=f'Cluster {i+1}',
            alpha=0.6
        )
        plt.scatter(
            centers[i, x_idx],
            centers[i, y_idx],
            marker='X',
            color=colors(i),
            s=200,
            edgecolor='k',
            linewidths=2
        )

    plt.xlabel(str(x_col))
    plt.ylabel(str(y_col))
    plt.title('FCM Clusters Visualization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_clustering(X, cluster_labels, true_labels):
    """
    Mengevaluasi hasil clustering FCM menggunakan silhouette score dan davies-bouldin score.

    Args:
        X (np.ndarray or pd.DataFrame): Data input (n_samples, n_features).
        u (np.ndarray): Matriks keanggotaan (n_samples, n_clusters).
        true_labels (array-like): Label spesies asli.

    Returns:
        dict: Dictionary berisi silhouette score dan davies-bouldin score.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    sil_score = silhouette_score(X, cluster_labels)
    db_score = davies_bouldin_score(X, cluster_labels)
    return {
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score
    }

def plot_confusion_matrix(true_labels, cluster_labels, class_names=None):
    """
    Menampilkan confusion matrix antara label asli dan hasil cluster, serta akurasi clustering.
    Fungsi ini secara otomatis memetakan label cluster (numerik) ke label asli (string)
    berdasarkan suara mayoritas (majority vote).
    """
    # ==============================================================================
    # BAGIAN YANG DIUBAH / DITAMBAHKAN
    # ==============================================================================
    
    # 1. Logika untuk memetakan cluster (angka) ke label asli (string)
    mapping_df = pd.DataFrame({'true_label': true_labels, 'cluster_label': cluster_labels})
    mapping = mapping_df.groupby('cluster_label')['true_label'].apply(lambda x: x.mode()[0])
    predicted_labels_mapped = pd.Series(cluster_labels).map(mapping)
    
    # Jika class_names tidak diberikan, buat dari label asli
    if class_names is None:
        class_names = sorted(list(true_labels.unique()))

    # 2. Gunakan hasil mapping untuk kalkulasi
    # Pastikan urutan label konsisten untuk matriks dan plot
    cm = confusion_matrix(true_labels, predicted_labels_mapped, labels=class_names)
    acc = accuracy_score(true_labels, predicted_labels_mapped)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label (from Cluster)') # Ubah label sumbu x agar lebih jelas
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Akurasi: {acc:.2f})')
    plt.tight_layout()
    plt.show()
    return acc


print("FCM model loaded successfully.")