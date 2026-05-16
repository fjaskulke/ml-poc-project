"""
metrics.py — Métriques d'évaluation pour le clustering
Projet : Segmentation client par analyse RFM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


# ─────────────────────────────────────────
# 1. SILHOUETTE SCORE
# ─────────────────────────────────────────

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcule le Silhouette Score pour évaluer la qualité du clustering.

    Interprétation :
        +1  → clusters parfaitement séparés et cohésifs
         0  → clusters qui se chevauchent
        -1  → points mal assignés à leur cluster

    Args:
        X      : array numpy des features normalisées (n_samples, n_features)
        labels : array des labels de clusters assignés

    Returns:
        score : float entre -1 et 1
    """
    if len(set(labels)) < 2:
        print("[silhouette] Attention : moins de 2 clusters, score non calculable.")
        return None

    score = silhouette_score(X, labels)
    print(f"[silhouette] Score : {score:.4f}")
    return score


# ─────────────────────────────────────────
# 2. INERTIE (WCSS)
# ─────────────────────────────────────────

def compute_inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Calcule l'inertie (Within-Cluster Sum of Squares).

    L'inertie mesure la compacité des clusters :
    plus elle est faible, plus les points sont proches de leur centroïde.
    Utilisée pour la méthode du coude (Elbow Method).

    Args:
        X       : array des features normalisées
        labels  : labels de clusters
        centers : centroïdes des clusters

    Returns:
        inertia : float
    """
    inertia = sum(
        np.sum((X[labels == k] - centers[k]) ** 2)
        for k in range(len(centers))
    )
    print(f"[inertia] Inertie : {inertia:.4f}")
    return inertia


# ─────────────────────────────────────────
# 3. ELBOW METHOD
# ─────────────────────────────────────────

def elbow_method(X: np.ndarray, k_range: range = range(2, 11),
                 random_state: int = 42) -> dict:
    """
    Applique la méthode du coude pour trouver le K optimal.

    Pour chaque valeur de K, entraîne un K-Means et enregistre
    l'inertie et le Silhouette Score.

    Args:
        X            : array des features normalisées
        k_range      : plage de valeurs K à tester (défaut 2 à 10)
        random_state : graine aléatoire pour reproductibilité

    Returns:
        results : dict avec clés 'k', 'inertia', 'silhouette'
    """
    inertias    = []
    silhouettes = []
    k_values    = list(k_range)

    print("[elbow] Test des valeurs de K...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, km.labels_)
        silhouettes.append(sil)
        print(f"  K={k} | Inertie={km.inertia_:.1f} | Silhouette={sil:.4f}")

    results = {
        'k':          k_values,
        'inertia':    inertias,
        'silhouette': silhouettes
    }
    return results


# ─────────────────────────────────────────
# 4. VISUALISATION ELBOW + SILHOUETTE
# ─────────────────────────────────────────

def plot_elbow_silhouette(results: dict) -> None:
    """
    Affiche les courbes Elbow (inertie) et Silhouette Score
    pour choisir le K optimal visuellement.

    Args:
        results : dict retourné par elbow_method()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Courbe Elbow
    axes[0].plot(results['k'], results['inertia'],
                 marker='o', color='steelblue', linewidth=2)
    axes[0].set_title('Méthode du Coude (Elbow)', fontweight='bold')
    axes[0].set_xlabel('Nombre de clusters K')
    axes[0].set_ylabel('Inertie (WCSS)')
    axes[0].grid(True, alpha=0.3)

    # Courbe Silhouette
    axes[1].plot(results['k'], results['silhouette'],
                 marker='o', color='coral', linewidth=2)
    axes[1].set_title('Silhouette Score par K', fontweight='bold')
    axes[1].set_xlabel('Nombre de clusters K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, alpha=0.3)

    # Meilleur K selon Silhouette
    best_k = results['k'][np.argmax(results['silhouette'])]
    best_s = max(results['silhouette'])
    axes[1].axvline(x=best_k, color='green', linestyle='--', alpha=0.7,
                    label=f'Meilleur K={best_k} ({best_s:.3f})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    print(f"[plot] Meilleur K selon Silhouette : {best_k} (score={best_s:.4f})")


# ─────────────────────────────────────────
# 5. RAPPORT DE CLUSTERING
# ─────────────────────────────────────────

def cluster_report(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Génère un rapport descriptif par cluster avec les moyennes RFM.

    Args:
        rfm    : DataFrame RFM brut (non normalisé)
        labels : labels de clusters assignés

    Returns:
        report : DataFrame avec statistiques par cluster
    """
    rfm = rfm.copy()
    rfm['Cluster'] = labels

    report = rfm.groupby('Cluster').agg(
        Nb_clients = ('Customer ID', 'count'),
        Recency_mean   = ('Recency',   'mean'),
        Frequency_mean = ('Frequency', 'mean'),
        Monetary_mean  = ('Monetary',  'mean'),
        Monetary_sum   = ('Monetary',  'sum')
    ).round(2)

    report['Pct_clients'] = (report['Nb_clients'] / report['Nb_clients'].sum() * 100).round(1)

    print("\n=== Rapport par cluster ===")
    print(report.to_string())
    return report


# ─────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from src.data import full_pipeline

    rfm, rfm_scaled, scaler = full_pipeline('../data/online_retail_II.csv')
    X = rfm_scaled[['R_scaled', 'F_scaled', 'M_scaled']].values

    results = elbow_method(X, k_range=range(2, 8))
    plot_elbow_silhouette(results)
