"""
metrics.py — Métriques d'évaluation pour le clustering
Projet : Segmentation client par analyse RFM

Contrat imposé :
    compute_metrics(y_true, y_pred) doit retourner dict[str, float]
"""

from __future__ import annotations
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


# ── Contrat imposé par le prof ──────────────────────────────────────────────

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Pour le clustering non-supervisé :
    - y_true : array des features normalisées (X) — utilisé pour calculer
               le Silhouette Score (pas de vérité terrain)
    - y_pred : array des labels de clusters assignés par le modèle

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"silhouette_score": 0.45, "inertia": 1234.5}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.
    """
    y_pred = np.array(y_pred)
    X      = np.array(y_true)   # convention : y_true = X (features)

    metrics = {}

    # Silhouette Score
    if len(set(y_pred)) >= 2:
        metrics['silhouette_score'] = float(silhouette_score(X, y_pred))
    else:
        metrics['silhouette_score'] = 0.0

    # Inertie (WCSS) — calculée manuellement pour être indépendante du modèle
    unique_labels = np.unique(y_pred)
    centers = np.array([X[y_pred == k].mean(axis=0) for k in unique_labels])
    inertia = float(sum(
        np.sum((X[y_pred == k] - centers[i]) ** 2)
        for i, k in enumerate(unique_labels)
    ))
    metrics['inertia'] = inertia

    # Nombre de clusters
    metrics['n_clusters'] = float(len(unique_labels))

    print(f"[compute_metrics] {metrics}")
    return metrics


# ── Fonctions utilitaires supplémentaires ───────────────────────────────────

def elbow_method(X: np.ndarray, k_range: range = range(2, 11),
                 random_state: int = 42) -> dict:
    """
    Applique la méthode du coude pour trouver le K optimal.
    Retourne un dict avec les listes de K, inertie et silhouette.
    """
    inertias, silhouettes, k_values = [], [], list(k_range)

    print("[elbow_method] Test des valeurs de K...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, km.labels_)
        silhouettes.append(sil)
        print(f"  K={k} | Inertie={km.inertia_:.1f} | Silhouette={sil:.4f}")

    return {'k': k_values, 'inertia': inertias, 'silhouette': silhouettes}


def plot_elbow_silhouette(results: dict) -> None:
    """Visualise les courbes Elbow et Silhouette Score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results['k'], results['inertia'], marker='o', color='steelblue', linewidth=2)
    axes[0].set_title('Méthode du Coude (Elbow)', fontweight='bold')
    axes[0].set_xlabel('Nombre de clusters K')
    axes[0].set_ylabel('Inertie (WCSS)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['k'], results['silhouette'], marker='o', color='coral', linewidth=2)
    axes[1].set_title('Silhouette Score par K', fontweight='bold')
    axes[1].set_xlabel('Nombre de clusters K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, alpha=0.3)

    best_k = results['k'][np.argmax(results['silhouette'])]
    best_s = max(results['silhouette'])
    axes[1].axvline(x=best_k, color='green', linestyle='--', alpha=0.7,
                    label=f'Meilleur K={best_k} ({best_s:.3f})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    print(f"[plot] Meilleur K : {best_k} (silhouette={best_s:.4f})")


def cluster_report(rfm, labels: np.ndarray):
    """Génère un rapport descriptif par cluster."""
    import pandas as pd
    rfm = rfm.copy()
    rfm['Cluster'] = labels
    report = rfm.groupby('Cluster').agg(
        Nb_clients     = ('Customer ID', 'count'),
        Recency_mean   = ('Recency',     'mean'),
        Frequency_mean = ('Frequency',   'mean'),
        Monetary_mean  = ('Monetary',    'mean'),
    ).round(2)
    report['Pct_clients'] = (report['Nb_clients'] / report['Nb_clients'].sum() * 100).round(1)
    print("\n=== Rapport par cluster ===")
    print(report.to_string())
    return report


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from data import full_pipeline

    rfm, rfm_scaled, scaler = full_pipeline()
    X = rfm_scaled[['R_scaled', 'F_scaled', 'M_scaled']].values

    results = elbow_method(X, k_range=range(2, 8))
    plot_elbow_silhouette(results)
