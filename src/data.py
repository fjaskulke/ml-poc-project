"""
data.py — Pipeline de nettoyage et feature engineering
Projet : Segmentation client par analyse RFM

Contrat imposé :
    load_dataset_split() doit retourner (X_train, X_test, y_train, y_test)
"""

from __future__ import annotations
from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/online_retail_II.csv'


# ── Fonctions internes ──────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"[load_data] {df.shape[0]:,} lignes chargées.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage :
    1. Suppression lignes sans Customer ID (guests)
    2. Suppression transactions annulées (Invoice starts with C)
    3. Suppression quantités / prix invalides
    4. Création TotalPrice = Quantity x Price
    """
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()
    df['TotalPrice'] = df['Quantity'] * df['Price']
    df['Customer ID'] = df['Customer ID'].astype(int)
    print(f"[clean_data] Dataset nettoyé : {df.shape}")
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule Recency, Frequency, Monetary par client."""
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency = ('Invoice',     'nunique'),
        Monetary  = ('TotalPrice',  'sum')
    ).reset_index()
    print(f"[compute_rfm] {len(rfm):,} clients uniques.")
    return rfm


def remove_outliers(rfm: pd.DataFrame, upper_quantile: float = 0.99) -> pd.DataFrame:
    """Supprime les outliers au-delà du 99e percentile sur Monetary et Frequency."""
    rfm = rfm[
        (rfm['Monetary']  <= rfm['Monetary'].quantile(upper_quantile)) &
        (rfm['Frequency'] <= rfm['Frequency'].quantile(upper_quantile))
    ].copy()
    print(f"[remove_outliers] {len(rfm):,} clients après suppression outliers.")
    return rfm


def normalize_rfm(rfm: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Applique log1p + StandardScaler sur les features RFM."""
    features = ['Recency', 'Frequency', 'Monetary']
    rfm_log = rfm[features].copy()
    for col in features:
        rfm_log[col] = np.log1p(rfm_log[col])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_log)

    rfm_scaled = pd.DataFrame(scaled, columns=['R_scaled', 'F_scaled', 'M_scaled'], index=rfm.index)
    rfm_scaled['Customer ID'] = rfm['Customer ID'].values
    return rfm_scaled, scaler


# ── Contrat imposé par le prof ──────────────────────────────────────────────

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Pour le clustering non-supervisé, il n'y a pas de y (labels vrais).
    X_train / X_test : features RFM normalisées (numpy arrays)
    y_train / y_test : None (pas de supervision)

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.
    """
    df        = load_data(DATA_PATH)
    df_clean  = clean_data(df)
    rfm       = compute_rfm(df_clean)
    rfm_clean = remove_outliers(rfm)
    rfm_scaled, _ = normalize_rfm(rfm_clean)

    X = rfm_scaled[['R_scaled', 'F_scaled', 'M_scaled']].values
    y = [None] * len(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[load_dataset_split] X_train={X_train.shape} | X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Pipeline complet (usage direct) ────────────────────────────────────────

def full_pipeline(filepath: str = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Exécute load → clean → RFM → outliers → normalize."""
    df        = load_data(filepath)
    df_clean  = clean_data(df)
    rfm       = compute_rfm(df_clean)
    rfm_clean = remove_outliers(rfm)
    rfm_scaled, scaler = normalize_rfm(rfm_clean)
    return rfm_clean, rfm_scaled, scaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset_split()
    print(f"X_train shape: {X_train.shape}")
