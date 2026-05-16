"""
data.py — Pipeline de nettoyage et feature engineering
Projet : Segmentation client par analyse RFM
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset Online Retail II depuis un fichier CSV.

    Args:
        filepath : chemin vers online_retail_II.csv

    Returns:
        DataFrame brut
    """
    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"[load_data] {df.shape[0]:,} lignes chargées.")
    return df


# ─────────────────────────────────────────
# 2. NETTOYAGE
# ─────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les étapes de nettoyage sur le dataset brut.

    Étapes :
        1. Suppression des lignes sans Customer ID (guests)
        2. Suppression des transactions annulées (Invoice starts with 'C')
        3. Suppression des quantités et prix négatifs ou nuls
        4. Création de la colonne TotalPrice = Quantity x Price
        5. Conversion Customer ID en entier

    Args:
        df : DataFrame brut

    Returns:
        DataFrame nettoyé
    """
    n_init = len(df)

    # 1. Supprimer les lignes sans Customer ID
    df = df.dropna(subset=['Customer ID'])
    print(f"[clean] Après suppression Customer ID nuls : {len(df):,} lignes (-{n_init - len(df):,})")

    # 2. Supprimer les annulations
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    print(f"[clean] Après suppression annulations : {len(df):,} lignes")

    # 3. Supprimer quantités et prix invalides
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    print(f"[clean] Après suppression invalides : {len(df):,} lignes")

    # 4. Créer TotalPrice
    df = df.copy()
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # 5. Customer ID en entier
    df['Customer ID'] = df['Customer ID'].astype(int)

    print(f"[clean] Dataset final : {df.shape}")
    return df


# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING — RFM
# ─────────────────────────────────────────

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features RFM par client.

    - Recency  : nb de jours depuis le dernier achat
    - Frequency: nb de commandes distinctes
    - Monetary : CA total (somme des TotalPrice)

    Args:
        df : DataFrame nettoyé

    Returns:
        DataFrame RFM avec une ligne par client
    """
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"[rfm] Date de référence : {reference_date.date()}")

    rfm = df.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency = ('Invoice',     'nunique'),
        Monetary  = ('TotalPrice',  'sum')
    ).reset_index()

    print(f"[rfm] {len(rfm):,} clients uniques.")
    return rfm


# ─────────────────────────────────────────
# 4. TRAITEMENT DES OUTLIERS
# ─────────────────────────────────────────

def remove_outliers(rfm: pd.DataFrame, upper_quantile: float = 0.99) -> pd.DataFrame:
    """
    Supprime les outliers extrêmes sur Monetary et Frequency
    en utilisant un seuil au percentile donné.

    Justification : les outliers extrêmes (gros grossistes) fausseraient
    les centroïdes des clusters et rendraient la segmentation moins pertinente
    pour la majorité des clients.

    Args:
        rfm            : DataFrame RFM
        upper_quantile : seuil percentile (défaut 0.99)

    Returns:
        DataFrame RFM sans outliers extrêmes
    """
    n_init = len(rfm)

    monetary_cap  = rfm['Monetary'].quantile(upper_quantile)
    frequency_cap = rfm['Frequency'].quantile(upper_quantile)

    rfm = rfm[
        (rfm['Monetary']  <= monetary_cap) &
        (rfm['Frequency'] <= frequency_cap)
    ].copy()

    print(f"[outliers] {n_init - len(rfm)} clients retirés (>{upper_quantile*100:.0f}e percentile).")
    print(f"[outliers] Dataset après nettoyage outliers : {len(rfm):,} clients.")
    return rfm


# ─────────────────────────────────────────
# 5. NORMALISATION
# ─────────────────────────────────────────

def normalize_rfm(rfm: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Applique une transformation log puis StandardScaler sur les features RFM.

    Étapes :
        1. Log-transformation : log1p(x) pour réduire l'asymétrie
           (distributions très skewed → log les rend plus gaussiennes)
        2. StandardScaler : centre et réduit chaque feature (mean=0, std=1)
           pour que K-Means ne soit pas dominé par Monetary

    Pourquoi pas MinMaxScaler ?
        MinMaxScaler est sensible aux outliers résiduels.
        StandardScaler est plus robuste pour le clustering.

    Args:
        rfm : DataFrame RFM (après suppression outliers)

    Returns:
        rfm_scaled : DataFrame normalisé (colonnes R, F, M)
        scaler     : objet StandardScaler fitté (pour inverse_transform si besoin)
    """
    features = ['Recency', 'Frequency', 'Monetary']

    # Log-transformation
    rfm_log = rfm[features].copy()
    rfm_log['Recency']   = np.log1p(rfm_log['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary']  = np.log1p(rfm_log['Monetary'])

    # StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(rfm_log)

    rfm_scaled = pd.DataFrame(
        scaled_values,
        columns=['R_scaled', 'F_scaled', 'M_scaled'],
        index=rfm.index
    )

    # Ajouter le Customer ID pour garder la traçabilité
    rfm_scaled['Customer ID'] = rfm['Customer ID'].values

    print(f"[normalize] Normalisation terminée. Shape : {rfm_scaled.shape}")
    return rfm_scaled, scaler


# ─────────────────────────────────────────
# 6. PIPELINE COMPLET
# ─────────────────────────────────────────

def full_pipeline(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Exécute le pipeline complet : load → clean → RFM → outliers → normalize.

    Args:
        filepath : chemin vers online_retail_II.csv

    Returns:
        rfm        : DataFrame RFM brut (avant normalisation)
        rfm_scaled : DataFrame RFM normalisé (prêt pour le clustering)
        scaler     : objet StandardScaler fitté
    """
    print("=" * 50)
    print("PIPELINE DATA — Online Retail II")
    print("=" * 50)

    df         = load_data(filepath)
    df_clean   = clean_data(df)
    rfm        = compute_rfm(df_clean)
    rfm_clean  = remove_outliers(rfm)
    rfm_scaled, scaler = normalize_rfm(rfm_clean)

    print("=" * 50)
    print("Pipeline terminé ✓")
    print(f"  → RFM brut     : {rfm_clean.shape}")
    print(f"  → RFM normalisé : {rfm_scaled.shape}")
    print("=" * 50)

    return rfm_clean, rfm_scaled, scaler


# ─────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────

if __name__ == '__main__':
    rfm, rfm_scaled, scaler = full_pipeline('../data/online_retail_II.csv')
    print(rfm.head())
    print(rfm_scaled.head())
