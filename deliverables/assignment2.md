# Assignment 2 — Feature Engineering

## Description des étapes de nettoyage des données

Le dataset brut contient 1 067 371 transactions. Plusieurs étapes de nettoyage sont nécessaires avant de pouvoir construire les features RFM.

### Étape 1 — Suppression des lignes sans Customer ID

**Problème** : ~25% des lignes n'ont pas de Customer ID. Ces transactions correspondent à des achats "guest" sans compte client enregistré.

**Action** : Suppression de ces lignes via `dropna(subset=['Customer ID'])`.

**Justification** : Le clustering RFM repose sur le suivi du comportement d'un client dans le temps. Sans identifiant, il est impossible de regrouper les transactions d'un même client.

---

### Étape 2 — Suppression des transactions annulées

**Problème** : Les factures dont le numéro commence par `C` sont des annulations (retours). Elles ont des quantités négatives et ne représentent pas un achat réel.

**Action** : Filtrage via `~df['Invoice'].astype(str).str.startswith('C')`.

**Justification** : Inclure les annulations fausserait les métriques Frequency et Monetary en les sous-estimant.

---

### Étape 3 — Suppression des quantités et prix invalides

**Problème** : Certaines lignes ont des quantités ou des prix négatifs ou nuls (erreurs de saisie, ajustements comptables).

**Action** : Filtrage `(Quantity > 0) & (Price > 0)`.

**Justification** : Ces lignes ne correspondent pas à des ventes réelles et pollueraient le calcul du CA total.

---

### Étape 4 — Création de TotalPrice

**Action** : `TotalPrice = Quantity × Price`

**Justification** : Cette colonne dérivée est nécessaire pour calculer la feature Monetary du RFM.

---

## Description des transformations appliquées

### Construction des features RFM

À partir du dataset nettoyé, on agrège les données par client pour obtenir 3 features :

| Feature   | Calcul | Interprétation |
|-----------|--------|----------------|
| Recency   | Nb de jours entre le dernier achat et la date de référence | Plus R est faible, plus le client est récent |
| Frequency | Nb de factures distinctes (`nunique`) | Plus F est élevé, plus le client est fidèle |
| Monetary  | Somme des TotalPrice | Plus M est élevé, plus le client est rentable |

**Date de référence** : lendemain de la dernière transaction du dataset, pour que tous les clients soient calculés sur la même base.

---

### Traitement des outliers

**Problème** : Les distributions RFM sont très asymétriques (quelques gros clients B2B avec des CA 100x supérieurs à la médiane).

**Action** : Suppression des clients au-delà du 99e percentile sur Monetary et Frequency.

**Justification** : Ces outliers extrêmes (grossistes) formeraient leurs propres clusters isolés et rendraient la segmentation inutilisable pour le marketing grand public. On les exclut pour se concentrer sur la majorité des clients.

**Alternative non retenue** : Winsorization (plafonner au lieu de supprimer). Rejetée car elle introduirait des valeurs artificielles qui fausseraient les centroïdes.

---

### Log-transformation

**Problème** : Les distributions de Frequency et Monetary sont très skewed (longue queue à droite). K-Means suppose des distributions relativement symétriques pour bien fonctionner.

**Action** : Application de `log1p(x)` sur les 3 features RFM.

**Justification** : `log1p` réduit l'asymétrie en compressant les grandes valeurs, ce qui améliore la qualité du clustering.

**Alternative non retenue** : Transformation racine carrée (`sqrt`). Moins efficace que `log` pour des distributions très skewed.

---

### StandardScaler

**Problème** : Les trois features RFM ont des échelles très différentes (Recency en jours 0-700, Monetary en £ 0-50 000). K-Means utilise la distance euclidienne, donc une feature à grande échelle dominerait les autres.

**Action** : `StandardScaler` → centre (mean=0) et réduit (std=1) chaque feature.

**Justification** : Toutes les features contribuent équitablement à la distance euclidienne.

**Alternative non retenue** : `MinMaxScaler`. Rejeté car sensible aux outliers résiduels après le 99e percentile.

---

## Description des nouvelles features créées

| Feature    | Type    | Source       | Transformation |
|------------|---------|--------------|----------------|
| Recency    | Numérique | InvoiceDate | Calcul en jours depuis référence |
| Frequency  | Numérique | Invoice     | Comptage factures distinctes |
| Monetary   | Numérique | TotalPrice  | Somme par client |
| R_scaled   | Numérique | Recency     | log1p + StandardScaler |
| F_scaled   | Numérique | Frequency   | log1p + StandardScaler |
| M_scaled   | Numérique | Monetary    | log1p + StandardScaler |

---

## Impact attendu des transformations sur les modèles

| Transformation | Impact attendu |
|----------------|----------------|
| Suppression guests | Réduction du bruit, focus sur clients traçables |
| Suppression annulations | Métriques RFM plus fiables |
| Suppression outliers (99e pct) | Clusters plus homogènes, centroïdes plus représentatifs |
| Log-transformation | Distributions plus symétriques, meilleure séparation des clusters |
| StandardScaler | Équilibre entre les 3 dimensions RFM dans la distance euclidienne |

---

## Données / Notebooks

### Datasets transformés

Le dataset RFM normalisé est généré par `src/data.py` et n'est pas versionné (calculé à la volée).

### Localisation des fichiers

```
src/
├── data.py       ← pipeline complet load → clean → RFM → normalize
└── metrics.py    ← fonctions Silhouette, Inertie, Elbow Method
notebooks/
└── feature_engineering.ipynb  ← exécution et visualisation des transformations
data/
└── online_retail_II.csv       ← dataset source (non versionné)
```

### Comment charger et utiliser

```python
from src.data import full_pipeline

rfm, rfm_scaled, scaler = full_pipeline('data/online_retail_II.csv')
# rfm        → DataFrame RFM brut (Recency, Frequency, Monetary)
# rfm_scaled → DataFrame normalisé (R_scaled, F_scaled, M_scaled) — prêt pour clustering
# scaler     → objet StandardScaler fitté (pour inverse_transform si besoin)
```

### Exécution du notebook

```bash
pip install -r requirements.txt
jupyter notebook notebooks/feature_engineering.ipynb
```
