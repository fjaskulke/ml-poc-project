# Assignment 3 — Modèles de Clustering

## Définition du problème ML

**Type** : Clustering non-supervisé  
**Objectif** : Segmenter les clients e-commerce en groupes homogènes à partir de leurs features RFM (Recency, Frequency, Monetary), sans labels prédéfinis.

---

## Définition de la métrique d'évaluation

Comme il n'existe pas de vérité terrain (pas de labels), on utilise des métriques internes :

### Métrique principale — Silhouette Score
- **Formule** : `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
  - `a(i)` = distance moyenne au sein du cluster
  - `b(i)` = distance moyenne au cluster le plus proche
- **Plage** : entre -1 et +1
- **Interprétation** : plus c'est élevé, meilleure est la séparation entre clusters
- **Seuils** : > 0.5 = bon, > 0.7 = excellent

### Métrique secondaire — Inertie (WCSS)
- Somme des distances au carré des points à leur centroïde
- Utilisée pour la méthode du coude (Elbow Method) pour choisir K
- Plus elle est faible, plus les clusters sont compacts

---

## Protocole d'évaluation

- **Pas de train/test split classique** : le clustering est non-supervisé, il n'y a pas de labels à prédire
- **Split utilisé** : 80/20 pour évaluer la stabilité des clusters sur des données non vues
- **Reproductibilité** : `random_state=42` sur tous les modèles stochastiques
- **Sélection de K** : méthode du coude + Silhouette Score testés sur K=2 à 10

---

## Présentation des trois modèles

### Modèle 1 — K-Means

**Hypothèses principales**
- Les clusters sont convexes et de taille comparable
- Chaque point appartient exactement à un cluster
- La distance euclidienne est une bonne mesure de similarité

**Avantages attendus**
- Simple à interpréter et à expliquer en soutenance
- Très efficace sur des données normalisées en dimension faible (3D RFM)
- Résultats reproductibles avec `random_state`
- Standard de l'industrie pour la segmentation RFM

**Limites attendues**
- Nécessite de fixer K à l'avance
- Sensible aux outliers (centroïdes tirés vers les extrêmes)
- Suppose des clusters sphériques → inadapté si les groupes ont des formes complexes

**Adéquation avec le problème**
Excellente. K-Means est le modèle de référence pour la segmentation RFM. Les données normalisées (log + StandardScaler) réduisent l'impact des outliers. K=4 est choisi via l'Elbow Method et le Silhouette Score.

---

### Modèle 2 — DBSCAN

**Hypothèses principales**
- Les clusters sont des régions denses séparées par des zones de faible densité
- Les outliers existent et doivent être détectés (label = -1)
- Les paramètres `eps` et `min_samples` définissent ce qu'est un "voisinage dense"

**Avantages attendus**
- Ne nécessite pas de spécifier K à l'avance
- Détecte automatiquement les outliers (clients atypiques)
- Peut trouver des clusters de forme non sphérique

**Limites attendues**
- Sensible au choix de `eps` et `min_samples` (nécessite du tuning)
- Peut produire un seul grand cluster si la densité est homogène
- Moins interprétable business que K-Means

**Adéquation avec le problème**
Moyenne. Utile pour détecter les clients vraiment atypiques (fraud, gros comptes B2B résiduels). Mais la segmentation RFM standard donne généralement des clusters assez sphériques, ce qui favorise K-Means.

---

### Modèle 3 — Agglomerative Clustering (Ward)

**Hypothèses principales**
- La similarité entre clusters peut être mesurée par la variance intra-cluster (linkage Ward)
- La hiérarchie des clusters est informative pour choisir le bon niveau de granularité
- Les données peuvent être représentées par un dendrogramme

**Avantages attendus**
- Ne nécessite pas de spécifier K avant l'entraînement (on coupe le dendrogramme après)
- Le dendrogramme permet une exploration visuelle de la structure des données
- Linkage Ward minimise la variance totale → clusters compacts et homogènes

**Limites attendues**
- Complexité O(n²) en mémoire → lent sur de grands datasets (>10k points)
- Pas de prédiction sur de nouveaux points (modèle non-paramétrique)
- Le dendrogramme peut être difficile à lire avec beaucoup de clients

**Adéquation avec le problème**
Bonne. Avec ~5700 clients, la complexité est gérable. Permet de valider le nombre de clusters trouvé par K-Means en observant le dendrogramme.

---

## Justification du choix des trois modèles

| Critère | K-Means | DBSCAN | Agglomerative |
|---------|---------|--------|---------------|
| Interprétabilité business | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Gestion des outliers | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Pas besoin de K fixé | ❌ | ✅ | ✅ |
| Clusters non sphériques | ❌ | ✅ | ❌ |
| Scalabilité | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Standard industrie RFM | ✅ | ❌ | ❌ |

Ces trois modèles ont des approches très différentes (partitionnement, densité, hiérarchie), ce qui permet une **comparaison riche** et une **validation croisée** des résultats.

---

## Données / Notebooks

### Localisation

```
notebooks/
└── models.ipynb         ← entraînement, comparaison, sauvegarde des modèles
models/
├── kmeans.pkl           ← modèle K-Means sérialisé
├── dbscan.pkl           ← modèle DBSCAN sérialisé
└── agglomerative.pkl    ← modèle Agglomerative sérialisé
results/
└── model_metrics.csv    ← métriques comparatives des 3 modèles
src/
└── config.py            ← déclaration des 3 modèles
```

### Exécution pour reproduire les expériences

```bash
pip install -r requirements.txt
jupyter notebook notebooks/models.ipynb
```

Le notebook entraîne les 3 modèles, sauvegarde les `.pkl` et génère `results/model_metrics.csv`.
