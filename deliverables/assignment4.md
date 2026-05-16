# Assignment 4 — Visualisations

## Présentation des visualisations

Trois visualisations principales ont été produites, couvrant les données brutes, les données après feature engineering, et les performances des modèles.

---

## Visualisation 1 — Données brutes : Distribution géographique et temporelle

### Objectif
Comprendre la structure brute du dataset avant tout traitement : qui sont les clients, d'où viennent-ils, et comment les ventes évoluent dans le temps.

### Choix du type de graphique
- **Bar chart** pour la distribution par pays : permet de comparer rapidement les volumes entre pays
- **Line chart** pour l'évolution mensuelle du CA : idéal pour visualiser les tendances et la saisonnalité

### Interprétation
- Le Royaume-Uni représente ~90% des transactions → biais géographique fort à mentionner
- Pic de ventes visible en novembre (Black Friday + préparation Noël)
- Creux en janvier/février → comportement saisonnier typique du retail

### Pertinence pour le projet
Justifie les choix de nettoyage (focus sur clients UK enregistrés) et anticipe les limites du modèle (données non représentatives à l'international).

---

## Visualisation 2 — Données après feature engineering : Distributions RFM avant/après normalisation

### Objectif
Montrer l'impact des transformations (log1p + StandardScaler) sur les distributions RFM et justifier leur nécessité pour le clustering.

### Choix du type de graphique
- **Histogrammes comparatifs** (avant vs après) : permettent de visualiser directement le changement de forme des distributions
- Disposition en grille 2×3 pour faciliter la comparaison côte à côte

### Interprétation
- Avant transformation : distributions très asymétriques (longue queue à droite), notamment Monetary
- Après log1p + StandardScaler : distributions beaucoup plus symétriques, centrées sur 0
- La normalisation rend les 3 features comparables en échelle → K-Means peut fonctionner correctement

### Pertinence pour le projet
Démontre visuellement pourquoi la normalisation est indispensable. Sans elle, Monetary (en £) dominerait complètement Frequency et Recency dans le calcul des distances euclidiennes.

---

## Visualisation 3 — Performances des modèles : Clusters K-Means et comparaison des métriques

### Objectif
Visualiser les clusters obtenus par K-Means dans l'espace RFM normalisé et comparer les performances des 3 modèles.

### Choix du type de graphique
- **Scatter plots 2D** (paires de features RFM) : permettent de voir la séparation des clusters dans l'espace
- **Bar chart** des Silhouette Scores : comparaison directe et lisible des 3 modèles
- **Radar chart / heatmap** des moyennes RFM par cluster : profil business de chaque segment

### Interprétation des clusters K-Means (K=4)
| Cluster | Recency | Frequency | Monetary | Profil |
|---------|---------|-----------|----------|--------|
| 0 | Faible | Élevée | Élevé | Champions — clients fidèles et à fort CA |
| 1 | Élevée | Faible | Faible | Clients perdus — inactifs depuis longtemps |
| 2 | Moyenne | Moyenne | Moyen | Clients réguliers — potentiel de fidélisation |
| 3 | Faible | Faible | Faible | Nouveaux clients — à convertir |

### Pertinence pour le projet
Les scatter plots montrent une séparation claire entre les clusters → validation visuelle que K-Means a bien segmenté les données. Le bar chart des Silhouette Scores justifie le choix de K-Means comme meilleur modèle.

---

## Données / Notebooks

### Localisation

```
notebooks/
└── visualizations.ipynb   ← génère toutes les visualisations
plots/                     ← visualisations sauvegardées en PNG
```

### Exécution

```bash
pip install -r requirements.txt
jupyter notebook notebooks/visualizations.ipynb
```
