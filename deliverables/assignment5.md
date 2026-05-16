# Assignment 5 — Application Streamlit

## Description de l'application

L'application **Segmentation Client RFM** est un tableau de bord interactif développé avec Streamlit. Elle permet d'explorer, visualiser et utiliser les résultats du clustering RFM réalisé sur le dataset Online Retail II.

---

## Objectif de l'interface

L'interface a trois objectifs principaux :
- **Démonstration** : présenter les résultats du clustering de façon visuelle et accessible
- **Exploration** : permettre d'explorer les données brutes et les segments clients
- **Prédiction** : classifier un nouveau client dans un segment à partir de ses valeurs RFM

---

## Description des fonctionnalités

L'application est organisée en 5 pages accessibles via la sidebar :

### 🏠 Accueil
- KPIs globaux (nb clients, transactions, CA total, pays)
- Description des 4 segments identifiés
- Navigation vers les autres pages

### 📊 Exploration
- Distribution géographique des ventes (slider pour le nombre de pays)
- Évolution mensuelle du CA
- Distributions RFM avec statistiques descriptives

### 🤖 Clustering
- Profils détaillés des 4 segments (métriques par segment)
- Scatter plots interactifs (choix des axes X/Y)
- Heatmap des profils RFM moyens par cluster
- Table filtrée par segment

### 🔮 Prédiction client
- Saisie des valeurs RFM via des sliders
- Prédiction du segment en temps réel
- Comparaison avec la moyenne du segment
- Recommandation marketing personnalisée

### 📈 Performances
- Tableau comparatif des 3 modèles
- Bar charts Silhouette Score et nb clusters
- Conclusion sur le meilleur modèle

---

## Description des inputs utilisateurs

| Input | Type | Page | Description |
|-------|------|------|-------------|
| Nombre de pays | Slider (5-20) | Exploration | Filtre le top N pays |
| Axe X / Axe Y | Selectbox | Clustering | Choisit les features pour le scatter plot |
| Filtre segment | Selectbox | Clustering | Filtre la table par segment |
| Recency | Slider (1-365) | Prédiction | Jours depuis le dernier achat |
| Frequency | Slider (1-50) | Prédiction | Nombre de commandes |
| Monetary | Slider (10-10000) | Prédiction | CA total en £ |

---

## Description des outputs affichés

- **Métriques** : KPIs numériques (clients, CA, recency moyenne, etc.)
- **Graphiques** : bar charts, line charts, scatter plots, heatmaps
- **Tables** : données RFM filtrées par segment
- **Prédiction** : segment prédit + recommandation marketing
- **Comparaison** : delta entre le client et la moyenne de son segment

---

## Structure de l'application

```
src/
└── app.py          ← application principale (5 pages)

Dépendances :
├── src/data.py     ← pipeline de données (load, clean, RFM, normalize)
├── models/*.pkl    ← modèles entraînés
└── results/model_metrics.csv  ← métriques des modèles
```

---

## Comment lancer l'application

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Lancement
```bash
cd ~/ml-poc-project
streamlit run src/app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse :
**http://localhost:8501**
