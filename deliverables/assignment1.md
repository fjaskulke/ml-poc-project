# Assignment 1 — Proposal & Exploratory Data Analysis

## Description du projet

Ce projet vise à segmenter les clients d'un retailer e-commerce britannique
en groupes homogènes, à partir de leur historique d'achats.
L'objectif est d'identifier des profils clients distincts pour permettre
des actions marketing ciblées (fidélisation, réactivation, upsell).

## Définition du problème

**Type de problème : Clustering (apprentissage non-supervisé)**

Il n'existe pas de labels de segmentation prédéfinis.
L'algorithme doit découvrir seul des groupes naturels dans les données,
en se basant sur des métriques comportementales issues des transactions.

## Dataset choisi

- **Nom** : Online Retail II
- **Source** : UCI Machine Learning Repository
- **URL** : https://archive.uci.edu/dataset/502/online+retail+ii
- **Période** : 01/12/2009 – 09/12/2011
- **Volume** : ~1 067 371 transactions, ~5 878 clients uniques
- **Format** : fichier Excel (.xlsx)
- **Licence** : Creative Commons Attribution 4.0 (CC BY 4.0)

## Description des features disponibles

| Feature      | Type      | Description                                         |
|--------------|-----------|-----------------------------------------------------|
| Invoice      | Nominal   | Numéro de facture (préfixe C = annulation)          |
| StockCode    | Nominal   | Code produit unique                                 |
| Description  | Texte     | Nom du produit                                      |
| Quantity     | Numérique | Quantité achetée par transaction                    |
| InvoiceDate  | Datetime  | Date et heure de la transaction                     |
| Price        | Numérique | Prix unitaire en livres sterling                    |
| Customer ID  | Nominal   | Identifiant client unique                           |
| Country      | Nominal   | Pays de résidence du client                         |

## Features construites — RFM

À partir des données brutes, trois features seront calculées par client :

- **Recency (R)** : nombre de jours depuis le dernier achat
- **Frequency (F)** : nombre de commandes passées sur la période
- **Monetary (M)** : montant total dépensé (Quantity x Price)

## Premières analyses exploratoires (EDA)

Voir le notebook : `notebooks/eda_online_retail.ipynb`

Analyses réalisées :
- Distribution des transactions par pays (majorité UK)
- Distribution du chiffre d'affaires par client
- Évolution mensuelle des ventes (saisonnalité novembre-décembre)
- Valeurs manquantes : Customer ID ~25% de NaN (à exclure)
- Transactions annulées : Invoice commençant par C (à filtrer)
- Distribution des quantités et prix (présence d'outliers)

## Objectif business

Permettre à l'équipe marketing de personnaliser ses campagnes selon le profil
de chaque segment :
- Récompenser les clients à haute valeur
- Réactiver les clients inactifs
- Convertir les clients occasionnels en clients réguliers

## Contexte machine learning

- **Paradigme** : Non-supervisé
- **Algorithmes envisagés** : K-Means, DBSCAN, Clustering Hiérarchique
- **Représentation** : chaque client est représenté par un vecteur RFM normalisé

## Métrique d'évaluation envisagée

- **Silhouette Score** : mesure la cohésion et la séparation entre clusters
- **Inertie (WCSS)** : utilisée pour la méthode du coude (Elbow Method)

## Hypothèses, risques et limites

| Catégorie  | Détail                                                                |
|------------|-----------------------------------------------------------------------|
| Hypothèse  | Le comportement passé (RFM) est prédicteur des profils futurs         |
| Hypothèse  | Les clients sans ID sont des guests non suivables, exclus             |
| Risque     | Forte concentration sur le marché UK, biais géographique              |
| Risque     | Données de 2009-2011, comportements potentiellement obsolètes         |
| Limite     | Pas d'information démographique disponible                            |
| Limite     | Le nombre de clusters K doit être choisi manuellement                 |

## Données / Notebooks

### Obtention du dataset

1. Aller sur : https://archive.uci.edu/dataset/502/online+retail+ii
2. Télécharger le fichier zip
3. Extraire online_retail_II.xlsx dans le dossier data/

### Localisation

