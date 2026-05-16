"""
app.py — Application Streamlit : Segmentation client RFM
Projet ML — Assignment 5
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier racine au path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import streamlit as st

from src.data import load_data, clean_data, compute_rfm, remove_outliers, normalize_rfm

# ── Configuration de la page ────────────────────────────────────────────────

st.set_page_config(
    page_title="Segmentation Client RFM",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé ────────────────────────────────────────────────────────

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    border-left: 4px solid #1976D2;
}
.cluster-champion { border-left: 4px solid #4CAF50; background: #E8F5E9; }
.cluster-lost     { border-left: 4px solid #F44336; background: #FFEBEE; }
.cluster-regular  { border-left: 4px solid #FF9800; background: #FFF3E0; }
.cluster-new      { border-left: 4px solid #2196F3; background: #E3F2FD; }
</style>
""", unsafe_allow_html=True)


# ── Chargement des données et modèles (mise en cache) ───────────────────────

@st.cache_data
def load_pipeline():
    DATA_PATH = ROOT / 'data' / 'online_retail_II.csv'
    df_raw    = load_data(str(DATA_PATH))
    df_clean  = clean_data(df_raw)
    rfm       = compute_rfm(df_clean)
    rfm_clean = remove_outliers(rfm)
    rfm_scaled, scaler = normalize_rfm(rfm_clean)
    return df_clean, rfm_clean, rfm_scaled, scaler


@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'K-Means': ROOT / 'models' / 'kmeans.pkl',
        'DBSCAN':  ROOT / 'models' / 'dbscan.pkl',
        'Agglomerative': ROOT / 'models' / 'agglomerative.pkl',
    }
    for name, path in model_files.items():
        if path.exists():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    return models


# ── Labels des clusters K-Means ─────────────────────────────────────────────

CLUSTER_LABELS = {
    0: ("Champions", "🏆", "#4CAF50", "Clients fidèles à fort CA"),
    1: ("Clients perdus", "😴", "#F44336", "Inactifs depuis longtemps"),
    2: ("Réguliers", "🔄", "#FF9800", "Potentiel de fidélisation"),
    3: ("Nouveaux", "🌱", "#2196F3", "À convertir en fidèles"),
}


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
st.sidebar.title("🛍️ RFM Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Exploration", "🤖 Clustering", "🔮 Prédiction client", "📈 Performances"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modèle actif** : K-Means (K=4)")
st.sidebar.markdown("**Dataset** : Online Retail II (UCI)")
st.sidebar.markdown("**Clients** : ~5 700")


# ── Chargement ───────────────────────────────────────────────────────────────

with st.spinner("Chargement des données..."):
    try:
        df_clean, rfm, rfm_scaled, scaler = load_pipeline()
        models = load_models()
        kmeans = models.get('K-Means')
        if kmeans is not None:
            X = rfm_scaled[['R_scaled', 'F_scaled', 'M_scaled']].values
            labels = kmeans.labels_
            rfm['Cluster'] = labels
            rfm['Segment'] = rfm['Cluster'].map(lambda k: CLUSTER_LABELS.get(k, (f"Cluster {k}", "", "", ""))[0])
        data_loaded = True
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        data_loaded = False


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ════════════════════════════════════════════════════════════════════════════

if page == "🏠 Accueil":
    st.title("🛍️ Segmentation Client par Analyse RFM")
    st.markdown("### Bienvenue sur le tableau de bord de segmentation client")
    st.markdown("""
    Cette application permet d'explorer et d'analyser la segmentation des clients
    d'un retailer e-commerce britannique à partir de leurs comportements d'achat.
    """)

    st.markdown("---")

    if data_loaded:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("👥 Clients uniques", f"{len(rfm):,}")
        with col2:
            st.metric("🧾 Transactions", f"{len(df_clean):,}")
        with col3:
            st.metric("💰 CA total", f"£{df_clean['TotalPrice'].sum()/1e6:.1f}M")
        with col4:
            st.metric("🌍 Pays", f"{df_clean['Country'].nunique()}")

        st.markdown("---")
        st.markdown("### 🗺️ Navigation")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("**📊 Exploration**\nDistribution des données brutes et RFM")
        with col2:
            st.success("**🤖 Clustering**\nSegments clients et profils RFM")
        with col3:
            st.warning("**🔮 Prédiction**\nClassifier un nouveau client")
        with col4:
            st.error("**📈 Performances**\nComparaison des 3 modèles")

        st.markdown("---")
        st.markdown("### 📋 Les 4 segments identifiés")

        cols = st.columns(4)
        for i, (k, (label, emoji, color, desc)) in enumerate(CLUSTER_LABELS.items()):
            n = (rfm['Cluster'] == k).sum()
            pct = n / len(rfm) * 100
            with cols[i]:
                st.markdown(f"**{emoji} {label}**")
                st.markdown(f"{n:,} clients ({pct:.1f}%)")
                st.markdown(f"*{desc}*")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION
# ════════════════════════════════════════════════════════════════════════════

elif page == "📊 Exploration":
    st.title("📊 Exploration des données")

    if data_loaded:
        tab1, tab2 = st.tabs(["Données brutes", "Features RFM"])

        with tab1:
            st.subheader("Distribution géographique")
            top_n = st.slider("Nombre de pays à afficher", 5, 20, 10)
            top_countries = df_clean.groupby('Country')['TotalPrice'].sum()\
                              .sort_values(ascending=False).head(top_n)

            fig, ax = plt.subplots(figsize=(12, 4))
            top_countries.plot(kind='bar', ax=ax, color='#1976D2', edgecolor='white')
            ax.set_title(f'Top {top_n} pays par CA', fontweight='bold')
            ax.set_xlabel('Pays')
            ax.set_ylabel('CA total (£)')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.subheader("Évolution mensuelle du CA")
            df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
            monthly = df_clean.groupby('YearMonth')['TotalPrice'].sum()

            fig, ax = plt.subplots(figsize=(14, 4))
            monthly.plot(ax=ax, color='#1565C0', linewidth=2, marker='o', markersize=4)
            ax.set_title('CA mensuel', fontweight='bold')
            ax.fill_between(range(len(monthly)), monthly.values, alpha=0.1, color='#1565C0')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            st.subheader("Distributions RFM")
            col1, col2, col3 = st.columns(3)

            for col, feature, color in zip(
                [col1, col2, col3],
                ['Recency', 'Frequency', 'Monetary'],
                ['#1976D2', '#E53935', '#388E3C']
            ):
                with col:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(rfm[feature], bins=50, color=color, edgecolor='white', alpha=0.85)
                    ax.set_title(feature, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.metric(f"Médiane {feature}", f"{rfm[feature].median():.1f}")

            st.subheader("Statistiques descriptives RFM")
            st.dataframe(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Clustering":
    st.title("🤖 Segmentation K-Means (K=4)")

    if data_loaded and kmeans is not None:

        # KPIs par segment
        st.subheader("Profils des segments")
        cols = st.columns(4)
        for k, (label, emoji, color, desc) in CLUSTER_LABELS.items():
            cluster_data = rfm[rfm['Cluster'] == k]
            if len(cluster_data) == 0:
                continue
            with cols[k]:
                st.markdown(f"### {emoji} {label}")
                st.metric("Clients", f"{len(cluster_data):,}")
                st.metric("Recency moy.", f"{cluster_data['Recency'].mean():.0f}j")
                st.metric("Frequency moy.", f"{cluster_data['Frequency'].mean():.1f}")
                st.metric("Monetary moy.", f"£{cluster_data['Monetary'].mean():.0f}")

        st.markdown("---")

        # Scatter plots
        st.subheader("Clusters dans l'espace RFM")
        x_feat = st.selectbox("Axe X", ['R_scaled', 'F_scaled', 'M_scaled'], index=0)
        y_feat = st.selectbox("Axe Y", ['R_scaled', 'F_scaled', 'M_scaled'], index=1)

        cluster_colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3']
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(4):
            mask = labels == k
            label_name = CLUSTER_LABELS[k][0]
            ax.scatter(rfm_scaled.loc[mask, x_feat], rfm_scaled.loc[mask, y_feat],
                       c=cluster_colors[k], alpha=0.4, s=10, label=label_name)
        ax.set_xlabel(x_feat, fontsize=12)
        ax.set_ylabel(y_feat, fontsize=12)
        ax.set_title(f'Clusters K-Means : {x_feat} vs {y_feat}', fontweight='bold')
        ax.legend(markerscale=3, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Heatmap profils
        st.subheader("Heatmap des profils RFM moyens")
        profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(profile_norm.T, annot=profile.T, fmt='.0f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax)
        ax.set_title('Profils RFM moyens par cluster', fontweight='bold')
        ax.set_yticklabels(['Recency', 'Frequency', 'Monetary'], rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Table détaillée
        st.subheader("Données brutes par segment")
        segment_filter = st.selectbox(
            "Filtrer par segment",
            ['Tous'] + [v[0] for v in CLUSTER_LABELS.values()]
        )
        if segment_filter == 'Tous':
            display_df = rfm[['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Segment']]
        else:
            display_df = rfm[rfm['Segment'] == segment_filter][
                ['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Segment']]

        st.dataframe(display_df.head(100).reset_index(drop=True))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Prédiction client":
    st.title("🔮 Prédire le segment d'un nouveau client")
    st.markdown("Entrez les valeurs RFM d'un client pour prédire son segment.")

    if data_loaded and kmeans is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📥 Paramètres client")
            recency   = st.slider("Recency (jours depuis dernier achat)", 1, 365, 30)
            frequency = st.slider("Frequency (nb de commandes)", 1, 50, 5)
            monetary  = st.slider("Monetary (CA total £)", 10, 10000, 500)

            st.markdown("---")
            predict_btn = st.button("🔮 Prédire le segment", type="primary", use_container_width=True)

        with col2:
            st.subheader("📊 Résultat")

            if predict_btn:
                # Normalisation
                import numpy as np
                x = np.array([[recency, frequency, monetary]], dtype=float)
                x_log = np.log1p(x)
                x_scaled = scaler.transform(x_log)

                # Prédiction
                cluster = kmeans.predict(x_scaled)[0]
                label, emoji, color, desc = CLUSTER_LABELS.get(cluster, (f"Cluster {cluster}", "❓", "#999", ""))

                st.markdown(f"## {emoji} {label}")
                st.markdown(f"*{desc}*")
                st.markdown("---")

                # Comparaison avec la moyenne du cluster
                cluster_avg = rfm[rfm['Cluster'] == cluster][['Recency', 'Frequency', 'Monetary']].mean()

                st.markdown("**Comparaison avec la moyenne du segment :**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    delta_r = recency - cluster_avg['Recency']
                    st.metric("Recency", f"{recency}j",
                              delta=f"{delta_r:+.0f}j vs moy.",
                              delta_color="inverse")
                with col_b:
                    delta_f = frequency - cluster_avg['Frequency']
                    st.metric("Frequency", f"{frequency}",
                              delta=f"{delta_f:+.1f} vs moy.")
                with col_c:
                    delta_m = monetary - cluster_avg['Monetary']
                    st.metric("Monetary", f"£{monetary}",
                              delta=f"£{delta_m:+.0f} vs moy.")

                st.markdown("---")

                # Recommandation marketing
                st.markdown("**💡 Recommandation marketing :**")
                reco = {
                    "Champions":     "Récompensez ce client ! Programme VIP, offres exclusives, early access.",
                    "Clients perdus": "Campagne de réactivation : email personnalisé + offre de retour.",
                    "Réguliers":     "Programme de fidélité : inciter à augmenter la fréquence d'achat.",
                    "Nouveaux":      "Onboarding : guide de découverte + offre de bienvenue sur 2e commande.",
                }
                st.info(reco.get(label, "Analyser le profil manuellement."))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PERFORMANCES
# ════════════════════════════════════════════════════════════════════════════

elif page == "📈 Performances":
    st.title("📈 Comparaison des modèles")

    if data_loaded:
        metrics_path = ROOT / 'results' / 'model_metrics.csv'
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)

            st.subheader("Métriques comparatives")
            st.dataframe(metrics_df, use_container_width=True)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(7, 4))
                colors = ['#1976D2', '#E53935', '#388E3C']
                bars = ax.bar(metrics_df['Modèle'], metrics_df['Silhouette'],
                              color=colors, edgecolor='white')
                ax.set_title('Silhouette Score', fontweight='bold')
                ax.set_ylabel('Score')
                for bar, val in zip(bars, metrics_df['Silhouette']):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.002,
                            f'{val:.4f}', ha='center', fontweight='bold')
                ax.tick_params(axis='x', rotation=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(metrics_df['Modèle'], metrics_df['Nb clusters'],
                       color=colors, edgecolor='white')
                ax.set_title('Nombre de clusters', fontweight='bold')
                ax.set_ylabel('Nb clusters')
                ax.tick_params(axis='x', rotation=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("---")
            st.subheader("🏆 Conclusion")
            best = metrics_df.loc[metrics_df['Silhouette'].idxmax(), 'Modèle']
            best_score = metrics_df['Silhouette'].max()
            st.success(f"**Meilleur modèle : {best}** avec un Silhouette Score de {best_score:.4f}")
            st.markdown("""
            K-Means (K=4) offre le meilleur équilibre entre :
            - **Performance** : Silhouette Score le plus élevé
            - **Interprétabilité** : segments clairs et actionnables business
            - **Scalabilité** : peut prédire le segment de nouveaux clients
            """)
        else:
            st.warning("Fichier results/model_metrics.csv non trouvé. Exécutez notebooks/models.ipynb d'abord.")
