from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# ── Modèles entraînés ──────────────────────────────────────────────────────
# Trois modèles de clustering RFM : K-Means, DBSCAN, Agglomerative Clustering
# Chaque modèle est sérialisé en .pkl dans le dossier models/

MODELS = {
    "kmeans": {
        "name": "K-Means (K=4)",
        "description": "Clustering par partitionnement. Divise les clients en K=4 groupes en minimisant l'inertie intra-cluster. Algorithme de référence pour la segmentation RFM.",
        "path": MODELS_DIR / "kmeans.pkl",
    },
    "dbscan": {
        "name": "DBSCAN",
        "description": "Clustering par densité. Identifie des clusters de forme arbitraire et détecte automatiquement les outliers (bruit). Ne nécessite pas de spécifier K à l'avance.",
        "path": MODELS_DIR / "dbscan.pkl",
    },
    "agglomerative": {
        "name": "Agglomerative Clustering (Ward)",
        "description": "Clustering hiérarchique ascendant avec linkage Ward. Construit un dendrogramme permettant de choisir le nombre de clusters a posteriori.",
        "path": MODELS_DIR / "agglomerative.pkl",
    },
}
