from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================================================
# 🔹 CHARGEMENTS GLOBAUX (UNE SEULE FOIS AU DÉMARRAGE)
# =========================================================

# Modèle K-Means
model = joblib.load('data/model.pkl')

# Vectoriseur TF-IDF
vectorizer = joblib.load('data/tfidf_vectorizer.pkl')

# Documents (liste de dicts)
documents = joblib.load('data/documents.pkl')

# Matrice TF-IDF (SPARSE, NE PAS CONVERTIR EN ARRAY)
tfidf_matrix = joblib.load('data/tfidf_matrix.pkl')

# Labels des clusters
labels = model.labels_

# PCA 2D PRÉ-CALCULÉ (IMPORTANT : pas de PCA ici)
reduced_2d = joblib.load('data/pca_2d.pkl')

# =========================================================
# 🔹 STATISTIQUES GLOBALES (calculées UNE FOIS)
# =========================================================

total_documents = len(documents)
nb_clusters = len(set(labels))

theme_series = pd.Series([doc['theme'] for doc in documents])
theme_counts = theme_series.value_counts()
theme_dominant = theme_counts.idxmax()
pourcentage_dominant = (theme_counts.max() / total_documents) * 100

cluster_series = pd.Series(labels)
cluster_counts = cluster_series.value_counts()

interpretation_principale = (
    f"Le clustering K-Means a identifié {nb_clusters} groupes, proches des catégories réelles du corpus BBC. "
    f"La catégorie dominante est {theme_dominant} avec {pourcentage_dominant:.1f}% des documents. "
    "La répartition montre une bonne cohérence globale entre les thèmes réels et les clusters découverts."
)

# =========================================================
# 🔹 ROUTES
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stats')
def stats():
    return render_template(
        'stats.html',
        total_documents=total_documents,
        nb_clusters=nb_clusters,
        theme_dominant=theme_dominant,
        pourcentage_dominant=round(pourcentage_dominant, 1),
        interpretation_principale=interpretation_principale
    )


@app.route('/api/stats')
def api_stats():
    return jsonify({
        'themes_labels': theme_counts.index.tolist(),
        'themes_values': theme_counts.values.tolist(),
        'clusters_labels': [f"Cluster {i}" for i in cluster_counts.index],
        'clusters_values': cluster_counts.values.tolist()
    })


@app.route('/clusters')
def clusters():
    return render_template('clusters.html')


# =========================================================
# 🔹 API CLUSTERS 2D (SANS PCA, SANS CALCUL)
# =========================================================
@app.route('/api/clusters')
def api_clusters():
    return jsonify({
        'x': reduced_2d[:, 0].tolist(),
        'y': reduced_2d[:, 1].tolist(),
        'titles': [doc['titre'] for doc in documents],
        'contents': [doc['contenu'][:150] + '...' for doc in documents],
        'themes': [doc['theme'] for doc in documents],
        'clusters': labels.tolist()
    })


# =========================================================
# 🔹 RECHERCHE (OPTIMISÉE MÉMOIRE)
# =========================================================
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()

        if not query:
            return render_template(
                'search.html',
                results=[],
                error="Veuillez entrer une requête."
            )

        query_vec = vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vec,
            tfidf_matrix,
            dense_output=False
        ).flatten()

        top_indices = np.argsort(similarities)[-8:][::-1]

        results = [
            {
                'titre': documents[i]['titre'],
                'contenu': documents[i]['contenu'][:350] + '...',
                'similarity': round(float(similarities[i]), 3),
                'theme': documents[i]['theme']
            }
            for i in top_indices if similarities[i] > 0.05
        ]

        return render_template(
            'search.html',
            results=results,
            query=query
        )

    return render_template('search.html', results=[])


# =========================================================
# 🔹 LANCEMENT LOCAL (PAS UTILISÉ PAR RENDER)
# =========================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
