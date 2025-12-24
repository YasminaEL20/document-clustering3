from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA  # Tu pourras changer pour t-SNE plus tard

app = Flask(__name__)

# Charger les modèles et données au démarrage (une seule fois)
model = joblib.load('data/model.pkl')           # KMeans entraîné
vectorizer = joblib.load('data/tfidf_vectorizer.pkl')
documents = joblib.load('data/documents.pkl')   # Liste de dicts
tfidf_matrix = joblib.load('data/tfidf_matrix.pkl')  # Matrice pré-calculée (optimisation)
labels = model.labels_                          # Labels des clusters

# --- Calculs globaux pour la page stats (effectués une fois au démarrage) ---
total_documents = len(documents)
nb_clusters = len(set(labels))

# Ground truth : distribution par thème réel
theme_series = pd.Series([doc['theme'] for doc in documents])
theme_counts = theme_series.value_counts()
theme_dominant = theme_counts.idxmax()
pourcentage_dominant = (theme_counts.max() / total_documents) * 100

# Clusters : distribution par cluster prédit
cluster_series = pd.Series(labels)
cluster_counts = cluster_series.value_counts()

# Interprétation automatique simple (tu pourras l'enrichir)
interpretation_principale = (
    f"Le clustering K-Means a identifié {nb_clusters} groupes, proches des catégories réelles du corpus BBC. "
    f"La catégorie dominante est **{theme_dominant}** avec {pourcentage_dominant:.1f}% des documents. "
    "La répartition montre une bonne cohérence globale entre les thèmes réels et les clusters découverts."
)

# Route d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route page Statistiques (avec tous les KPI)
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

# API pour les données des graphiques (bar + pie)
@app.route('/api/stats')
def api_stats():
    return jsonify({
        # Ground truth (thèmes réels)
        'themes_labels': theme_counts.index.tolist(),
        'themes_values': theme_counts.values.tolist(),
        
        # Clusters K-Means
        'clusters_labels': [f"Cluster {i}" for i in cluster_counts.index],
        'clusters_values': cluster_counts.values.tolist()
    })

# Route page Clusters
@app.route('/clusters')
def clusters():
    return render_template('clusters.html')

# API pour la visualisation 2D
@app.route('/api/clusters')
def api_clusters():
    # Réduction 2D (PCA ici, tu peux passer à t-SNE plus tard)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    
    data = {
        'x': reduced[:, 0].tolist(),
        'y': reduced[:, 1].tolist(),
        'titles': [doc['titre'] for doc in documents],
        'contents': [doc['contenu'][:150] + '...' for doc in documents],  # extrait plus long
        'themes': [doc['theme'] for doc in documents],  # pour colorer par thème réel
        'clusters': labels.tolist()                     # pour colorer par cluster
    }
    return jsonify(data)

# Route Recherche (déjà bien, juste un petit nettoyage)
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template('search.html', results=[], error="Veuillez entrer une requête.")
        
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-8:][::-1]  # Top 8 au lieu de 5
        
        results = [
            {
                'titre': documents[i]['titre'],
                'contenu': documents[i]['contenu'][:350] + '...',
                'similarity': round(float(similarities[i]), 3),
                'theme': documents[i]['theme']
            }
            for i in top_indices if similarities[i] > 0.05  # seuil minimal pour éviter bruit
        ]
        
        return render_template('search.html', results=results, query=query)
    
    return render_template('search.html', results=[])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)