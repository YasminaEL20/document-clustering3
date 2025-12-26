import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def find_csv():
    candidates = [
        "data/bbc-text-cleaned.csv",
        "notebook/data/bbc-text-cleaned.csv",
        "data/bbc-text.csv",
        "notebook/data/bbc-text.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Aucun fichier CSV trouvé dans : " + ", ".join(candidates)
    )


def build_documents(df):
    docs = []
    for i, row in df.iterrows():
        titre = row.get('title') or row.get('titre') or f"Document {i+1}"
        contenu = (
            row.get('clean_text')
            or row.get('text')
            or row.get('contenu')
            or row.get(df.columns[0])
        )
        theme = row.get('category') or row.get('theme') or ''
        docs.append({
            'titre': titre,
            'contenu': contenu,
            'theme': theme
        })
    return docs


def train_and_save(k=5, out_dir='data'):
    csv_path = find_csv()
    print(f"Chargement des données depuis {csv_path}")
    df = pd.read_csv(csv_path)

    texts = df['clean_text'] if 'clean_text' in df.columns else df.iloc[:, 0]

    # =====================================================
    # 🔹 TF-IDF OPTIMISÉ MÉMOIRE
    # =====================================================
    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        max_features=3000,      # 🔴 CRUCIAL
        ngram_range=(1, 1),     # 🔴 UNIGRAMS SEULEMENT
        stop_words='english',
        dtype=np.float32        # 🔴 RAM ÷ 2
    )

    print("Entraînement du TF-IDF...")
    X_tfidf = vectorizer.fit_transform(texts)

    # =====================================================
    # 🔹 K-MEANS
    # =====================================================
    print(f"Entraînement de KMeans (k={k})...")
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_tfidf)

    documents = build_documents(df)

    # =====================================================
    # 🔹 PCA 2D (PRÉ-CALCULÉ POUR L'INTERFACE)
    # =====================================================
    print("Calcul de la PCA 2D...")
    pca = PCA(n_components=2, random_state=42)
    reduced_2d = pca.fit_transform(X_tfidf)

    # =====================================================
    # 🔹 SAUVEGARDE
    # =====================================================
    os.makedirs(out_dir, exist_ok=True)
    print(f"Sauvegarde des artefacts dans {out_dir}/")

    joblib.dump(kmeans, os.path.join(out_dir, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(out_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(documents, os.path.join(out_dir, 'documents.pkl'))
    joblib.dump(X_tfidf, os.path.join(out_dir, 'tfidf_matrix.pkl'))
    joblib.dump(reduced_2d, os.path.join(out_dir, 'pca_2d.pkl'))

    print("✅ Terminé. Fichiers générés :")
    print("- model.pkl")
    print("- tfidf_vectorizer.pkl")
    print("- documents.pkl")
    print("- tfidf_matrix.pkl")
    print("- pca_2d.pkl")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Entraîne TF-IDF, KMeans et PCA puis sauvegarde les artefacts.'
    )
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--out', type=str, default='data')
    args = parser.parse_args()

    train_and_save(k=args.k, out_dir=args.out)
