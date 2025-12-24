# Document Clustering — BBC News

Projet pédagogique et prototype d’un pipeline de clustering de documents textuels (corpus BBC News) avec notebooks d’analyse, scripts reproductibles et une application web Flask pour la visualisation et la recherche de documents similaires.

---

## 1. Titre du projet

Document Clustering — BBC News

Prototype et démonstrateur d’un pipeline de traitement de texte (NLP) : TF‑IDF → K‑Means → visualisation et recherche par similarité.

---

## 2. Objectifs du projet

- Fournir un pipeline reproductible pour le clustering non supervisé d’articles de presse.
- Documenter chaque étape (exploration, prétraitement, vectorisation, clustering, visualisation).
- Produire des artefacts réutilisables (vectorizer, matrice TF‑IDF, modèle K‑Means, index de documents) pour une application web.
- Offrir une interface web simple pour explorer les clusters et rechercher des documents similaires.
- Servir de base pour travaux ultérieurs (embeddings, indexation, evaluation).

---

## 3. Technologies & bibliothèques (rôle)

- `Python` : langage principal.
- `pandas` : lecture/manipulation des jeux de données.
- `numpy`, `scipy` : calculs numériques et structures.
- `scikit-learn` : TF‑IDF (`TfidfVectorizer`), K‑Means, PCA, métriques (cosine_similarity).
- `joblib` : sérialisation des artefacts (`.pkl`) pour réutilisation par l’application.
- `nltk` : ressources NLP (stopwords, tokenization, lemmatisation si nécessaire).
- `matplotlib` : visualisations dans les notebooks (courbes, nuages de points).
- `flask` : application web (routes, rendu de templates).
- `nbformat`, `nbconvert`, `nbclient`, `ipykernel`, `jupyter` : exécution et automatisation des notebooks.

Voir `requirements.txt` pour la liste exacte des paquets et versions recommandées.

---

## 4. Architecture globale

Le projet est structuré pour séparer la recherche exploratoire et la production reproducible :

- `notebook/` : notebooks Jupyter (étapes pédagogiques et expérimentales).
- `scripts/` : scripts réutilisables pour automatiser l’entraînement et la génération d’artefacts (ex. `scripts/train_models.py`).
- `data/` : jeux de données bruts et artefacts sauvegardés (`.pkl`).
- `templates/` et `static/` : frontend de l’application Flask (HTML, CSS, JS).
- `app.py` : application Flask qui charge les artefacts et sert les pages et APIs.

Cette organisation facilite : reproduction, automatisation, et évolution (tests, CI, remplacement de composants).

---

## 5. Arborescence du projet et description

Extrait de l’arborescence :

```
app.py
requirements.txt
scripts/
  train_models.py
notebook/
  01_exploration.ipynb
  02_preprocessing.ipynb
  03_tfidf.ipynb
  04_kmeans.ipynb
  05_visualisation.ipynb
  06_similarity_search.ipynb
data/
  bbc-text.csv
  bbc-text-cleaned.csv
  model.pkl
  tfidf_vectorizer.pkl
  documents.pkl
  tfidf_matrix.pkl
templates/
  base.html
  index.html
  clusters.html
  search.html
  stats.html
static/
  css/
  js/
```

- `data/` : contient les CSV sources et les artefacts ML générés par `scripts/train_models.py` ou par l’exécution des notebooks. `documents.pkl` contient la liste des documents (titre, contenu, thème) utilisée par l’UI.
- `notebook/` : séquence pédagogique, utile pour comprendre les choix méthodologiques et tester d’autres variantes.
- `scripts/train_models.py` : script automatisé pour entraîner TF‑IDF + K‑Means et sauvegarder `model.pkl`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl`, `documents.pkl` dans `data/`.
- `templates/` : pages HTML Jinja2; `base.html` définit le layout et intègre les assets statiques.
- `static/` : éléments JavaScript (visualisations interactives) et CSS (styles, Tailwind si utilisé).
- `app.py` : charge les artefacts en mémoire au démarrage et expose les routes web et APIs JSON.

---

## 6. Pipeline Data Science — description détaillée

1) Exploration des données

- Lecture du CSV (`pandas.read_csv`).
- Vérification des colonnes, distribution des classes, longueur des documents.
- Analyse descriptive : counts, histogrammes, quelques exemples d’articles.

2) Prétraitement

- Nettoyage du texte : minuscules, suppression de la ponctuation et des caractères non désirés.
- Suppression des stopwords (NLTK ou liste personnalisée).
- Optionnel : lemmatisation/stemming pour normaliser les formes.
- Sauvegarde du champ `clean_text` dans un CSV dédié (`bbc-text-cleaned.csv`).

3) Vectorisation TF‑IDF

- `TfidfVectorizer` (scikit‑learn) avec paramètres par défaut du projet :
  - `max_df=0.9`
  - `min_df=5`
  - `ngram_range=(1,2)`
  - `stop_words='english'`
- Résultat : matrice creuse TF‑IDF `X_tfidf`.

4) Clustering K‑Means

- Choix de `k` à l’aide de la méthode du coude (Elbow method).
- Entraînement final : `KMeans(n_clusters=k, random_state=42, n_init=10)`.
- Assignation des labels au DataFrame (colonne `cluster`).

5) Réduction de dimension

- PCA pour réduction rapide et stable.
- t‑SNE ou UMAP pour visualisations non linéaires plus informatives (coût computationnel plus élevé).

6) Recherche par similarité

- Transforme la requête utilisateur avec le `TfidfVectorizer` sauvegardé.
- Calcule `cosine_similarity` entre vecteur requête et matrice TF‑IDF.
- Renvoie les indices des documents les plus proches (top‑k) pour affichage.

---

## 7. Application web Flask — pages & API

- `/` — Page d’accueil : résumé et navigation.
- `/stats` — Page statistiques : KPI et graphiques (nombre de documents, clusters, distribution thèmes).
- `/clusters` — Visualisation 2D interactive (PCA / t‑SNE) : chaque point correspond à un document, survol affiche titre/extrait.
- `/search` — Recherche textuelle : formulaire, résultats triés par similarité cosinus.
- API :
  - `/api/stats` : JSON pour graphiques statistiques.
  - `/api/clusters` : JSON des coordonnées 2D et métadonnées pour la visualisation.

L’application charge en mémoire les artefacts présents dans `data/` pour répondre rapidement aux requêtes.

---

## 8. Installation & exécution

Prérequis : Python 3.9+ recommandé.

1) Cloner le dépôt

```bash
git clone <repo-url>
cd <project-root>
```

2) Créer et activer un environnement virtuel

- Windows (PowerShell) :

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

- macOS / Linux :

```bash
python3 -m venv venv
source venv/bin/activate
```

3) Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4) Préparer les données

- Placer `bbc-text.csv` (ou `bbc-text-cleaned.csv`) dans `data/`.
- Si vous n’avez que `bbc-text.csv`, exécutez le prétraitement (notebook `02_preprocessing.ipynb`) pour générer `bbc-text-cleaned.csv` contenant `clean_text`.

5) Entraîner et générer les artefacts

Option A — Script automatisé (recommandé) :

```bash
python scripts/train_models.py --k 5 --out data
```

Option B — Exécuter les notebooks (répétable et pédagogique) :

```bash
jupyter nbconvert --to notebook --execute notebook/03_tfidf.ipynb --ExecutePreprocessor.timeout=600 --output notebook/03_tfidf.executed.ipynb
jupyter nbconvert --to notebook --execute notebook/04_kmeans.ipynb --ExecutePreprocessor.timeout=1200 --output notebook/04_kmeans.executed.ipynb
```

Les notebooks incluent des cellules qui sauvegardent `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl`, `model.pkl` et `documents.pkl` dans `data/`.

6) Lancer l’application

```bash
python app.py
# Ouvrir http://127.0.0.1:5000/
```

---

## 9. Guide utilisateur — utilisation rapide

1. Accéder à l’accueil pour contexte et liens.
2. Consulter `Statistiques` pour avoir un aperçu global (distribution thèmes vs clusters).
3. Aller dans `Clusters` pour explorer visuellement les groupes ; cliquer/survoler pour lire un extrait.
4. Utiliser `Recherche` pour trouver des articles similaires à une requête (phrase courte ou mots‑clés).

Conseils : utiliser des expressions régulièrement utilisées dans la presse (noms propres, lieux) pour des résultats plus robustes avec TF‑IDF.

---

## 10. Guide développeur — pour continuer le projet

Priorités possibles et points d’entrée :

- **Prétraitement** : modifier `notebook/02_preprocessing.ipynb` pour améliorer nettoyage (entités nommées, regex, normalisation typographique).
- **Vectorisation** : remplacer TF‑IDF par embeddings (Sentence‑BERT, Universal Sentence Encoder) pour similarité sémantique.
- **Clustering** : tester `HDBSCAN`, `GaussianMixture`, `MiniBatchKMeans` pour scalabilité et robustesse.
- **Visualisation** : comparer PCA / t‑SNE / UMAP, ajouter contours, labels automatiques.
- **App** : ajouter authentification, pagination des résultats, API RESTful documentée, endpoints d’annotation.

Bonnes pratiques :

- Versionner les dépendances (`requirements.txt` ou `pip-tools`, `poetry`).
- Ajouter des tests unitaires (prétraitement, transformation, recherche).
- Logger les événements et erreurs (`logging`).
- Ajouter un `Dockerfile` pour déploiement reproductible.

---

## 11. Remarques sur choix techniques

- TF‑IDF + K‑Means : baseline rapide et interprétable, adapté pour corpus homogène et démonstrations pédagogiques.
- Limites : TF‑IDF ne capture pas le contexte profond ; les embeddings contextualisés (BERT, SBERT) donnent de meilleurs résultats pour similarité sémantique.
- Méthode du coude pour K : heuristique utile ; compléter par silhouette score ou évaluation manuelle.
- t‑SNE/UMAP sont coûteux mais souvent nécessaires pour une représentation visuelle plus lisible.

---

## 12. Perspectives & améliorations

- Intégrer `sentence-transformers` (SBERT) pour embeddings documents.
- Utiliser `FAISS` ou `Annoy` pour recherche de similarité approximative et scalabilité.
- Mettre en place CI (tests, lint, exécution notebooks) et packaging Docker.
- Ajouter interface d’étiquetage humain pour créer dataset supervisé et affiner clustering.

---

## Annexes — commandes utiles

- Créer venv & installer :
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

- Lancer entraînement :
```
python scripts/train_models.py --k 5 --out data
```

- Exécuter notebooks :
```
jupyter nbconvert --to notebook --execute notebook/03_tfidf.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute notebook/04_kmeans.ipynb --ExecutePreprocessor.timeout=1200
```

---

## Réalisé par

Projet académique réalisé par **Salsabil Elkhlouf** et **Yasmina El Mansouri**.

Encadrement : **Mme Manar Kassou**.



