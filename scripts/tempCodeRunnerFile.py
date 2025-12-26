
        dtype=np.float32       
    )

    print("Entraînement du TF-IDF...")
    X_tfidf = vectorizer.fit_transform(texts)