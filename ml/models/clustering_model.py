"""
Clustering Model — TF-IDF + KMeans
Groups similar products for analog-based predictions
"""

import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_texts_for_clustering(products):
    """Extract clean text for clustering"""
    texts = []
    for p in products:
        text = (
            str(p.get("title", "")) + " " +
            str(p.get("description", "")) + " " +
            str(p.get("category", ""))
        )
        texts.append(text.strip().lower())
    return texts


class ClusteringModel:
    """TF-IDF + KMeans clustering"""

    def __init__(self, config=None):
        self.config = config or {"n_clusters": 6}
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1500
        )
        self.model = KMeans(
            n_clusters=self.config["n_clusters"],
            random_state=42
        )
        self.is_trained = False

    def train(self, texts):
        X_vec = self.vectorizer.fit_transform(texts)
        self.model.fit(X_vec)
        self.is_trained = True

    def predict(self, texts):
        if not self.is_trained:
            raise ValueError("Clustering model must be trained before prediction.")
        
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "model": self.model,
                "config": self.config,
                "is_trained": self.is_trained,
            }, f)

    def load(self, filepath):
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.config = data["config"]
        self.is_trained = data["is_trained"]
