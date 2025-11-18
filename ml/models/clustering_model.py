"""
Clustering Model — TF-IDF + KMeans
Groups similar products for analog-based predictions
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.models.base_model import BaseModel


class ClusteringModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1500,
            ngram_range=(1, 1)
        )
        self.model = KMeans(
            n_clusters=self.config.get("n_clusters", 6),
            random_state=42
        )

    def train(self, product_texts):
        if len(product_texts) == 0:
            raise ValueError("No product texts provided")

        X_vec = self.vectorizer.fit_transform(product_texts)
        self.model.fit(X_vec)

        self.is_trained = True
        print("✔ Clustering Model trained (TF-IDF + KMeans)")

    def predict(self, product_texts):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_vec = self.vectorizer.transform(product_texts)
        return self.model.predict(X_vec)

    def save(self, filepath):
        """Save both vectorizer + kmeans model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "config": self.config,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        print(f"✔ Saved clustering model → {filepath}")

    def load(self, filepath):
        """Load both components."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.config = data["config"]
        self.is_trained = data["is_trained"]

        print(f"✔ Loaded clustering model → {filepath}")
