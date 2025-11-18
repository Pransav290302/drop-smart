"""Product clustering module using embeddings and clustering algorithms"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan

logger = logging.getLogger(__name__)


class ProductClustering:
    """
    Product clustering module that:
    - Generates embeddings using MiniLM/SentenceTransformers (FR-13)
    - Runs k-means or HDBSCAN for clustering (FR-14)
    - Computes cluster-level success rates (FR-15)
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        clustering_method: str = "kmeans",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize product clustering.
        
        Args:
            embedding_model: SentenceTransformer model name
            clustering_method: Clustering method ("kmeans" or "hdbscan")
            config: Clustering configuration dictionary
        """
        self.embedding_model_name = embedding_model
        self.clustering_method = clustering_method.lower()
        
        if self.clustering_method not in ["kmeans", "hdbscan"]:
            raise ValueError(f"Invalid clustering_method: {clustering_method}. Must be 'kmeans' or 'hdbscan'")
        
        self.config = config or {}
        self.embedding_model = None
        self.clusterer = None
        self.scaler = None
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.is_fitted = False
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize clusterer
        if self.clustering_method == "kmeans":
            self._init_kmeans()
        else:
            self._init_hdbscan()
    
    def _init_embedding_model(self) -> None:
        """Initialize SentenceTransformer model for embeddings."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized SentenceTransformer model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _init_kmeans(self) -> None:
        """Initialize k-means clusterer."""
        n_clusters = self.config.get("n_clusters", 10)
        random_state = self.config.get("random_state", 42)
        
        self.clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        logger.info(f"Initialized k-means clusterer with n_clusters={n_clusters}")
    
    def _init_hdbscan(self) -> None:
        """Initialize HDBSCAN clusterer."""
        min_cluster_size = self.config.get("min_cluster_size", 5)
        min_samples = self.config.get("min_samples", 3)
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        logger.info(f"Initialized HDBSCAN clusterer with min_cluster_size={min_cluster_size}")
    
    def generate_embeddings(
        self,
        texts: Union[List[str], pd.Series],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings from product titles/descriptions using SentenceTransformers.
        
        FR-13: Generate embeddings using MiniLM / SentenceTransformers.
        
        Args:
            texts: List of product titles/descriptions
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        
        # Convert to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        if not texts:
            raise ValueError("Texts list is empty")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better clustering
        )
        
        self.embeddings = embeddings
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def fit(
        self,
        texts: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> None:
        """
        Fit clustering model on product texts.
        
        Args:
            texts: List of product titles/descriptions
            embeddings: Pre-computed embeddings (optional, will generate if None)
            batch_size: Batch size for embedding generation
        """
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        else:
            self.embeddings = embeddings
        
        logger.info(f"Fitting {self.clustering_method} clusterer on {len(embeddings)} samples")
        
        # Fit clusterer
        if self.clustering_method == "kmeans":
            self.clusterer.fit(embeddings)
            self.cluster_labels = self.clusterer.labels_
        else:  # HDBSCAN
            self.clusterer.fit(embeddings)
            self.cluster_labels = self.clusterer.labels_
            # HDBSCAN may assign -1 for noise points
            n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            n_noise = list(self.cluster_labels).count(-1)
            logger.info(f"HDBSCAN found {n_clusters} clusters with {n_noise} noise points")
        
        self.is_fitted = True
        logger.info(f"Clustering completed. Found {len(set(self.cluster_labels))} clusters")
    
    def predict(self, texts: Union[List[str], pd.Series], embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict cluster labels for new products.
        
        Args:
            texts: List of product titles/descriptions
            embeddings: Pre-computed embeddings (optional, will generate if None)
            
        Returns:
            Array of cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before making predictions")
        
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.generate_embeddings(texts)
        
        # Predict clusters
        if self.clustering_method == "kmeans":
            cluster_labels = self.clusterer.predict(embeddings)
        else:  # HDBSCAN
            # HDBSCAN doesn't have predict, use approximate_predict
            cluster_labels, strengths = hdbscan.approximate_predict(self.clusterer, embeddings)
        
        return cluster_labels
    
    def compute_cluster_success_rates(
        self,
        cluster_labels: np.ndarray,
        success_labels: Union[List[bool], np.ndarray, pd.Series],
        min_cluster_size: int = 3
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute cluster-level success rates for analog-based insights.
        
        FR-15: Use clusters to compute analog-based success rates for SKUs.
        
        Args:
            cluster_labels: Cluster assignments for each product
            success_labels: Binary success labels (True/False or 1/0)
            min_cluster_size: Minimum cluster size to include in results
            
        Returns:
            Dictionary mapping cluster_id to success rate statistics:
            {
                cluster_id: {
                    "success_rate": float,
                    "total_products": int,
                    "successful_products": int,
                    "failed_products": int,
                }
            }
        """
        if len(cluster_labels) != len(success_labels):
            raise ValueError(
                f"cluster_labels and success_labels must have same length. "
                f"Got {len(cluster_labels)} and {len(success_labels)}"
            )
        
        # Convert success labels to boolean
        if isinstance(success_labels, pd.Series):
            success_labels = success_labels.values
        success_labels = np.array(success_labels, dtype=bool)
        
        # Compute success rates per cluster
        cluster_stats = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            # Skip noise points (HDBSCAN uses -1)
            if cluster_id == -1:
                continue
            
            # Get products in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Skip small clusters
            if cluster_size < min_cluster_size:
                continue
            
            # Calculate success rate
            cluster_successes = success_labels[cluster_mask]
            successful_count = np.sum(cluster_successes)
            success_rate = successful_count / cluster_size if cluster_size > 0 else 0.0
            
            cluster_stats[int(cluster_id)] = {
                "success_rate": float(success_rate),
                "total_products": int(cluster_size),
                "successful_products": int(successful_count),
                "failed_products": int(cluster_size - successful_count),
            }
        
        logger.info(f"Computed success rates for {len(cluster_stats)} clusters")
        
        return cluster_stats
    
    def get_cluster_analogs(
        self,
        product_text: str,
        cluster_labels: np.ndarray,
        product_texts: List[str],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find analog products in the same cluster.
        
        Args:
            product_text: Text of the product to find analogs for
            cluster_labels: Cluster assignments for all products
            product_texts: List of all product texts
            top_n: Number of analogs to return
            
        Returns:
            List of analog products with their cluster information
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before finding analogs")
        
        # Get embedding for the product
        product_embedding = self.generate_embeddings([product_text])
        
        # Predict cluster for the product
        product_cluster = self.predict([product_text], embeddings=product_embedding)[0]
        
        # Find products in the same cluster
        same_cluster_mask = cluster_labels == product_cluster
        same_cluster_indices = np.where(same_cluster_mask)[0]
        
        if len(same_cluster_indices) == 0:
            logger.warning(f"No products found in cluster {product_cluster}")
            return []
        
        # Calculate similarity to other products in the cluster
        cluster_embeddings = self.embeddings[same_cluster_indices]
        similarities = np.dot(cluster_embeddings, product_embedding[0])
        
        # Get top N most similar
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        analogs = []
        for idx in top_indices:
            original_idx = same_cluster_indices[idx]
            analogs.append({
                "index": int(original_idx),
                "text": product_texts[original_idx],
                "cluster_id": int(product_cluster),
                "similarity": float(similarities[idx]),
            })
        
        return analogs
    
    def get_cluster_summary(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Get summary statistics about clusters.
        
        Args:
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with cluster summary statistics
        """
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        
        # Filter out noise points
        if -1 in unique_clusters:
            noise_idx = np.where(unique_clusters == -1)[0][0]
            noise_count = counts[noise_idx]
            unique_clusters = np.delete(unique_clusters, noise_idx)
            counts = np.delete(counts, noise_idx)
        else:
            noise_count = 0
        
        cluster_sizes = dict(zip(unique_clusters.tolist(), counts.tolist()))
        
        return {
            "n_clusters": len(unique_clusters),
            "n_noise_points": int(noise_count),
            "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
            "min_cluster_size": int(counts.min()) if len(counts) > 0 else 0,
            "max_cluster_size": int(counts.max()) if len(counts) > 0 else 0,
            "mean_cluster_size": float(counts.mean()) if len(counts) > 0 else 0.0,
        }
    
    def fit_predict(
        self,
        texts: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Fit clusterer and return cluster labels in one step.
        
        Args:
            texts: List of product titles/descriptions
            embeddings: Pre-computed embeddings (optional)
            batch_size: Batch size for embedding generation
            
        Returns:
            Array of cluster labels
        """
        self.fit(texts, embeddings=embeddings, batch_size=batch_size)
        return self.cluster_labels
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save clustering model to disk.
        
        Note: SentenceTransformer model is not saved (will be reloaded from name).
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "embedding_model_name": self.embedding_model_name,
            "clustering_method": self.clustering_method,
            "clusterer": self.clusterer,
            "embeddings": self.embeddings,
            "cluster_labels": self.cluster_labels,
            "is_fitted": self.is_fitted,
            "config": self.config,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Clustering model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load clustering model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.embedding_model_name = model_data["embedding_model_name"]
        self.clustering_method = model_data["clustering_method"]
        self.clusterer = model_data["clusterer"]
        self.embeddings = model_data.get("embeddings")
        self.cluster_labels = model_data.get("cluster_labels")
        self.is_fitted = model_data["is_fitted"]
        self.config = model_data.get("config", {})
        
        # Reinitialize embedding model
        self._init_embedding_model()
        
        logger.info(f"Clustering model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the clustering model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "embedding_model": self.embedding_model_name,
            "clustering_method": self.clustering_method,
            "is_fitted": self.is_fitted,
            "config": self.config,
        }
        
        if self.is_fitted and self.cluster_labels is not None:
            info["n_clusters"] = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            info["n_samples"] = len(self.cluster_labels)
        
        return info


def create_product_text(
    product_name: str,
    description: Optional[str] = None,
    category: Optional[str] = None
) -> str:
    """
    Create combined text from product information for embedding.
    
    Args:
        product_name: Product name/title
        description: Product description (optional)
        category: Product category (optional)
        
    Returns:
        Combined text string
    """
    text_parts = [product_name]
    
    if description:
        text_parts.append(description)
    
    if category:
        text_parts.append(category)
    
    return " ".join(text_parts)


def prepare_texts_for_clustering(
    products: List[Dict[str, Any]],
    name_field: str = "product_name",
    description_field: str = "description",
    category_field: str = "category"
) -> List[str]:
    """
    Prepare product texts for clustering from product dictionaries.
    
    Args:
        products: List of product dictionaries
        name_field: Field name for product name
        description_field: Field name for description
        category_field: Field name for category
        
    Returns:
        List of combined text strings
    """
    texts = []
    for product in products:
        product_name = product.get(name_field, "")
        description = product.get(description_field)
        category = product.get(category_field)
        text = create_product_text(product_name, description, category)
        texts.append(text)
    
    return texts

