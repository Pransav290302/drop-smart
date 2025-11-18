# Clustering/Embeddings Module Implementation

## ✅ Complete Implementation

### File Created

**`ml/models/clustering.py`** - Product clustering module with embeddings and clustering

## ProductClustering Class

### Features

- **FR-13**: Generates embeddings using MiniLM/SentenceTransformers
- **FR-14**: Runs k-means or HDBSCAN for grouping similar items
- **FR-15**: Computes cluster-level success rates for analog-based insights
- **Text Embeddings**: Uses SentenceTransformer for semantic embeddings
- **Dual Clustering**: Supports both k-means and HDBSCAN
- **Analog Products**: Find similar products in the same cluster

### Key Methods

#### `generate_embeddings(texts, batch_size=32, show_progress=False)`
- Generates embeddings from product titles/descriptions
- Uses SentenceTransformer (MiniLM by default)
- Returns normalized embeddings for better clustering
- FR-13: Generate embeddings using MiniLM / SentenceTransformers

#### `fit(texts, embeddings=None, batch_size=32)`
- Fits clustering model on product texts
- Generates embeddings if not provided
- Supports both k-means and HDBSCAN
- FR-14: Run k-means or HDBSCAN for grouping similar items

#### `predict(texts, embeddings=None)`
- Predicts cluster labels for new products
- Uses pre-computed embeddings if available

#### `compute_cluster_success_rates(cluster_labels, success_labels, min_cluster_size=3)`
- Computes success rates for each cluster
- Returns statistics per cluster:
  - Success rate
  - Total products
  - Successful products
  - Failed products
- FR-15: Use clusters to compute analog-based success rates

#### `get_cluster_analogs(product_text, cluster_labels, product_texts, top_n=5)`
- Finds analog products in the same cluster
- Returns most similar products with similarity scores

#### `get_cluster_summary(cluster_labels)`
- Returns summary statistics about clusters
- Includes cluster sizes, noise points (HDBSCAN), etc.

## Requirements Met

### FR-13: Generate Embeddings
✅ **MiniLM/SentenceTransformers**: Uses `sentence-transformers/all-MiniLM-L6-v2`  
✅ **Text Embeddings**: Generates embeddings from product titles/descriptions  
✅ **Normalized Embeddings**: Normalizes for better clustering performance  

### FR-14: Clustering
✅ **k-means**: Full k-means implementation with configurable clusters  
✅ **HDBSCAN**: Full HDBSCAN implementation with noise detection  
✅ **Configurable**: Both methods fully configurable via config  

### FR-15: Cluster Success Rates
✅ **Success Rate Calculation**: Computes success rates per cluster  
✅ **Analog-Based Insights**: Enables analog-based success rate analysis  
✅ **Statistics**: Provides detailed cluster statistics  

## Usage Examples

### Basic Clustering

```python
from ml.models.clustering import ProductClustering, prepare_texts_for_clustering

# Initialize clustering
clusterer = ProductClustering(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    clustering_method="kmeans",
    config={
        "n_clusters": 10,
        "random_state": 42,
    }
)

# Prepare product texts
products = [
    {"product_name": "Wireless Headphones", "description": "High-quality audio", "category": "Electronics"},
    {"product_name": "Bluetooth Earbuds", "description": "Wireless earbuds", "category": "Electronics"},
    # ... more products
]

texts = prepare_texts_for_clustering(products)

# Fit clustering
cluster_labels = clusterer.fit_predict(texts)

print(f"Found {len(set(cluster_labels))} clusters")
```

### Using HDBSCAN

```python
# Initialize HDBSCAN clusterer
clusterer = ProductClustering(
    clustering_method="hdbscan",
    config={
        "min_cluster_size": 5,
        "min_samples": 3,
    }
)

# Fit and predict
cluster_labels = clusterer.fit_predict(texts)

# Get summary (includes noise points)
summary = clusterer.get_cluster_summary(cluster_labels)
print(f"Clusters: {summary['n_clusters']}, Noise: {summary['n_noise_points']}")
```

### Computing Success Rates

```python
# Assume we have success labels (e.g., sold within 30 days)
success_labels = [True, False, True, True, False, ...]  # Same length as products

# Compute cluster success rates
success_rates = clusterer.compute_cluster_success_rates(
    cluster_labels,
    success_labels,
    min_cluster_size=3
)

# Print results
for cluster_id, stats in success_rates.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Total Products: {stats['total_products']}")
    print(f"  Successful: {stats['successful_products']}")
    print(f"  Failed: {stats['failed_products']}")
```

### Finding Analog Products

```python
# Find analogs for a specific product
product_text = "Wireless Bluetooth Headphones"
analogs = clusterer.get_cluster_analogs(
    product_text,
    cluster_labels,
    texts,  # All product texts
    top_n=5
)

print(f"Found {len(analogs)} analog products:")
for analog in analogs:
    print(f"  - {analog['text']} (similarity: {analog['similarity']:.3f})")
```

### Using Pre-computed Embeddings

```python
# Generate embeddings once
embeddings = clusterer.generate_embeddings(texts, batch_size=64)

# Use embeddings for multiple operations
cluster_labels = clusterer.fit_predict(texts, embeddings=embeddings)

# Predict for new products
new_texts = ["New Product Name"]
new_embeddings = clusterer.generate_embeddings(new_texts)
new_labels = clusterer.predict(new_texts, embeddings=new_embeddings)
```

### Integration with Product Data

```python
import pandas as pd

# Load product data
df = pd.DataFrame({
    "sku": ["SKU001", "SKU002", ...],
    "product_name": ["Product 1", "Product 2", ...],
    "description": ["Desc 1", "Desc 2", ...],
    "category": ["Cat 1", "Cat 2", ...],
    "sold_within_30_days": [1, 0, 1, ...],  # Success labels
})

# Prepare texts
texts = prepare_texts_for_clustering(
    df.to_dict("records"),
    name_field="product_name",
    description_field="description",
    category_field="category"
)

# Cluster
clusterer = ProductClustering(clustering_method="kmeans", config={"n_clusters": 10})
cluster_labels = clusterer.fit_predict(texts)

# Add cluster labels to dataframe
df["cluster_id"] = cluster_labels

# Compute success rates
success_rates = clusterer.compute_cluster_success_rates(
    cluster_labels,
    df["sold_within_30_days"].astype(bool)
)

# Use success rates for analog-based viability estimation
for cluster_id, stats in success_rates.items():
    cluster_products = df[df["cluster_id"] == cluster_id]
    print(f"Cluster {cluster_id} has {stats['success_rate']:.2%} success rate")
```

## Helper Functions

### `create_product_text(product_name, description=None, category=None)`
Combines product information into a single text string for embedding.

### `prepare_texts_for_clustering(products, name_field="product_name", ...)`
Prepares product texts from a list of product dictionaries.

## Model Persistence

```python
# Save model
clusterer.save("data/models/clustering/model.pkl")

# Load model
clusterer = ProductClustering()
clusterer.load("data/models/clustering/model.pkl")
```

## Configuration

### k-means Configuration
```python
config = {
    "n_clusters": 10,
    "random_state": 42,
}
```

### HDBSCAN Configuration
```python
config = {
    "min_cluster_size": 5,
    "min_samples": 3,
}
```

## Integration with Other Modules

The clustering module can be integrated with:

1. **Viability Model**: Use cluster success rates as a feature
2. **Price Optimizer**: Use cluster-based analog pricing
3. **FastAPI Backend**: Provide cluster information in API responses

Example integration:
```python
# In feature engineering
cluster_success_rate = success_rates.get(cluster_id, {}).get("success_rate", 0.5)
features["cluster_success_rate"] = cluster_success_rate
```

## Performance Considerations

- **Embedding Generation**: Batch processing for efficiency
- **Clustering**: Both k-means and HDBSCAN are efficient
- **Memory**: Embeddings are stored for reuse
- **Scalability**: Handles large product catalogs

## Next Steps

1. Integrate with feature engineering pipeline
2. Use cluster success rates in viability model
3. Add cluster information to API responses
4. Visualize clusters for analysis
5. Tune clustering parameters for your data

