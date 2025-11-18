"""
TRAINING PIPELINE V3 — DropSmart (COURSE ALIGNED, FAST VERSION)
---------------------------------------------------------------

Trains all production ML models:

1. ViabilityModel         → data/models/viability/model.pkl
2. ConversionModel        → data/models/price_optimizer/conversion_model.pkl
3. StockoutRiskModel      → data/models/stockout_risk/model.pkl
4. Product Clustering     → data/models/clustering/kmeans.pkl + embeddings.npy

Dataset:
C:/Users/Dell/Downloads/dropsmart_supplier_enhanced.xlsx
"""

import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

# ---------------------------
# IMPORT ML MODELS
# ---------------------------
from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel

# (ClusteringModel not needed — faster custom MiniBatchKMeans is used)


# ---------------------------
# CONFIG
# ---------------------------
INPUT_FILE = r"C:/Users/Dell/Downloads/dropsmart_supplier_enhanced.xlsx"
MODEL_ROOT = "data/models"

os.makedirs(f"{MODEL_ROOT}/viability", exist_ok=True)
os.makedirs(f"{MODEL_ROOT}/price_optimizer", exist_ok=True)
os.makedirs(f"{MODEL_ROOT}/stockout_risk", exist_ok=True)
os.makedirs(f"{MODEL_ROOT}/clustering", exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_excel(INPUT_FILE)
print("Loaded:", df.shape)


# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
df["landed_cost"] = df["cost"] + df["shipping_cost"] + df["duties"]
df["margin"] = (df["price"] - df["landed_cost"]) / df["price"]
df["margin"] = df["margin"].clip(lower=0)

# LABELS
df["sale_30d"] = ((df["margin"] > 0.10) & (df["inventory"] > 10)).astype(int)
df["conversion_flag"] = (df["margin"] > 0.08).astype(int)
df["stockout_flag"] = ((df["stock"] < 25) | (df["lead_time_days"] > 14)).astype(int)


# ---------------------------
# FEATURE MATRIX
# ---------------------------
FEATURES = [
    "price", "cost", "shipping_cost", "duties",
    "lead_time_days", "stock", "inventory", "quantity",
    "demand", "past_sales",
    "weight_kg", "length_cm", "width_cm", "height_cm",
    "margin", "supplier_reliability_score"
]

X = df[FEATURES].fillna(0)

y_viability = df["sale_30d"]
y_conversion = df["conversion_flag"]
y_stockout = df["stockout_flag"]


# ---------------------------
# TRAIN/TEST SPLIT FOR VIABILITY
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_viability, test_size=0.25, random_state=42
)


# ================================================================
# 1️⃣ TRAIN VIABILITY MODEL
# ================================================================
print("🔵 Training ViabilityModel (RandomForest)...")
viability_model = ViabilityModel()  # RandomForest (course-aligned)
viability_model.train(X_train, y_train)

viability_model.save(f"{MODEL_ROOT}/viability/model.pkl")
print("✅ Saved viability model")


# ================================================================
# 2️⃣ TRAIN CONVERSION (PRICING) MODEL
# ================================================================
print("🟢 Training ConversionModel (LogisticRegression)...")
conv_model = ConversionModel()  # Logistic Regression (course-aligned)
conv_model.train(X, y_conversion)

conv_model.save(f"{MODEL_ROOT}/price_optimizer/conversion_model.pkl")
print("✅ Saved conversion model")


# ================================================================
# 3️⃣ TRAIN STOCKOUT RISK MODEL
# ================================================================
print("🟠 Training StockoutRiskModel (RandomForest)...")
stockout_model = StockoutRiskModel()
stockout_model.train(X, y_stockout)

stockout_model.save(f"{MODEL_ROOT}/stockout_risk/model.pkl")
print("✅ Saved stockout risk model")


# ================================================================
# 4️⃣ PRODUCT CLUSTERING (FAST — <1 second)
# ================================================================
print("🟣 Training Product Clustering (MiniBatchKMeans)...")

X_cluster = X.copy()

kmeans = MiniBatchKMeans(
    n_clusters=6,           # fewer clusters → faster
    batch_size=64,          # bigger batch = speed
    max_iter=30,            # lower iterations = faster
    n_init=2,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_cluster)

# Save
np.save(f"{MODEL_ROOT}/clustering/embeddings.npy", X_cluster.values)
dump(kmeans, f"{MODEL_ROOT}/clustering/kmeans.pkl", protocol=4)

print("✅ Clustering done (FAST — under 1 second)")


# ================================================================
# DONE
# ================================================================
print("\n🎉 ALL MODELS TRAINED & SAVED SUCCESSFULLY!")
print("📁 Output directory:", MODEL_ROOT)
