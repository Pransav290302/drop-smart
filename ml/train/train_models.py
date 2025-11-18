"""
TRAINING PIPELINE V4 — DropSmart
-----------------------------------------
Uses:
1. ViabilityModel (RandomForest)
2. ConversionModel (LogisticRegression)
3. StockoutRiskModel (RandomForest)
4. ClusteringModel (TF-IDF + KMeans)  ← OPTION A (recommended)

Dataset:
C:/Users/Dell/Downloads/dropsmart_supplier_enhanced.xlsx
"""

import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split

# ---------------------------
# IMPORT ML MODELS
# ---------------------------
from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel
from ml.models.clustering_model import ClusteringModel  # ← NEW


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

# Labels
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


# ================================================================
# 1️⃣ TRAIN VIABILITY MODEL
# ================================================================
print("🔵 Training ViabilityModel...")
viability_model = ViabilityModel()
viability_model.train(X, y_viability)

viability_model.save(f"{MODEL_ROOT}/viability/model.pkl")
print("✅ Saved viability model")


# ================================================================
# 2️⃣ TRAIN CONVERSION MODEL
# ================================================================
print("🟢 Training ConversionModel...")
conv_model = ConversionModel()
conv_model.train(X, y_conversion)

conv_model.save(f"{MODEL_ROOT}/price_optimizer/conversion_model.pkl")
print("✅ Saved conversion model")


# ================================================================
# 3️⃣ TRAIN STOCKOUT RISK MODEL
# ================================================================
print("🟠 Training StockoutRiskModel...")
stockout_model = StockoutRiskModel()
stockout_model.train(X, y_stockout)

stockout_model.save(f"{MODEL_ROOT}/stockout_risk/model.pkl")
print("✅ Saved stockout risk model")


# ================================================================
# 4️⃣ CLUSTERING MODEL (TF-IDF + KMeans)
# ================================================================
print("🟣 Training ClusteringModel (TF-IDF + KMeans text clustering)...")

# Build product text
def build_text(row):
    parts = [
        str(row.get("product_name", "")),
        str(row.get("description", "")),
        str(row.get("category", "")),
    ]
    return " ".join([p for p in parts if p.strip()])

product_texts = df.apply(build_text, axis=1).tolist()

cluster_model = ClusteringModel(config={"n_clusters": 6})
cluster_model.train(product_texts)

cluster_model.save(f"{MODEL_ROOT}/clustering/model.pkl")
print("✅ Saved clustering model (TF-IDF + KMeans)")


# ================================================================
# DONE
# ================================================================
print("\n🎉 ALL MODELS TRAINED & SAVED SUCCESSFULLY!")
print("📁 Output directory:", MODEL_ROOT)
