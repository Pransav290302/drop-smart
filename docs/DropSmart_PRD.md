# DropSmart — Product Requirements Document (PRD)

**Version:** 1.0  
**Date:** 2025  
**Authors:** Pransav M. Patel, Bilal Qader  

---

## 1. Product Overview

### 1.1 Product Name

**DropSmart — Product & Price Intelligence for Dropshipping Sellers**

### 1.2 Summary

DropSmart is an end-to-end decision intelligence platform that enables dropshipping sellers to:

- Identify high-viability products  
- Price them optimally  
- Predict stockout risks  

The system:

- Ingests supplier Excel sheets  
- Validates the schema  
- Engineers features  
- Performs ML-based:
  - Viability scoring  
  - Price optimization  
  - Stockout/lead-time risk prediction  

**Outputs:**

- Ranked list of products  
- Recommended prices  
- Stockout-risk alerts  
- Exportable CSV  
- Fully operational UI + API (Dockerized)  

---

## 2. Problem Statement

Dropshipping sellers face three critical challenges:

1. **Product selection**  
   - Too many SKUs  
   - Uncertain product viability  

2. **Price setting**  
   - Need to balance:  
     - Profitability  
     - MAP (Minimum Advertised Price) constraints  
     - Price elasticity  

3. **Inventory & stockout risk**  
   - Uncertain lead times  
   - Unreliable or low availability  

Without automated intelligence, decisions become:

- Slow  
- Inconsistent  
- Error-prone  

**DropSmart’s solution:**

- AI-assisted product selection  
- Price optimization  
- Inventory risk detection  

---

## 3. Product Goals & Success Metrics

### 3.1 Goals

- Automate product viability scoring  
- Recommend profit-maximizing but safe prices  
- Flag SKUs at high stockout or long lead-time risk  
- Provide a simple UI workflow:  
  **Upload Excel → View recommendations → Export CSV**

### 3.2 Success Metrics

| Metric                             | Target                                         |
| ---------------------------------- | ---------------------------------------------- |
| Viability model ROC-AUC           | ≥ 0.80                                         |
| PR-AUC for stockout risk          | ≥ 0.60                                         |
| Price recommendation improvement  | ≥ 10% expected profit vs fixed-margin baseline |
| Processing time per file          | < 30 sec for 10k SKUs                          |
| UI usability score                | > 4.0 / 5                                      |

---

## 4. Users & Use Cases

### 4.1 Primary Users

- Small / medium dropshipping sellers  
- E-commerce operators  
- Marketplace listers (Amazon, Shopify, Etsy, etc.)

### 4.2 Core User Use Cases

1. **Upload Supplier File**  
   - Seller uploads supplier workbook (Excel)  
   - System validates schema automatically  

2. **Get Product Viability Score**  
   - User sees which SKUs are likely to sell within 30 days  

3. **Receive Recommended Price**  
   - User views optimized price respecting:
     - MAP constraints  
     - Minimum margin thresholds  

4. **Check Stockout Risk**  
   - Identify items likely to face shortages due to:
     - Long lead times  
     - Low inventory  

5. **Export Results**  
   - Download results as CSV to upload into:
     - Amazon  
     - Shopify  
     - ERP or other systems  

---

## 5. Scope

### 5.1 In Scope

- Excel ingestion & schema validation  
- Feature engineering:
  - Landed cost  
  - Margin %  
  - Volumetric weight  
  - Seasonality features  
- Unsupervised product clustering  
- Viability classification model  
- Price optimization module  
- Stockout / lead-time risk model  
- FastAPI backend  
- Streamlit UI  
- Dockerization  
- CSV export  
- SHAP & calibration plots  
- Evaluation metrics & ablation studies  
- Final report + demo video  

### 5.2 Out of Scope

- Real-time API integrations with Amazon or Shopify  
- Multi-user authentication  
- Historical training pipelines over long time-series  
- Live demand forecasting from external APIs  

---

## 6. Functional Requirements

### 6.1 Excel Ingestion Requirements

- **FR-1:** System must accept an Excel workbook with required fields.  
- **FR-2:** Automatically validate schema and provide error messages.  
- **FR-3:** Normalize currencies, dimensions, and weights.  

### 6.2 Feature Engineering

- **FR-4:** Compute derived fields:
  - **Landed cost** = cost + shipping + duties  
  - **Margin %**  
  - **Volumetric weight** & size tier  
  - **Lead-time buckets**  
  - **Seasonality indicator**  

### 6.3 Machine Learning Models

#### A. Viability Model

- **FR-5:** Predict probability of sale within 30 days: `P(sale within 30 days)`  
- **FR-6:** Support **LightGBM** + baseline **logistic regression**  
- **FR-7:** Use **SHAP** for model explainability  

#### B. Price Recommendation

- **FR-8:** Predict conversion probability `p(price, features)`  
- **FR-9:** Optimize expected profit:  

  \[
  \arg\max_{\text{price}} \; p(\text{price}) \times (\text{price} - \text{landed\_cost})
  \]

- **FR-10:** Enforce:
  - MAP constraints  
  - Minimum margin threshold  

#### C. Stockout / Lead-Time Risk

- **FR-11:** Predict if SKU is at risk due to:
  - Long lead time, and/or  
  - Low availability  

- **FR-12:** Output:
  - Binary risk label (e.g., “High Risk” / “Low Risk”)  
  - Calibrated probability score  

### 6.4 Clustering Module

- **FR-13:** Generate embeddings using **MiniLM / SentenceTransformers**.  
- **FR-14:** Run **k-means** or **HDBSCAN** for grouping similar items.  
- **FR-15:** Use clusters to compute analog-based success rates for SKUs.  

### 6.5 UI Requirements

- **FR-16:** Provide file upload screen.  
- **FR-17:** Display ranked results in a table.  
- **FR-18:** Provide per-product detail page with SHAP visualization.  
- **FR-19:** Support one-click CSV export.  

### 6.6 API Requirements

- **FR-20:** Provide FastAPI endpoints:

  - `POST /upload`  
  - `POST /validate`  
  - `POST /predict_viability`  
  - `POST /optimize_price`  
  - `POST /stockout_risk`  
  - `GET  /get_results`  

### 6.7 Performance Requirements

- Must handle **10k SKUs** in **< 30 seconds**  
- Must be **CPU-compatible** (no GPU dependency)  

---

## 7. Non-Functional Requirements

| Category       | Requirement                                     |
| -------------- | ----------------------------------------------- |
| Security       | No PII storage; no cloud upload unless local   |
| Reliability    | API uptime 99% during demo                     |
| Maintainability| Modular Python code with docstrings            |
| Scalability    | Supports future multi-vendor ingestion         |
| Usability      | Simple UX for non-technical sellers            |
| Portability    | Fully Dockerized                               |

---

## 8. UX & UI Flow

### 8.1 Flow Diagram

**End-to-end workflow:**

1. Upload Excel  
2. Validate Schema  
3. Run Models  
4. Show Dashboard  
5. Export CSV  

### 8.2 Key Screens

1. **Home / Upload Page**
   - File upload control  
   - Schema validation status & errors  

2. **Validation Page**
   - Missing or invalid columns  
   - Normalization/standardization warnings  

3. **Dashboard**
   - Ranked products table  
   - Columns (examples):
     - Product ID / SKU  
     - Name / Title  
     - Viability score  
     - Recommended price  
     - Margin %  
     - Stockout risk flag  

4. **Detail View**
   - SHAP explanations  
   - Feature breakdown per SKU  
   - Cluster membership and analog performance  

5. **CSV Export**
   - One-click export  
   - Ready for import into Amazon / Shopify / ERP  

---

## 9. Technical Architecture

### 9.1 Components

- **Backend:** FastAPI  
- **ML Models:** Python
  - LightGBM  
  - Scikit-learn (Logistic Regression, etc.)  
  - SentenceTransformers (MiniLM)  

- **Frontend:** Streamlit  
- **Containerization:** Docker  
- **Artifact Storage:** Local (for project demo)  

### 9.2 Architecture Diagram (Text-Based)

```text
[User]
   ↓
[Streamlit UI]
   ↓
[FastAPI Backend]
   ↓
[ML Pipeline]
   ├─ Viability Model
   ├─ Price Optimizer
   ├─ Stockout Risk Model
   └─ Clustering Engine

Outputs → Results Table → CSV Export
10. Data Requirements
10.1 Input Dataset

Source: Supplier Excel workbooks

Size: ~100–10,000 SKUs

Must be parsed into a structured DataFrame

10.2 Secondary Data

Online Retail II / Olist dataset for proxy label mapping

Optional: Google Trends or similar external signals

10.3 Derived Features

Landed cost

Margin %

Volumetric weight

Seasonality

Supplier reliability proxy

Title embeddings

11. Algorithmic Requirements (High-Level)
11.1 Viability Model (Pseudocode)
# Input: SKU features X
model = LightGBMClassifier()
prob = model.predict_proba(X)
return prob, SHAP_values

11.2 Price Optimization (Pseudocode)
for price in candidate_range:
    conv_prob = conversion_model(price, features)
    profit = conv_prob * (price - landed_cost)

# Choose price with maximum profit
best_price = argmax(profit)

# Apply MAP and min-margin rules
best_price = apply_constraints(best_price, MAP, min_margin)

return best_price

11.3 Stockout Risk (Pseudocode)
risk_model = LightGBM()
risk = risk_model.predict_proba(features)
return risk

12. Evaluation Plan
12.1 Model Metrics
Model	Metrics
Viability	ROC-AUC, PR-AUC, Top-K recall, Calibration
Price	Policy regret, Expected profit uplift
Stockout	PR-AUC, Brier score
Clustering	Silhouette score, manual spot checks
12.2 Ablation Studies

Evaluate performance under variations:

With / without embeddings

Class-weighting vs SMOTE for imbalance handling

MAP constraints on vs off

With / without rolling features

13. Ethical Considerations

No personal or sensitive customer data is used.

Models must be calibrated to avoid misleading recommendations.

Pricing recommendations must respect MAP rules.

System should be transparent about uncertainty and limitations.