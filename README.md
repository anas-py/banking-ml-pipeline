<div align="center">

<h1>🏦 Banking ML Pipeline</h1>

<p><em>A production-grade, end-to-end machine learning pipeline for predicting customer term deposit subscriptions — built with MLOps best practices.</em></p>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Pipeline Stages](#-pipeline-stages)
- [Results & Metrics](#-results--metrics)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project implements a **complete MLOps lifecycle** for a binary classification task in the banking domain — predicting whether a customer will subscribe to a term deposit product based on demographic, financial, and campaign-related features.

The pipeline is designed to be **reproducible**, **modular**, and **scalable**, covering everything from raw data ingestion to model serving via a REST API. Experiment tracking and model versioning are handled through **MLflow**, enabling teams to audit, compare, and promote models with confidence.

> **Business Problem:** Banks spend significant resources on marketing campaigns. This model enables targeted outreach by identifying customers with the highest likelihood of subscription — reducing cost-per-acquisition and improving campaign ROI.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔄 **Automated Pipeline** | Single-command execution of the full ML workflow |
| 🧹 **Robust Preprocessing** | Handles missing values, outliers, and class imbalance |
| 🧬 **Feature Engineering** | Scikit-learn Pipelines with `ColumnTransformer` for reproducible transforms |
| 📊 **Experiment Tracking** | MLflow integration for logging params, metrics, and artifacts |
| 📦 **Model Registry** | Versioned model storage with staging/production promotion |
| 🚀 **REST API** | FastAPI endpoint for real-time inference |
| ✅ **85% Accuracy** | Validated on held-out real-world banking data |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Banking ML Pipeline                         │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │   Raw    │───▶│  Preprocess  │───▶│  Feature Engineering  │ │
│  │  Data    │    │  & Cleaning  │    │  (ColumnTransformer)  │ │
│  └──────────┘    └──────────────┘    └───────────┬───────────┘ │
│                                                  │             │
│  ┌──────────────────────────────┐    ┌───────────▼───────────┐ │
│  │       MLflow Registry        │◀───│    Model Training     │ │
│  │  (Versioning & Promotion)    │    │   (Random Forest)     │ │
│  └──────────────┬───────────────┘    └───────────────────────┘ │
│                 │                                               │
│  ┌──────────────▼───────────────┐    ┌───────────────────────┐ │
│  │      MLflow Tracking         │    │   FastAPI Inference   │ │
│  │  (Params, Metrics, Artifacts)│    │      Endpoint         │ │
│  └──────────────────────────────┘    └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.9+ | Core development |
| **ML Framework** | Scikit-learn | Preprocessing & modeling |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Experiment Tracking** | MLflow | Logging, registry, artifact storage |
| **API Framework** | FastAPI | REST inference endpoint |
| **Server** | Uvicorn | ASGI server for FastAPI |

---

## 📂 Dataset

The pipeline is designed for the **UCI Bank Marketing Dataset** — a publicly available dataset with over 45,000 records and 16 features capturing customer demographics, financial history, and direct marketing campaign interactions.

**Target Variable:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)

**Key Feature Categories:**
- **Client attributes** — age, job, marital status, education, credit default, housing loan, personal loan
- **Campaign attributes** — contact type, month, day of week, call duration, number of contacts
- **Social/Economic indicators** — employment variation rate, consumer price index, Euribor 3-month rate

> **Class Imbalance:** The dataset is imbalanced (~88% "no"). The pipeline addresses this through appropriate evaluation metrics and optional resampling strategies.

---

## 🔬 Pipeline Stages

### 1. Data Ingestion & Validation
- Load raw CSV data into Pandas DataFrames
- Schema validation and type enforcement
- Missing value detection and imputation strategy selection

### 2. Preprocessing
- Numerical features: `StandardScaler` for normalization
- Categorical features: `OneHotEncoder` with `handle_unknown='ignore'`
- All transformations encapsulated in a Scikit-learn `Pipeline` to prevent data leakage

### 3. Feature Engineering
- `ColumnTransformer` applies different transformations to numerical and categorical columns simultaneously
- Ensures the same transformation logic is applied consistently at training and inference time

### 4. Model Training
- Algorithm: `RandomForestClassifier`
- Hyperparameter configuration logged to MLflow
- Stratified train/test split to preserve class distribution

### 5. Evaluation
- Metrics computed: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix and classification report generated as artifacts
- All metrics automatically logged to MLflow

### 6. MLflow Tracking & Registry
- Each run is tracked with a unique `run_id`
- Trained model registered under a named experiment in the MLflow Model Registry
- Supports model stage transitions: `Staging` → `Production`

### 7. FastAPI Deployment
- `/predict` endpoint accepts JSON payloads with customer features
- Returns prediction (`0`/`1`) and subscription probability
- Loads the registered MLflow model at startup

---

## 📈 Results & Metrics

| Metric | Score |
|---|---|
| **Accuracy** | 85% |
| **Dataset** | UCI Bank Marketing |
| **Algorithm** | Random Forest Classifier |
| **Validation** | Stratified Train/Test Split |

> Full run metrics, confusion matrices, and feature importance plots are available in the MLflow UI after executing the pipeline.

---

## 📁 Project Structure

```
banking-ml-pipeline/
│
├── run_pipeline.py         # Main entry point — executes the full ML pipeline
├── requirements.txt        # Python dependencies
├── mlflow.db               # Local MLflow tracking database (SQLite)
│
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- (Optional) Virtual environment tool (`venv` or `conda`)

### 1. Clone the Repository

```bash
git clone https://github.com/anas-py/banking-ml-pipeline.git
cd banking-ml-pipeline
```

### 2. Create & Activate a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
python run_pipeline.py
```

This single command will execute all pipeline stages: data ingestion, preprocessing, training, evaluation, and MLflow logging.

---

## 📊 MLflow Experiment Tracking

After running the pipeline, launch the MLflow UI to explore all tracked experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open your browser at **[http://localhost:5000](http://localhost:5000)**

**What you'll see in the UI:**
- All experiment runs with timestamps
- Logged hyperparameters (e.g., `n_estimators`, `max_depth`)
- Evaluation metrics per run (accuracy, F1, ROC-AUC)
- Stored model artifacts
- Model Registry with version history

---

## 🌐 API Reference

Once deployed, the FastAPI inference server can be started with:

```bash
uvicorn app:app --reload
```

### `POST /predict`

Predict whether a customer will subscribe to a term deposit.

**Request Body:**
```json
{
  "age": 35,
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "duration": 200,
  "campaign": 2,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp_var_rate": -1.8,
  "cons_price_idx": 92.893,
  "cons_conf_idx": -46.2,
  "euribor3m": 1.313,
  "nr_employed": 5099.1
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.76,
  "label": "yes — likely to subscribe"
}
```

Interactive API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)** (Swagger UI).

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the pipeline, add new models, or extend the API, please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add: descriptive commit message'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [anas-py](https://github.com/anas-py)

⭐ If you found this project helpful, consider giving it a star!

</div>
