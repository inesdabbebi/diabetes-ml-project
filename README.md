# 🩺 Diabetes Risk Prediction Using Clinical and Lifestyle Data

**Course:** Python for Data Science 2: Guided Machine Learning Project  
**Tutor:** Haythem Ghazouani

---

## Project Overview

This project focuses on predicting **diabetes risk** using clinical measurements, demographics, and lifestyle factors. The goal is to build a **complete machine learning pipeline** including data preprocessing, modeling, evaluation, API deployment, and a front-end interface.

Key objectives:
- Understand health and lifestyle factors contributing to diabetes.
- Predict whether a patient is diabetic or not using machine learning.
- Identify clusters of patients with similar health risk profiles.
- Deploy an interactive predictive system with FastAPI and React.

---

## Dataset

**Title:** Diabetes Risk Prediction Dataset  
**Source:** Public health dataset "https://data.mendeley.com/datasets/xv25yjbzkm/2" 
**Format:** CSV   
**Rows:** 20,000 patients

| Feature | Description |
|---------|-------------|
| `Gender` | Gender of the patient (Male/Female) |
| `Age` | Age of the patient (years) |
| `Physical Activity` | Activity level (Low / Moderate / High) |
| `Smoking Status` | Smoking habit (Never / Former / Current) |
| `Alcohol Intake` | Alcohol consumption (None / Occasional / Regular) |
| `Glucose` | Blood glucose level (mg/dL) |
| `Blood Pressure` | Diastolic blood pressure (mmHg) |
| `Skin Thickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (µU/mL) |
| `BMI` | Body Mass Index (kg/m²) |
| `Cholesterol` | Total cholesterol level |
| `Diabetes Pedigree Function` | Genetic risk score |
| `Family History` | Family history of diabetes (Yes/No) |
| `Hypertension` | Patient has hypertension (Yes/No) |
| `Outcome` | **Target variable** — Diabetic / Non-diabetic |

---

## 🗓️ 7-Week Roadmap

| Week | Focus |
|------|-------|
| 1 | **Setup & EDA:** Load dataset, explore distributions, check missing values  |
| 2 | **Preprocessing & Feature Engineering:** Handle missing data, encode categories, normalize numeric features  |
| 3 | **Modeling & MLflow:** Build classification models to predict diabetes risk, track experiments  |
| 4 | **API Development (FastAPI):** Deploy trained model as a REST API |
| 5 | **Frontend Development (React):** Build a UI to interact with predictions |
| 6 | **Containerization (Docker):** Containerize API and frontend |
| 7 | **Deployment & Final Review:** Deploy solution, evaluate performance, document insights |

---

## Methodology

### 1. Data Preprocessing
- Handle missing values (e.g. `Alcohol Intake` — 49% missing).
- Encode categorical variables using `LabelEncoder`.
- Normalize numeric features using `StandardScaler`.
- Split data: **80% train / 20% test** with stratification.
- Balance classes using **SMOTE** (83% Diabetic vs 17% Non-diabetic → balanced).

### 2. Exploratory Data Analysis (EDA)
- Compute summary statistics and correlations.
- Visualize distributions using histograms, boxplots, and heatmaps.
- Detect patterns between BMI, Glucose, Age, and diabetes status.

### 3. Machine Learning

**Classification** — Predict `Outcome` (Diabetic = 1, Non-diabetic = 0):

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline model — fast and interpretable |
| KNN | K-Nearest Neighbors — best k selected via accuracy loop (k=1 to 20) |
| Random Forest | Ensemble of 200 decision trees — best overall performance |



### 4. Evaluation Metrics

**Classification:** Accuracy, F1-Score, ROC-AUC, Confusion Matrix  

### 5. API Development (FastAPI) *(Week 4)*
Expose the best trained model as a REST API for external applications.

### 6. Frontend Development (React) *(Week 5)*
Build an interactive interface to input patient features and get predictions.

### 7. Containerization (Docker) *(Week 6)*
Containerize API and frontend for consistent deployment.

### 8. Deployment & Final Review *(Week 7)*
Deploy solution, test end-to-end functionality, and document insights.

---


