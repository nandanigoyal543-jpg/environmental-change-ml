# 🌍 Environmental Change Analysis using Machine Learning

> Predicting environmental change patterns from climate data using supervised ML models

---

## 📌 Problem Statement
Analyze historical climate data to identify patterns and predict an **Environmental Change Index (ECI)** based on temperature, rainfall, CO₂ levels, and humidity trends.

---

## 📊 Dataset
- **Source:** Public climate datasets (temperature and rainfall trends)
- **Size:** 500 climate records (1970–2020)
- **Features:** Year, Month, Temperature (°C), Rainfall (mm), Humidity (%), CO₂ Level (ppm)

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas & NumPy | Data cleaning, feature engineering |
| Scikit-learn | ML models, evaluation |
| Matplotlib | Visualization |

---

## 🔄 Project Pipeline

```
Raw Climate Data
      ↓
Data Cleaning (median imputation for missing values)
      ↓
Feature Engineering (Decade, Season, Rain-Temp Ratio)
      ↓
Normalization (StandardScaler)
      ↓
Model Training & Hyperparameter Tuning
      ↓
Evaluation (RMSE, MAE, R², Cross-validation)
      ↓
Visualization & Insights
```

---

## 🤖 Models Used
- **Linear Regression** — baseline model
- **Ridge Regression** — handles multicollinearity
- **Decision Tree Regressor** (max_depth=5)
- **Random Forest Regressor** (100 estimators) ← best performer

---

## 📈 Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | ~0.55 | ~0.44 | ~0.92 |
| Ridge Regression | ~0.55 | ~0.44 | ~0.92 |
| Decision Tree | ~0.38 | ~0.28 | ~0.96 |
| **Random Forest** | **~0.22** | **~0.16** | **~0.99** |

> ✅ Random Forest achieved the best performance with lowest RMSE

---

## 🔍 Key Insights
- Temperature shows a clear **upward trend** (~0.02°C per year) — consistent with global warming data
- CO₂ levels rise steadily, strongly correlated with the Environmental Change Index
- Rainfall shows a **declining trend** over decades
- Random Forest captures non-linear interactions better than linear models

---

## 📊 Visualizations Generated
1. Temperature trend over years
2. Rainfall trend over years
3. CO₂ level trend
4. Correlation matrix (heatmap)
5. Model RMSE comparison
6. Actual vs Predicted plot
7. Feature importance chart
8. Residuals distribution

---

## ▶️ How to Run

**Option 1 — Google Colab (recommended):**
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `environmental_change_analysis.py`
3. Run all cells

**Option 2 — Local:**
```bash
pip install numpy pandas scikit-learn matplotlib
python environmental_change_analysis.py
```

---

## 👩‍💻 Author
**Nandani Goyal**
B.Tech Biotechnology (Minor: AI/ML) | JIIT Delhi
- LinkedIn: [linkedin.com/in/nandani-goyal543](https://linkedin.com/in/nandani-goyal543)
- Email: nandanigoyal543@gmail.com
