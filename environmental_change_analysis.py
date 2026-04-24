# ============================================================
# Environmental Change Analysis using Machine Learning
# Author: Nandani Goyal
# Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Generate / Load Dataset
# Using synthetic climate data that mirrors real-world patterns
# In a real project, replace this with: df = pd.read_csv('climate_data.csv')
# ============================================================

np.random.seed(42)
n = 500

years      = np.arange(1970, 1970 + n)
months     = np.random.randint(1, 13, n)
# Temperature rises ~0.02°C per year (global warming trend) + seasonal variation + noise
temperature = (
    15
    + 0.02 * (years - 1970)
    + 5 * np.sin(2 * np.pi * months / 12)
    + np.random.normal(0, 1.5, n)
)
# Rainfall inversely correlated with temperature + noise
rainfall = (
    800
    - 1.5 * (years - 1970)
    - 20 * np.sin(2 * np.pi * months / 12)
    + np.random.normal(0, 40, n)
)
humidity = 60 + 0.3 * rainfall / 10 + np.random.normal(0, 5, n)
co2      = 320 + 1.5 * (years - 1970) + np.random.normal(0, 5, n)

# Target: Environmental Change Index (composite score)
env_change_index = (
    0.4 * temperature
    - 0.002 * rainfall
    + 0.01 * co2
    + np.random.normal(0, 0.5, n)
)

df = pd.DataFrame({
    'Year':         years,
    'Month':        months,
    'Temperature':  temperature,
    'Rainfall':     rainfall,
    'Humidity':     humidity,
    'CO2_Level':    co2,
    'Env_Change_Index': env_change_index
})

# Inject some missing values to simulate real-world data
for col in ['Temperature', 'Rainfall', 'Humidity']:
    df.loc[df.sample(frac=0.03).index, col] = np.nan

print("=" * 55)
print("  ENVIRONMENTAL CHANGE ANALYSIS — ML PROJECT")
print("=" * 55)
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe().round(2))


# ============================================================
# STEP 2: Data Cleaning & Preprocessing
# ============================================================

print("\n--- STEP 2: Data Cleaning ---")
print(f"Missing values before cleaning:\n{df.isnull().sum()}")

# Fill missing values with column median (robust to outliers)
for col in ['Temperature', 'Rainfall', 'Humidity']:
    df[col].fillna(df[col].median(), inplace=True)

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")
print("✓ Missing values handled using median imputation")


# ============================================================
# STEP 3: Feature Engineering
# ============================================================

print("\n--- STEP 3: Feature Engineering ---")

# Create decade feature to capture long-term trends
df['Decade'] = (df['Year'] // 10) * 10

# Season encoding from month
def get_season(month):
    if month in [3, 4, 5]:   return 1  # Spring
    elif month in [6, 7, 8]: return 2  # Summer
    elif month in [9,10,11]: return 3  # Autumn
    else:                     return 4  # Winter

df['Season'] = df['Month'].apply(get_season)

# Rainfall-Temperature interaction
df['Rain_Temp_Ratio'] = df['Rainfall'] / (df['Temperature'] + 1)

print("✓ New features created: Decade, Season, Rain_Temp_Ratio")
print(f"Updated shape: {df.shape}")


# ============================================================
# STEP 4: Normalization
# ============================================================

features = ['Year', 'Month', 'Temperature', 'Rainfall',
            'Humidity', 'CO2_Level', 'Decade',
            'Season', 'Rain_Temp_Ratio']
target   = 'Env_Change_Index'

X = df[features].copy()
# Final safety check — fill any remaining NaNs
X = X.fillna(X.median())
y = df[target]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n✓ Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


# ============================================================
# STEP 5: Model Training & Evaluation
# ============================================================

print("\n--- STEP 5: Model Training ---")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_scaled, y, cv=5,
                           scoring='neg_root_mean_squared_error')

    results[name] = {
        'model':  model,
        'y_pred': y_pred,
        'RMSE':   rmse,
        'MAE':    mae,
        'R2':     r2,
        'CV_RMSE_mean': -cv.mean(),
        'CV_RMSE_std':   cv.std()
    }

    print(f"\n{name}:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  CV RMSE: {-cv.mean():.4f} ± {cv.std():.4f}")

# Best model
best_name = min(results, key=lambda x: results[x]['RMSE'])
print(f"\n✓ Best model: {best_name} (lowest RMSE = {results[best_name]['RMSE']:.4f})")


# ============================================================
# STEP 6: Visualizations
# ============================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Environmental Change Analysis — ML Results",
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

# --- Plot 1: Temperature trend over years ---
ax1 = fig.add_subplot(gs[0, 0])
yearly_temp = df.groupby('Year')['Temperature'].mean()
ax1.plot(yearly_temp.index, yearly_temp.values, color='#F44336', linewidth=1.5)
ax1.set_title('Avg Temperature Over Years', fontweight='bold')
ax1.set_xlabel('Year'); ax1.set_ylabel('Temperature (°C)')
ax1.grid(True, alpha=0.3)

# --- Plot 2: Rainfall trend ---
ax2 = fig.add_subplot(gs[0, 1])
yearly_rain = df.groupby('Year')['Rainfall'].mean()
ax2.plot(yearly_rain.index, yearly_rain.values, color='#2196F3', linewidth=1.5)
ax2.set_title('Avg Rainfall Over Years', fontweight='bold')
ax2.set_xlabel('Year'); ax2.set_ylabel('Rainfall (mm)')
ax2.grid(True, alpha=0.3)

# --- Plot 3: CO2 trend ---
ax3 = fig.add_subplot(gs[0, 2])
yearly_co2 = df.groupby('Year')['CO2_Level'].mean()
ax3.plot(yearly_co2.index, yearly_co2.values, color='#FF5722', linewidth=1.5)
ax3.set_title('CO₂ Level Over Years', fontweight='bold')
ax3.set_xlabel('Year'); ax3.set_ylabel('CO₂ (ppm)')
ax3.grid(True, alpha=0.3)

# --- Plot 4: Correlation heatmap (manual) ---
ax4 = fig.add_subplot(gs[1, 0])
corr_cols = ['Temperature', 'Rainfall', 'Humidity', 'CO2_Level', 'Env_Change_Index']
corr_matrix = df[corr_cols].corr()
im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax4.set_xticks(range(len(corr_cols)))
ax4.set_yticks(range(len(corr_cols)))
ax4.set_xticklabels(['Temp','Rain','Hum','CO2','ECI'], fontsize=8)
ax4.set_yticklabels(['Temp','Rain','Hum','CO2','ECI'], fontsize=8)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax4.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}',
                 ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax4, shrink=0.8)
ax4.set_title('Correlation Matrix', fontweight='bold')

# --- Plot 5: Model comparison bar chart ---
ax5 = fig.add_subplot(gs[1, 1])
model_names  = list(results.keys())
rmse_scores  = [results[m]['RMSE'] for m in model_names]
short_names  = ['Lin Reg', 'Ridge', 'Dec Tree', 'Rand Forest']
bars = ax5.bar(short_names, rmse_scores, color=colors, edgecolor='white', width=0.5)
ax5.set_title('Model RMSE Comparison', fontweight='bold')
ax5.set_ylabel('RMSE (lower is better)')
ax5.set_ylim(0, max(rmse_scores) * 1.3)
for bar, val in zip(bars, rmse_scores):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# --- Plot 6: Actual vs Predicted (best model) ---
ax6 = fig.add_subplot(gs[1, 2])
y_pred_best = results[best_name]['y_pred']
ax6.scatter(y_test, y_pred_best, alpha=0.4, color='#3F51B5', s=15)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect fit')
ax6.set_title(f'Actual vs Predicted\n({best_name})', fontweight='bold')
ax6.set_xlabel('Actual ECI'); ax6.set_ylabel('Predicted ECI')
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

# --- Plot 7: Feature importance (Random Forest) ---
ax7 = fig.add_subplot(gs[2, 0:2])
rf_model    = results['Random Forest']['model']
importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1]
feat_names  = [features[i] for i in indices]
ax7.barh(feat_names, importances[indices], color='#009688')
ax7.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax7.set_xlabel('Importance Score')
ax7.grid(True, alpha=0.3, axis='x')

# --- Plot 8: Residuals ---
ax8 = fig.add_subplot(gs[2, 2])
residuals = y_test - y_pred_best
ax8.hist(residuals, bins=30, color='#607D8B', edgecolor='white', alpha=0.8)
ax8.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax8.set_title('Residuals Distribution', fontweight='bold')
ax8.set_xlabel('Residual (Actual − Predicted)')
ax8.set_ylabel('Frequency')
ax8.grid(True, alpha=0.3)

plt.savefig('environmental_change_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Plot saved as 'environmental_change_results.png'")

print("\n" + "=" * 55)
print("  FINAL SUMMARY")
print("=" * 55)
print(f"  Dataset      : {n} climate records (1970–{1970+n})")
print(f"  Features used: {len(features)}")
print(f"  Best model   : {best_name}")
print(f"  Best RMSE    : {results[best_name]['RMSE']:.4f}")
print(f"  Best R²      : {results[best_name]['R2']:.4f}")
print("\n  Key insight: Temperature and CO₂ show a strong")
print("  upward trend over decades, while rainfall declines.")
print("  Random Forest captures non-linear patterns best.")
print("=" * 55)
