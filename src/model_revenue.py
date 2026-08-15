"""
ML-задача №1 (регресія): прогноз доходу (revenue) із замовлення
на основі його характеристик (категорія, матеріал, регіон, кількість тощо).

Порівнюємо Linear Regression (baseline) vs Random Forest Regressor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "/home/claude/furniture-sales-analysis/data/furniture_sales.csv"
OUT_DIR = "/home/claude/furniture-sales-analysis/visuals"

df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])

# Ознаки, доступні ДО того, як ми знаємо суму замовлення
# (unit_price та revenue напряму пов'язані - тому unit_price лишаємо,
#  бо на практиці ціну за одиницю ми знаємо з прайс-листа наперед;
#  але discount_pct теж відомий заздалегідь, як умова угоди)
features = [
    "category", "material", "region", "sales_channel", "customer_type",
    "is_custom_order", "quantity", "unit_price", "discount_pct",
]
target = "revenue"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = ["category", "material", "region", "sales_channel", "customer_type"]
num_cols = ["is_custom_order", "quantity", "unit_price", "discount_pct"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
}

results = []
predictions = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    if name == "Random Forest":
        best_pipe = pipe

results_df = pd.DataFrame(results)
print("=" * 60)
print("РЕЗУЛЬТАТИ: прогноз доходу із замовлення (регресія)")
print("=" * 60)
print(results_df.to_string(index=False))

# Графік: факт vs прогноз (краща модель)
best_name = results_df.sort_values("R2", ascending=False).iloc[0]["model"]
plt.figure(figsize=(7, 7))
plt.scatter(y_test, predictions[best_name], alpha=0.35, s=15)
lims = [0, max(y_test.max(), predictions[best_name].max())]
plt.plot(lims, lims, "r--", label="Ідеальний прогноз")
plt.xlabel("Фактичний дохід, PLN")
plt.ylabel("Прогнозований дохід, PLN")
plt.title(f"Факт vs Прогноз ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_revenue_actual_vs_predicted.png")
plt.close()

# Feature importance (Random Forest)
rf_model = best_pipe.named_steps["model"]
feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)

plt.figure(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, palette="mako", hue=importances.index, legend=False)
plt.title("Топ-12 найважливіших ознак (Random Forest)")
plt.xlabel("Важливість")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_feature_importance_revenue.png")
plt.close()

print(f"\nНайкраща модель: {best_name}")
print("Графіки збережено в /visuals")
