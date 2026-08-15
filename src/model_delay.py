"""
ML-задача №2 (класифікація): прогноз ризику затримки доставки замовлення.

Це складніша та бізнес-цінніша задача: якщо ми можемо заздалегідь
(на етапі оформлення замовлення) передбачити ризик затримки -
можна попередити клієнта, виділити пріоритетні ресурси виробництву тощо.

Порівнюємо Logistic Regression (baseline) vs Random Forest Classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

DATA_PATH = "/home/claude/furniture-sales-analysis/data/furniture_sales.csv"
OUT_DIR = "/home/claude/furniture-sales-analysis/visuals"

df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])

# Ознаки, відомі ЗАЗДАЛЕГІДЬ (на етапі оформлення замовлення) -
# НЕ включаємо actual_delivery_days, бо це вже наслідок затримки (data leakage)
features = [
    "category", "material", "region", "sales_channel", "customer_type",
    "is_custom_order", "quantity", "unit_price", "discount_pct",
    "planned_production_days", "planned_delivery_days",
]
target = "is_delayed"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols = ["category", "material", "region", "sales_channel", "customer_type"]
num_cols = ["is_custom_order", "quantity", "unit_price", "discount_pct",
            "planned_production_days", "planned_delivery_days"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1,
        class_weight="balanced",
    ),
}

roc_data = {}
print("=" * 60)
print("РЕЗУЛЬТАТИ: прогноз ризику затримки доставки (класифікація)")
print("=" * 60)

best_auc = -1
best_pipe = None
best_name = None

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr, tpr, auc)

    print(f"\n--- {name} (ROC-AUC: {auc:.3f}) ---")
    print(classification_report(y_test, pred, target_names=["Вчасно", "Затримка"]))

    if auc > best_auc:
        best_auc = auc
        best_pipe = pipe
        best_name = name

# ROC-крива
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Випадковий класифікатор")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-крива: прогноз затримки доставки")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_roc_curve_delay.png")
plt.close()

# Матриця плутанини (найкраща модель)
pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Вчасно", "Затримка"], yticklabels=["Вчасно", "Затримка"])
plt.title(f"Матриця плутанини ({best_name})")
plt.ylabel("Факт")
plt.xlabel("Прогноз")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_confusion_matrix_delay.png")
plt.close()

# Feature importance (якщо найкраща модель - Random Forest)
if best_name == "Random Forest":
    rf_model = best_pipe.named_steps["model"]
    feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="flare", hue=importances.index, legend=False)
    plt.title("Топ-12 найважливіших ознак ризику затримки (Random Forest)")
    plt.xlabel("Важливість")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/11_feature_importance_delay.png")
    plt.close()

print(f"\nНайкраща модель: {best_name} (ROC-AUC = {best_auc:.3f})")
print("Графіки збережено в /visuals")
