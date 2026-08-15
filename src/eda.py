"""
EDA (Exploratory Data Analysis) для датасету продажів меблевої компанії.
Генерує набір графіків у /visuals та друкує ключові інсайти в консоль.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "/home/claude/furniture-sales-analysis/data/furniture_sales.csv"
OUT_DIR = "/home/claude/furniture-sales-analysis/visuals"

df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
df["month"] = df["order_date"].dt.to_period("M").astype(str)
df["year"] = df["order_date"].dt.year

print("=" * 60)
print("ЗАГАЛЬНА СТАТИСТИКА")
print("=" * 60)
print(f"Період: {df['order_date'].min().date()} -- {df['order_date'].max().date()}")
print(f"Кількість замовлень: {len(df)}")
print(f"Загальний дохід: {df['revenue'].sum():,.0f} PLN")
print(f"Середній чек: {df['revenue'].mean():,.0f} PLN")
print(f"Частка затриманих доставок: {df['is_delayed'].mean()*100:.1f}%")
print(f"Середня оцінка задоволеності: {df['customer_satisfaction'].mean():.2f}/5")

# 1. Динаміка доходу по місяцях
monthly = df.groupby("month")["revenue"].sum().reset_index()
plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly, x="month", y="revenue", marker="o")
plt.xticks(rotation=45, ha="right")
plt.title("Динаміка місячного доходу (2024-2025)")
plt.ylabel("Дохід, PLN")
plt.xlabel("Місяць")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_monthly_revenue.png")
plt.close()

# 2. Дохід за категоріями
cat_revenue = df.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(9, 5))
sns.barplot(data=cat_revenue, x="revenue", y="category", palette="viridis", hue="category", legend=False)
plt.title("Дохід за категоріями товарів")
plt.xlabel("Дохід, PLN")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_revenue_by_category.png")
plt.close()

# 3. Розподіл замовлень за регіонами
plt.figure(figsize=(8, 5))
region_counts = df["region"].value_counts()
sns.barplot(x=region_counts.values, y=region_counts.index, palette="crest", hue=region_counts.index, legend=False)
plt.title("Кількість замовлень за регіонами")
plt.xlabel("Кількість замовлень")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_orders_by_region.png")
plt.close()

# 4. Затримки доставки: кастомні vs стандартні
delay_by_custom = df.groupby("is_custom_order")["is_delayed"].mean().reset_index()
delay_by_custom["is_custom_order"] = delay_by_custom["is_custom_order"].map({0: "Стандартне", 1: "На замовлення"})
plt.figure(figsize=(6, 5))
sns.barplot(data=delay_by_custom, x="is_custom_order", y="is_delayed", palette="rocket", hue="is_custom_order", legend=False)
plt.title("Частка затриманих доставок: стандартні vs кастомні")
plt.ylabel("Частка затримок")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_delay_by_custom.png")
plt.close()

# 5. Кореляційна матриця числових ознак
num_cols = ["quantity", "unit_price", "discount_pct", "revenue",
            "planned_production_days", "actual_delivery_days",
            "is_delayed", "customer_satisfaction"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Кореляційна матриця числових ознак")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_correlation_heatmap.png")
plt.close()

# 6. Задоволеність клієнтів vs затримка
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="is_delayed", y="customer_satisfaction", palette="Set2", hue="is_delayed", legend=False)
plt.xticks([0, 1], ["Вчасно", "Затримано"])
plt.title("Задоволеність клієнтів: вчасні vs затримані замовлення")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_satisfaction_vs_delay.png")
plt.close()

print("\nГрафіки збережено в /visuals (6 файлів)")
