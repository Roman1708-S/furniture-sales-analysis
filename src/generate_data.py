"""
Генерація синтетичного датасету продажів меблевої компанії.

Компанія: виробник меблів на замовлення (custom furniture manufacturer).
Період: 2 роки (2024-01-01 -- 2025-12-31).
Логіка закладена навмисно, щоб EDA та ML знаходили реальні патерни:
  - сезонність (пік навесні/восени - ремонти, спад влітку)
  - преміум-матеріали -> вища ціна і довша доставка
  - кастомні замовлення -> довша доставка, вища ціна, вищий ризик затримки
  - гуртові (wholesale) клієнти -> знижки, більші обсяги
  - за 2 роки - невеликий загальний тренд росту
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_ORDERS = 4200

CATEGORIES = {
    # category: (base_price, price_std)
    "Диван": (1400, 350),
    "Стіл": (650, 200),
    "Шафа": (1100, 300),
    "Ліжко": (900, 250),
    "Стілець": (180, 60),
    "Комод": (500, 150),
    "Кухня на замовлення": (3200, 800),
}

MATERIALS = {
    # material: (price_multiplier, prod_days_extra)
    "ДСП/МДФ": (0.75, 0),
    "Сосна": (0.95, 2),
    "Дуб": (1.35, 5),
    "Ясен": (1.25, 4),
    "Шпон": (1.10, 3),
    "Метал + скло": (1.15, 2),
}

REGIONS = ["Вроцлав", "Варшава", "Краків", "Познань", "Гданськ", "Лодзь", "Катовіце"]
REGION_WEIGHTS = [0.28, 0.22, 0.14, 0.12, 0.10, 0.08, 0.06]

CHANNELS = ["Шоурум", "Онлайн", "Дилер"]
CHANNEL_WEIGHTS = [0.35, 0.40, 0.25]

CUSTOMER_TYPES = ["Роздрібний", "Дизайнер", "Гуртовий"]
CUSTOMER_WEIGHTS = [0.65, 0.20, 0.15]


def random_dates(n):
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-12-31")
    days_range = (end - start).days
    # Сезонна вага: пік Бер-Тра та Вер-Жов, спад Лип-Сер
    dates = []
    months_weights = {
        1: 0.8, 2: 0.9, 3: 1.3, 4: 1.35, 5: 1.15, 6: 0.95,
        7: 0.6, 8: 0.65, 9: 1.25, 10: 1.3, 11: 1.05, 12: 0.9,
    }
    all_days = pd.date_range(start, end, freq="D")
    weights = np.array([months_weights[d.month] for d in all_days], dtype=float)
    # невеликий тренд росту рік-до-року
    year_boost = np.array([1.0 if d.year == 2024 else 1.12 for d in all_days])
    weights = weights * year_boost
    weights = weights / weights.sum()
    chosen = RNG.choice(all_days, size=n, p=weights)
    return pd.to_datetime(chosen)


def generate():
    n = N_ORDERS
    order_id = np.arange(100001, 100001 + n)
    order_date = random_dates(n)

    category = RNG.choice(list(CATEGORIES.keys()), size=n)
    material = RNG.choice(list(MATERIALS.keys()), size=n)
    region = RNG.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    channel = RNG.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)
    customer_type = RNG.choice(CUSTOMER_TYPES, size=n, p=CUSTOMER_WEIGHTS)

    is_custom = RNG.choice([0, 1], size=n, p=[0.55, 0.45])

    quantity = np.where(
        customer_type == "Гуртовий",
        RNG.integers(3, 15, size=n),
        RNG.integers(1, 3, size=n),
    )

    base_price = np.array([CATEGORIES[c][0] for c in category], dtype=float)
    price_std = np.array([CATEGORIES[c][1] for c in category], dtype=float)
    mat_mult = np.array([MATERIALS[m][0] for m in material], dtype=float)
    mat_extra_days = np.array([MATERIALS[m][1] for m in material], dtype=float)

    unit_price = RNG.normal(base_price, price_std) * mat_mult
    unit_price = np.clip(unit_price, base_price * 0.5, None)
    unit_price = unit_price * np.where(is_custom == 1, RNG.uniform(1.1, 1.35, size=n), 1.0)
    unit_price = np.round(unit_price, 2)

    discount_pct = np.where(
        customer_type == "Гуртовий",
        RNG.uniform(0.08, 0.22, size=n),
        np.where(customer_type == "Дизайнер", RNG.uniform(0.03, 0.10, size=n), RNG.uniform(0.0, 0.05, size=n)),
    )
    discount_pct = np.round(discount_pct, 3)

    revenue = np.round(unit_price * quantity * (1 - discount_pct), 2)

    production_days = (
        7
        + mat_extra_days
        + np.where(is_custom == 1, RNG.integers(5, 20, size=n), RNG.integers(0, 4, size=n))
        + RNG.integers(-1, 3, size=n)
    )
    production_days = np.clip(production_days, 2, None).astype(int)

    delivery_days = production_days + RNG.integers(1, 8, size=n)

    # Ризик затримки: вища для кастомних, преміум-матеріалів, гуртових великих партій, певних регіонів
    delay_risk_score = (
        0.05
        + 0.32 * is_custom
        + 0.22 * (mat_mult - 0.75) / (1.35 - 0.75)
        + 0.20 * (quantity > 5).astype(float)
        + 0.10 * (channel == "Дилер").astype(float)
        + RNG.normal(0, 0.03, size=n)
    )
    delay_risk_score = np.clip(delay_risk_score, 0.02, 0.92)
    is_delayed = (RNG.uniform(0, 1, size=n) < delay_risk_score).astype(int)
    actual_delivery_days = delivery_days + np.where(
        is_delayed == 1, RNG.integers(3, 15, size=n), 0
    )

    satisfaction = np.clip(
        RNG.normal(4.3 - 0.9 * is_delayed - 0.05 * (mat_mult < 0.9), 0.5, size=n),
        1, 5,
    )
    satisfaction = np.round(satisfaction, 1)

    df = pd.DataFrame({
        "order_id": order_id,
        "order_date": order_date,
        "category": category,
        "material": material,
        "region": region,
        "sales_channel": channel,
        "customer_type": customer_type,
        "is_custom_order": is_custom,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_pct": discount_pct,
        "revenue": revenue,
        "planned_production_days": production_days,
        "planned_delivery_days": delivery_days,
        "actual_delivery_days": actual_delivery_days,
        "is_delayed": is_delayed,
        "customer_satisfaction": satisfaction,
    })

    df = df.sort_values("order_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/furniture-sales-analysis/data/furniture_sales.csv"
    df.to_csv(out_path, index=False)
    print(f"Згенеровано {len(df)} замовлень -> {out_path}")
    print(df.head())
    print("\nПеревірка типів та пропусків:")
    print(df.info())
