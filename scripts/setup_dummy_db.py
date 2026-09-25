"""
Builds the dummy SQLITE database. Run anually once whenever you need a fresh version

run python scripts/setup_dummy_db.py
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

from config import get_settings

SCHEMA = """
CREATE TABLE regions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region_id INTEGER NOT NULL REFERENCES regions(id),
    signup_date TEXT NOT NULL,
    churned_at TEXT
);

CREATE TABLE subscriptions (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    plan TEXT NOT NULL,
    mrr REAL NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT
);

CREATE TABLE invoices (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    amount REAL NOT NULL,
    invoice_date TEXT NOT NULL
);
"""

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
PLANS = [("Starter", 49.0), ("Growth", 199.0), ("Enterprise", 799.0)]
FIRST_NAMES = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark",
               "Wayne", "Wonka", "Soylent", "Vandelay", "Pied Piper", "Massive Dynamic"]
SUFFIXES = ["Corp", "Inc", "LLC", "Ltd", "Group", "Holdings"]


def _random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta_days, 0)))


def build_and_seed(db_path, seed: int = 42, num_customers: int = 80) -> None:
    random.seed(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            DROP TABLE IF EXISTS invoices;
            DROP TABLE IF EXISTS subscriptions;
            DROP TABLE IF EXISTS customers;
            DROP TABLE IF EXISTS regions;
        """)
        conn.executescript(SCHEMA)

        conn.executemany(
            "INSERT INTO regions (id, name) VALUES (?, ?)",
            list(enumerate(REGIONS, start=1)),
        )

        window_start = date(2023, 1, 1)
        window_end = date(2024, 12, 31)

        for customer_id in range(1, num_customers + 1):
            name = f"{random.choice(FIRST_NAMES)} {random.choice(SUFFIXES)} {customer_id}"
            region_id = random.randint(1, len(REGIONS))
            signup = _random_date(window_start, window_end - timedelta(days=30))

            # ~20% of customers have churned.
            churned_at = None
            if random.random() < 0.2:
                churned = _random_date(signup + timedelta(days=30), window_end)
                churned_at = churned.isoformat()

            conn.execute(
                "INSERT INTO customers (id, name, region_id, signup_date, churned_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (customer_id, name, region_id, signup.isoformat(), churned_at),
            )

            plan_name, mrr = random.choice(PLANS)
            sub_end = churned_at
            conn.execute(
                "INSERT INTO subscriptions (customer_id, plan, mrr, start_date, end_date) "
                "VALUES (?, ?, ?, ?, ?)",
                (customer_id, plan_name, mrr, signup.isoformat(), sub_end),
            )

            # One invoice per active month between signup and churn/window end.
            invoice_end = date.fromisoformat(churned_at) if churned_at else window_end
            cursor_date = signup
            while cursor_date <= invoice_end:
                jitter = mrr * random.uniform(0.95, 1.05)
                conn.execute(
                    "INSERT INTO invoices (customer_id, amount, invoice_date) "
                    "VALUES (?, ?, ?)",
                    (customer_id, round(jitter, 2), cursor_date.isoformat()),
                )
                # Advance roughly one month.
                next_month = cursor_date.month % 12 + 1
                next_year = cursor_date.year + (1 if cursor_date.month == 12 else 0)
                cursor_date = cursor_date.replace(year=next_year, month=next_month, day=1)

        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    settings = get_settings()
    build_and_seed(settings.sqlite_path)
    print(f"Dummy database created at {settings.sqlite_path}")