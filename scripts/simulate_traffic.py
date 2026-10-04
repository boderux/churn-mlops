"""Send synthetic customers to the API (to populate dashboards and the Evidently window).

    python scripts/simulate_traffic.py --n 300                # healthy traffic
    python scripts/simulate_traffic.py --n 300 --drift        # shifted population -> drift alert
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.data import generate_synthetic  # noqa: E402

FIELDS = ["SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
          "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
          "MonthlyCharges", "TotalCharges"]


def base_population(n: int, seed: int) -> pd.DataFrame:
    """Healthy traffic = resample of the REAL hold-out set (same distribution as training).
    Falls back to the synthetic generator when the pipeline has not been run yet."""
    test_csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        return df.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    return generate_synthetic(n, seed=seed)


def make_customers(n: int, drift: bool, seed: int) -> list[dict]:
    df = base_population(n, seed)
    if drift:  # a new acquisition campaign: short-tenure, fibre, month-to-month, pricey customers
        rng = np.random.default_rng(seed)
        df["tenure"] = rng.integers(0, 6, n)
        df["Contract"] = "Month-to-month"
        df["InternetService"] = "Fiber optic"
        df["PaymentMethod"] = "Electronic check"
        df["MonthlyCharges"] = (df["MonthlyCharges"] + 35).clip(upper=199).round(2)
        df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
    out = df[FIELDS].to_dict("records")
    for r in out:
        r["SeniorCitizen"] = int(r["SeniorCitizen"])
        r["tenure"] = int(r["tenure"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--rps", type=float, default=50, help="requests per second")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    ok = fail = 0
    for c in make_customers(a.n, a.drift, a.seed):
        try:
            r = requests.post(f"{a.url}/v1/predict", json=c, timeout=5)
            ok += r.status_code == 200
            fail += r.status_code != 200
        except requests.RequestException:
            fail += 1
        time.sleep(1 / a.rps)
    print(f"sent={a.n} ok={ok} failed={fail} drift={a.drift}")


if __name__ == "__main__":
    main()
