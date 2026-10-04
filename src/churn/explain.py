"""Model explainability: SHAP (global + local) and LIME (local)."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn.config import ARTIFACT_DIR, CHAMPION_DIR, resolve  # noqa: E402
from churn.model_io import load_bundle, load_reference  # noqa: E402

log = logging.getLogger(__name__)


def _transform(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pre = pipe[:-1]
    names = [n.split("__", 1)[-1] for n in pre[-1].get_feature_names_out()]
    return np.asarray(pre.transform(X), dtype=float), names


class ShapExplainer:
    """Wraps a SHAP explainer for the final estimator of a pipeline."""

    def __init__(self, pipe: Pipeline, background: pd.DataFrame, max_bg: int = 100):
        self.pipe = pipe
        bg, self.names = _transform(pipe, background.head(max_bg))
        self.explainer = shap.Explainer(pipe[-1], bg)

    def values(self, X: pd.DataFrame) -> np.ndarray:
        Xt, _ = _transform(self.pipe, X)
        kw = {"check_additivity": False} if isinstance(self.explainer, shap.explainers.Tree) else {}
        sv = self.explainer(Xt, **kw).values
        return sv[:, :, 1] if sv.ndim == 3 else sv

    def top_features(self, X: pd.DataFrame, k: int = 5) -> list[dict[str, Any]]:
        sv = self.values(X)[0]
        Xt, _ = _transform(self.pipe, X)
        idx = np.argsort(-np.abs(sv))[:k]
        return [
            {
                "feature": self.names[i],
                "value": round(float(Xt[0, i]), 4),
                "shap": round(float(sv[i]), 4),
                "direction": "increases churn risk" if sv[i] > 0 else "decreases churn risk",
            }
            for i in idx
        ]


def lime_explain(
    pipe: Pipeline, train_X: pd.DataFrame, row: pd.DataFrame, k: int = 8
) -> list[tuple[str, float]]:
    Xt, names = _transform(pipe, train_X)
    explainer = LimeTabularExplainer(
        Xt,
        feature_names=names,
        class_names=["stay", "churn"],
        mode="classification",
        random_state=42,
        discretize_continuous=True,
    )
    rt, _ = _transform(pipe, row)
    exp = explainer.explain_instance(rt[0], pipe[-1].predict_proba, num_features=k, num_samples=1000)
    return [(f, float(w)) for f, w in exp.as_list()]


def run(params: dict[str, Any], model_dir=None) -> dict[str, Any]:
    """Generate global SHAP plots + a LIME explanation for the highest-risk test customer."""
    mdir = model_dir or (
        CHAMPION_DIR if (CHAMPION_DIR / "model.joblib").exists() else ARTIFACT_DIR / "candidate"
    )
    pipe, meta = load_bundle(mdir)
    proc = resolve(params["data"]["processed_dir"])
    dtype = {"SeniorCitizen": str}
    train = pd.read_csv(proc / "train.csv", dtype=dtype)
    test = pd.read_csv(proc / "test.csv", dtype=dtype)
    feats = meta["feature_columns"]
    out = ARTIFACT_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)

    sh = ShapExplainer(pipe, load_reference(mdir) if load_reference(mdir) is not None else train[feats])
    sample = test[feats].sample(min(300, len(test)), random_state=42)
    sv = sh.values(sample)
    Xt, names = _transform(pipe, sample)
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=15)
    plt.savefig(out / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close("all")
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", show=False, max_display=15)
    plt.savefig(out / "shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    importance = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    riskiest = test[feats].iloc[[int(np.argmax(pipe.predict_proba(test[feats])[:, 1]))]]
    result = {
        "global_top_features": {k: round(float(v), 4) for k, v in importance.head(10).items()},
        "local_shap": sh.top_features(riskiest),
        "local_lime": lime_explain(pipe, train[feats].head(500), riskiest),
    }
    import json

    (out / "explainability.json").write_text(json.dumps(result, indent=2))
    log.info("Explainability artifacts saved to %s", out)
    try:
        from churn.storage import sync_reports_to_minio

        sync_reports_to_minio()
    except Exception as exc:  # noqa: BLE001
        log.debug("Explainability report sync to MinIO skipped: %s", exc)
    return result
