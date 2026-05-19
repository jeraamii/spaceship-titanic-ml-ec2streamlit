"""Build sklearn pipelines for each candidate model.

Binary classification: predict whether a passenger was Transported (True/False).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_pipelines() -> dict:
    """Return a dict of {name: pipeline} for all candidate models."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )),
        ]),
    }
