from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RankerConfig:
    family: str
    parameters: dict[str, Any]

    @property
    def name(self) -> str:
        values = "_".join(
            f"{key}-{value}" for key, value in sorted(self.parameters.items())
        )
        return f"{self.family}_{values}"


def default_ranker_configs() -> list[RankerConfig]:
    return [
        RankerConfig("logistic_regression", {"C": 0.1}),
        RankerConfig("logistic_regression", {"C": 1.0}),
        RankerConfig(
            "hist_gradient_boosting",
            {"learning_rate": 0.05, "max_leaf_nodes": 15},
        ),
        RankerConfig(
            "hist_gradient_boosting",
            {"learning_rate": 0.1, "max_leaf_nodes": 15},
        ),
    ]


def build_ranker(config: RankerConfig, *, seed: int):
    if config.family == "logistic_regression":
        return Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(config.parameters["C"]),
                        class_weight="balanced",
                        max_iter=300,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if config.family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=False,
            l2_regularization=0.1,
            learning_rate=float(config.parameters["learning_rate"]),
            max_iter=50,
            max_leaf_nodes=int(config.parameters["max_leaf_nodes"]),
            min_samples_leaf=10,
            random_state=seed,
        )
    raise ValueError(f"Unknown ranker family: {config.family}")


def fit_ranker(model, features: np.ndarray, labels: np.ndarray):
    labeled = labels >= 0
    y = labels[labeled]
    if len(y) == 0:
        raise ValueError("No labeled candidates are available for ranker training")
    if len(np.unique(y)) < 2:
        raise ValueError("Ranker training requires both clicked and unclicked candidates")
    model.fit(features[labeled], y)
    return model


def predict_ranker_scores(model, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Ranker produced non-finite scores")
    return scores


def model_diagnostics(
    model,
    *,
    feature_names: list[str],
) -> list[dict[str, float | str]]:
    classifier = (
        model.named_steps["classifier"]
        if isinstance(model, Pipeline)
        else model
    )
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is None:
        return []
    values = np.asarray(coefficients).reshape(-1)
    return sorted(
        [
            {
                "feature": feature_name,
                "coefficient": float(value),
                "absolute_coefficient": float(abs(value)),
            }
            for feature_name, value in zip(feature_names, values, strict=True)
        ],
        key=lambda row: (-float(row["absolute_coefficient"]), str(row["feature"])),
    )
