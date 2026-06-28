from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
(Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)
(Path(os.environ["XDG_CACHE_HOME"])).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*", category=UserWarning)

try:
    import optuna
except ImportError:  # pragma: no cover - handled with a clear runtime error.
    optuna = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import SVC


LOGGER = logging.getLogger("radiomics_project")
MODEL_ORDER = ["LR", "SVM", "RF", "ET", "GB", "KNN", "MLP"]
ENSEMBLE_MEMBERS = ["RF", "ET", "GB"]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    figures: Path
    tables: Path
    models: Path
    report: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            figures=root / "outputs" / "figures",
            tables=root / "outputs" / "tables",
            models=root / "outputs" / "models",
            report=root / "report",
        )

    def ensure(self) -> None:
        for path in [self.data_raw, self.figures, self.tables, self.models, self.report]:
            path.mkdir(parents=True, exist_ok=True)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove features that are highly correlated with an earlier feature."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CorrelationFilter":
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        if n_features <= 1:
            self.keep_mask_ = np.ones(n_features, dtype=bool)
            return self

        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.triu(np.abs(corr), k=1)
        drop_mask = (upper > self.threshold).any(axis=0)
        self.keep_mask_ = ~drop_mask
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(X)[:, self.keep_mask_]

    def get_support(self) -> np.ndarray:
        self._check_is_fitted()
        return self.keep_mask_

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "keep_mask_"):
            raise RuntimeError("CorrelationFilter must be fitted before transform.")


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Greedy MRMR selector using mutual information and Pearson redundancy."""

    def __init__(self, n_features: int = 30, random_state: int = 42):
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MRMRSelector":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_total = X.shape[1]
        n_select = max(1, min(int(self.n_features), n_total))

        relevance = mutual_info_classif(X, y, discrete_features=False, random_state=self.random_state)
        relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)

        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)

        selected: list[int] = []
        candidates = set(range(n_total))
        first = int(np.argmax(relevance))
        selected.append(first)
        candidates.remove(first)

        while len(selected) < n_select and candidates:
            best_idx = None
            best_score = -np.inf
            for idx in candidates:
                redundancy = float(np.mean(corr[idx, selected])) if selected else 0.0
                score = float(relevance[idx]) - redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(int(best_idx))
            candidates.remove(int(best_idx))

        self.selected_idx_ = np.array(selected, dtype=int)
        self.support_mask_ = np.zeros(n_total, dtype=bool)
        self.support_mask_[self.selected_idx_] = True
        self.relevance_ = relevance
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(X)[:, self.selected_idx_]

    def get_support(self) -> np.ndarray:
        self._check_is_fitted()
        return self.support_mask_

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "selected_idx_"):
            raise RuntimeError("MRMRSelector must be fitted before transform.")


class FittedSoftVotingEnsemble:
    """Average probabilities from already-fitted calibrated estimators."""

    def __init__(self, estimators: dict[str, Any]):
        self.estimators = estimators
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = [est.predict_proba(X) for est in self.estimators.values()]
        return np.mean(probabilities, axis=0)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class FittedWeightedSoftVotingEnsemble:
    """Weighted probability average from fitted calibrated estimators."""

    def __init__(self, estimators: dict[str, Any], weights: dict[str, float]):
        self.estimators = estimators
        total = float(sum(weights.values()))
        self.weights = {name: float(weight) / total for name, weight in weights.items()}
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        weighted = None
        for name, estimator in self.estimators.items():
            proba = estimator.predict_proba(X) * self.weights[name]
            weighted = proba if weighted is None else weighted + proba
        return np.asarray(weighted)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def load_dataset(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    normal_path = raw_dir / "normal_radiomics.csv"
    pap_path = raw_dir / "papilodem_radiomics.csv"
    normal_raw = pd.read_csv(normal_path)
    pap_raw = pd.read_csv(pap_path)

    normal = pd.concat(
        [normal_raw, pd.DataFrame({"Target": 0, "ClassName": "Normal"}, index=normal_raw.index)],
        axis=1,
    ).copy()
    pap = pd.concat(
        [pap_raw, pd.DataFrame({"Target": 1, "ClassName": "Papilledema"}, index=pap_raw.index)],
        axis=1,
    ).copy()

    data = pd.concat([normal, pap], ignore_index=True).copy()
    group_id = data["ClassName"] + "_" + data["PatientIndex"].astype(str)
    data = pd.concat([data, pd.DataFrame({"GroupID": group_id}, index=data.index)], axis=1).copy()
    feature_cols = [col for col in data.columns if col.startswith("Feature_")]
    return data, feature_cols


def write_dataset_summary(data: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    summary = {
        "rows": len(data),
        "patients": int(data["GroupID"].nunique()),
        "normal_rows": int((data["Target"] == 0).sum()),
        "papilledema_rows": int((data["Target"] == 1).sum()),
        "normal_patients": int(data.loc[data["Target"] == 0, "GroupID"].nunique()),
        "papilledema_patients": int(data.loc[data["Target"] == 1, "GroupID"].nunique()),
        "feature_count": len(feature_cols),
        "missing_values": int(data[feature_cols].isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
    }
    pd.DataFrame([summary]).to_csv(output_path, index=False)


def choose_n_splits(groups: pd.Series, y: pd.Series, requested: int) -> int:
    group_labels = pd.DataFrame({"group": groups, "y": y}).drop_duplicates("group")
    min_class_groups = int(group_labels["y"].value_counts().min())
    return max(2, min(requested, min_class_groups))


def make_patient_level_splits(
    data: pd.DataFrame,
    seed: int = 42,
    test_folds: int = 5,
    val_folds: int = 4,
) -> dict[str, np.ndarray]:
    y = data["Target"].to_numpy()
    groups = data["GroupID"].astype(str)

    n_test_splits = choose_n_splits(groups, data["Target"], test_folds)
    outer = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(data, y, groups))

    train_val = data.iloc[train_val_idx].reset_index(drop=False)
    tv_y = train_val["Target"].to_numpy()
    tv_groups = train_val["GroupID"].astype(str)
    n_val_splits = choose_n_splits(tv_groups, train_val["Target"], val_folds)
    inner_holdout = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(inner_holdout.split(train_val, tv_y, tv_groups))

    train_idx = train_val.iloc[train_local]["index"].to_numpy()
    val_idx = train_val.iloc[val_local]["index"].to_numpy()
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def summarize_split(data: pd.DataFrame, splits: dict[str, np.ndarray], output_path: Path) -> None:
    rows = []
    for split_name, idx in splits.items():
        part = data.iloc[idx]
        rows.append(
            {
                "split": split_name,
                "rows": len(part),
                "patients": int(part["GroupID"].nunique()),
                "normal_rows": int((part["Target"] == 0).sum()),
                "papilledema_rows": int((part["Target"] == 1).sum()),
                "normal_patients": int(part.loc[part["Target"] == 0, "GroupID"].nunique()),
                "papilledema_patients": int(part.loc[part["Target"] == 1, "GroupID"].nunique()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def make_preprocessor(correlation_threshold: float) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("finite", FunctionTransformer(replace_non_finite, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("correlation", CorrelationFilter(threshold=correlation_threshold)),
            ("scaler", RobustScaler()),
        ]
    )
    return numeric_pipe


def replace_non_finite(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def make_model_pipeline(
    estimator: Any,
    mrmr_k: int,
    correlation_threshold: float,
    seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(correlation_threshold)),
            ("mrmr", MRMRSelector(n_features=mrmr_k, random_state=seed)),
            ("model", estimator),
        ]
    )


def suggest_model_params(model_name: str, trial: Any, seed: int, fast: bool) -> dict[str, Any]:
    tree_low, tree_high = (50, 160) if fast else (100, 500)
    params: dict[str, Any] = {}
    if model_name == "LR":
        params["C"] = trial.suggest_float("C", 1e-3, 100.0, log=True)
    elif model_name == "SVM":
        params["C"] = trial.suggest_float("C", 1e-2, 100.0, log=True)
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    elif model_name == "RF":
        params["n_estimators"] = trial.suggest_int("n_estimators", tree_low, tree_high)
        params["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 8)
        params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    elif model_name == "ET":
        params["n_estimators"] = trial.suggest_int("n_estimators", tree_low, tree_high)
        params["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 8)
        params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    elif model_name == "GB":
        params["n_estimators"] = trial.suggest_int("n_estimators", tree_low, tree_high)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 1, 5)
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    elif model_name == "KNN":
        params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 31, step=2)
        params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        params["p"] = trial.suggest_categorical("p", [1, 2])
    elif model_name == "MLP":
        params["hidden_layer_sizes"] = trial.suggest_categorical("hidden_layer_sizes", ["32", "64", "64_32", "128_64"])
        params["activation"] = trial.suggest_categorical("activation", ["relu", "tanh"])
        params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    params["mrmr_k"] = trial.suggest_int("mrmr_k", 10, 80, step=5)
    return params


def parse_hidden_layers(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split("_"))


def estimator_from_params(model_name: str, params: dict[str, Any], seed: int) -> Any:
    if model_name == "LR":
        return LogisticRegression(
            C=float(params["C"]),
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    if model_name == "SVM":
        return SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            class_weight="balanced",
            random_state=seed,
        )
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    if model_name == "ET":
        return ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    if model_name == "GB":
        return GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=seed,
        )
    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=int(params["n_neighbors"]),
            weights=params["weights"],
            p=int(params["p"]),
        )
    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=parse_hidden_layers(params["hidden_layer_sizes"]),
            activation=params["activation"],
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            max_iter=700,
            n_iter_no_change=30,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline_from_params(
    model_name: str,
    params: dict[str, Any],
    correlation_threshold: float,
    seed: int,
) -> Pipeline:
    estimator = estimator_from_params(model_name, params, seed)
    return make_model_pipeline(
        estimator=estimator,
        mrmr_k=int(params["mrmr_k"]),
        correlation_threshold=correlation_threshold,
        seed=seed,
    )


def optimize_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    trials: int,
    seed: int,
    correlation_threshold: float,
    cv_folds: int,
    fast: bool,
) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("optuna is not installed. Run: pip install -r requirements.txt")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=model_name)
    n_splits = max(2, min(cv_folds, choose_n_splits(pd.Series(groups), pd.Series(y), cv_folds)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: Any) -> float:
        params = suggest_model_params(model_name, trial, seed, fast)
        pipe = build_pipeline_from_params(model_name, params, correlation_threshold, seed)
        scores = []
        for train_idx, valid_idx in cv.split(X, y, groups):
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[valid_idx])
            scores.append(f1_score(y[valid_idx], pred, average="macro", zero_division=0))
        return float(np.mean(scores))

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {
        "model": model_name,
        "best_macro_f1": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": trials,
    }


def fit_prefit_calibrator(base_pipe: Pipeline, X_cal: pd.DataFrame, y_cal: np.ndarray) -> CalibratedClassifierCV:
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(base_pipe), method="sigmoid", cv=None)
        calibrator.fit(X_cal, y_cal)
        return calibrator
    except ImportError:
        pass

    try:
        calibrator = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=base_pipe, method="sigmoid", cv="prefit")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def get_positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    class_index = list(model.classes_).index(1)
    return proba[:, class_index]


def get_binary_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return get_positive_proba(model, X)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def metrics_from_proba(
    model_name: str,
    y: np.ndarray,
    proba: np.ndarray,
    split: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    metrics = {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "brier_score": brier_score_loss(y, proba),
    }
    metrics["roc_auc"] = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    metrics["pr_auc"] = average_precision_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    return metrics


def evaluate_model(model_name: str, model: Any, X: pd.DataFrame, y: np.ndarray, split: str) -> dict[str, Any]:
    proba = get_positive_proba(model, X)
    return metrics_from_proba(model_name, y, proba, split, threshold=0.5)


def save_classification_report(model_name: str, model: Any, X: pd.DataFrame, y: np.ndarray, output_path: Path) -> None:
    proba = get_positive_proba(model, X)
    pred = (proba >= 0.5).astype(int)
    report = classification_report(
        y,
        pred,
        target_names=["Normal", "Papilledema"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_path)


def collect_cv_scores(
    best_params: dict[str, dict[str, Any]],
    models: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    correlation_threshold: float,
    seed: int,
    cv_folds: int,
) -> pd.DataFrame:
    n_splits = max(2, min(cv_folds, choose_n_splits(pd.Series(groups), pd.Series(y), cv_folds)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 10)
    rows = []
    for model_name in models:
        params = best_params[model_name]
        pipe = build_pipeline_from_params(model_name, params, correlation_threshold, seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[valid_idx])
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "macro_f1": f1_score(y[valid_idx], pred, average="macro", zero_division=0),
                }
            )
    return pd.DataFrame(rows)


def run_statistical_tests(cv_scores: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except ImportError:
        result = pd.DataFrame(
            [{"test": "not_run", "comparison": "scipy_missing", "statistic": np.nan, "p_value": np.nan, "p_value_bonferroni": np.nan}]
        )
        result.to_csv(output_path, index=False)
        return result

    pivot = cv_scores.pivot(index="fold", columns="model", values="macro_f1").dropna(axis=1)
    rows = []
    if pivot.shape[1] >= 3:
        stat, p_value = friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
        rows.append(
            {
                "test": "Friedman",
                "comparison": "all_models",
                "statistic": float(stat),
                "p_value": float(p_value),
                "p_value_bonferroni": float(p_value),
            }
        )

    if "Ensemble" in pivot.columns:
        base_model = "Ensemble"
    else:
        base_model = str(pivot.mean().idxmax())
    comparisons = [col for col in pivot.columns if col != base_model]
    correction = max(1, len(comparisons))
    for col in comparisons:
        try:
            stat, p_value = wilcoxon(pivot[base_model], pivot[col], zero_method="zsplit")
        except ValueError:
            stat, p_value = np.nan, 1.0
        rows.append(
            {
                "test": "Wilcoxon signed-rank",
                "comparison": f"{base_model} vs {col}",
                "statistic": float(stat) if not math.isnan(float(stat)) else np.nan,
                "p_value": float(p_value),
                "p_value_bonferroni": min(1.0, float(p_value) * correction),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def selected_feature_names(fitted_pipeline: Pipeline, feature_cols: list[str]) -> np.ndarray:
    names = np.array(feature_cols)
    variance_mask = fitted_pipeline.named_steps["preprocess"].named_steps["variance"].get_support()
    names = names[variance_mask]
    corr_mask = fitted_pipeline.named_steps["preprocess"].named_steps["correlation"].get_support()
    names = names[corr_mask]
    mrmr_mask = fitted_pipeline.named_steps["mrmr"].get_support()
    return names[mrmr_mask]


def compute_feature_importance(
    best_params: dict[str, dict[str, Any]],
    cv_summary: dict[str, float],
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    correlation_threshold: float,
    seed: int,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    candidates = [model for model in ENSEMBLE_MEMBERS if model in best_params]
    if not candidates:
        pd.DataFrame().to_csv(output_csv, index=False)
        return pd.DataFrame()
    model_name = max(candidates, key=lambda name: cv_summary.get(name, -np.inf))
    pipe = build_pipeline_from_params(model_name, best_params[model_name], correlation_threshold, seed)
    pipe.fit(X, y)

    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        pd.DataFrame().to_csv(output_csv, index=False)
        return pd.DataFrame()
    names = selected_feature_names(pipe, feature_cols)
    importances = np.asarray(model.feature_importances_)
    table = pd.DataFrame({"feature": names, "importance": importances})
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table.to_csv(output_csv, index=False)

    top = table.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top, x="importance", y="feature", color="#2f6f9f")
    plt.title(f"Top radiomics features ({model_name})")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return table


def plot_roc_curves(models: dict[str, Any], X: pd.DataFrame, y: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        proba = get_positive_proba(model, X)
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("ROC Curve - Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pr_curves(models: dict[str, Any], X: pd.DataFrame, y: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        proba = get_positive_proba(model, X)
        precision, recall, _ = precision_recall_curve(y, proba)
        auc = average_precision_score(y, proba)
        plt.plot(recall, precision, label=f"{name} (AP={auc:.3f})")
    baseline = float(np.mean(y))
    plt.axhline(baseline, linestyle="--", color="gray", linewidth=1, label=f"Baseline={baseline:.3f}")
    plt.title("Precision-Recall Curve - Test Set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrix(model: Any, X: pd.DataFrame, y: np.ndarray, output_path: Path) -> None:
    proba = get_positive_proba(model, X)
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Papilledema"], yticklabels=["Normal", "Papilledema"])
    plt.title("Confusion Matrix - Ensemble Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_calibration_curves(models: dict[str, Any], X: pd.DataFrame, y: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        proba = get_positive_proba(model, X)
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    plt.title("Calibration Curve - Test Set")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_model_comparison(test_metrics: pd.DataFrame, output_path: Path) -> None:
    plot_df = test_metrics.melt(
        id_vars=["model"],
        value_vars=["macro_f1", "roc_auc", "pr_auc", "balanced_accuracy"],
        var_name="metric",
        value_name="score",
    )
    plot_df["display_model"] = plot_df["model"].replace({"WeightedEnsemble": "WeightedEns"})
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="display_model", y="score", hue="metric")
    plt.ylim(0, 1.02)
    plt.title("Model Comparison - Test Set")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def optimize_ensemble_weights(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_csv: Path,
    output_png: Path,
    grid_steps: int = 10,
) -> FittedWeightedSoftVotingEnsemble | None:
    if not all(name in fitted_models for name in ENSEMBLE_MEMBERS):
        pd.DataFrame().to_csv(output_csv, index=False)
        return None

    estimators = {name: fitted_models[name] for name in ENSEMBLE_MEMBERS}
    val_probs = {name: get_positive_proba(model, X_val) for name, model in estimators.items()}
    best_weights: dict[str, float] | None = None
    best_score = -np.inf
    best_brier = np.inf

    for raw_weights in product(range(grid_steps + 1), repeat=len(ENSEMBLE_MEMBERS)):
        if sum(raw_weights) != grid_steps or sum(raw_weights) == 0:
            continue
        weights = {name: weight / grid_steps for name, weight in zip(ENSEMBLE_MEMBERS, raw_weights)}
        proba = sum(val_probs[name] * weights[name] for name in ENSEMBLE_MEMBERS)
        pred = (proba >= 0.5).astype(int)
        macro = f1_score(y_val, pred, average="macro", zero_division=0)
        brier = brier_score_loss(y_val, proba)
        if macro > best_score or (math.isclose(macro, best_score) and brier < best_brier):
            best_score = float(macro)
            best_brier = float(brier)
            best_weights = weights

    if best_weights is None:
        pd.DataFrame().to_csv(output_csv, index=False)
        return None

    optimized = FittedWeightedSoftVotingEnsemble(estimators, best_weights)
    test_proba = get_positive_proba(optimized, X_test)
    test_metrics = metrics_from_proba("WeightedEnsemble", y_test, test_proba, "test", threshold=0.5)
    record: dict[str, Any] = {
        "model": "WeightedEnsemble",
        "rf_weight": best_weights["RF"],
        "et_weight": best_weights["ET"],
        "gb_weight": best_weights["GB"],
        "validation_macro_f1": best_score,
        "validation_brier_score": best_brier,
    }
    for key, value in test_metrics.items():
        if key not in {"model", "split"}:
            record[f"test_{key}"] = value
    pd.DataFrame([record]).to_csv(output_csv, index=False)

    plt.figure(figsize=(6.5, 4.5))
    sns.barplot(x=list(best_weights.keys()), y=list(best_weights.values()), color="#356c9b")
    plt.ylim(0, 1.0)
    plt.title("Optimized Soft Voting Ensemble Weights")
    plt.xlabel("Model")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return optimized


def run_threshold_optimization(
    fitted_models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for model_name, model in fitted_models.items():
        val_proba = get_positive_proba(model, X_val)
        val_scores = [
            f1_score(y_val, (val_proba >= threshold).astype(int), average="macro", zero_division=0)
            for threshold in thresholds
        ]
        best_idx = int(np.argmax(val_scores))
        best_threshold = float(thresholds[best_idx])
        test_proba = get_positive_proba(model, X_test)
        test_metrics = metrics_from_proba(model_name, y_test, test_proba, "test_threshold_optimized", threshold=best_threshold)
        rows.append(
            {
                "model": model_name,
                "best_threshold": best_threshold,
                "validation_macro_f1": float(val_scores[best_idx]),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_brier_score": test_metrics["brier_score"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
            }
        )

    result = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    result.to_csv(output_csv, index=False)

    plot_df = result.copy()
    plot_df["display_model"] = plot_df["model"].replace({"WeightedEnsemble": "WeightedEns"})
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=plot_df, x="display_model", y="test_macro_f1", color="#2f7f5f")
    for container in ax.containers:
        ax.bar_label(container, labels=[f"t={threshold:.2f}" for threshold in plot_df["best_threshold"]], fontsize=8, padding=3)
    plt.ylim(0, 1.05)
    plt.title("Threshold Optimization - Validation Selected, Test Reported")
    plt.xlabel("Model")
    plt.ylabel("Test Macro-F1")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return result


def transformed_selected_space(fitted_pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    preprocessed = fitted_pipeline.named_steps["preprocess"].transform(X)
    return fitted_pipeline.named_steps["mrmr"].transform(preprocessed)


def run_feature_stability(
    model_name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    correlation_threshold: float,
    seed: int,
    cv_folds: int,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    n_splits = max(2, min(cv_folds, choose_n_splits(pd.Series(groups), pd.Series(y), cv_folds)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 30)
    counts: dict[str, int] = {}
    for train_idx, _valid_idx in cv.split(X, y, groups):
        pipe = build_pipeline_from_params(model_name, params, correlation_threshold, seed)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        for feature in selected_feature_names(pipe, feature_cols):
            counts[str(feature)] = counts.get(str(feature), 0) + 1

    result = pd.DataFrame(
        [
            {
                "feature": feature,
                "selection_count": count,
                "n_folds": n_splits,
                "stability_percent": 100.0 * count / n_splits,
            }
            for feature, count in counts.items()
        ]
    )
    if result.empty:
        result.to_csv(output_csv, index=False)
        return result
    result = result.sort_values(["selection_count", "feature"], ascending=[False, True]).reset_index(drop=True)
    result.to_csv(output_csv, index=False)

    top = result.head(20).iloc[::-1]
    plt.figure(figsize=(8.5, 6.5))
    sns.barplot(data=top, x="stability_percent", y="feature", color="#7b5ca7")
    plt.xlim(0, 100)
    plt.title(f"Feature Stability Across Grouped Folds ({model_name})")
    plt.xlabel("Selection frequency (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return result


def run_shap_analysis(
    best_params: dict[str, dict[str, Any]],
    cv_summary: dict[str, float],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    correlation_threshold: float,
    seed: int,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        pd.DataFrame([{"status": "skipped", "reason": "shap_missing"}]).to_csv(output_csv, index=False)
        return pd.DataFrame()

    candidates = [model for model in ENSEMBLE_MEMBERS if model in best_params]
    if not candidates:
        pd.DataFrame().to_csv(output_csv, index=False)
        return pd.DataFrame()
    model_name = max(candidates, key=lambda name: cv_summary.get(name, -np.inf))
    pipe = build_pipeline_from_params(model_name, best_params[model_name], correlation_threshold, seed)
    pipe.fit(X_train, y_train)
    X_test_selected = transformed_selected_space(pipe, X_test)
    feature_names = selected_feature_names(pipe, feature_cols)
    sample_count = min(120, X_test_selected.shape[0])
    X_explain = X_test_selected[:sample_count]

    model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[1])
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3 and shap_array.shape[2] >= 2:
            shap_array = shap_array[:, :, 1]
        elif shap_array.ndim == 3:
            shap_array = shap_array[:, :, 0]

    mean_abs = np.abs(shap_array).mean(axis=0)
    result = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result.to_csv(output_csv, index=False)

    top = result.head(20).iloc[::-1]
    plt.figure(figsize=(8.5, 6.5))
    sns.barplot(data=top, x="mean_abs_shap", y="feature", color="#b45f3c")
    plt.title(f"SHAP Mean Absolute Impact ({model_name})")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return result


def run_lime_analysis(
    best_params: dict[str, dict[str, Any]],
    cv_summary: dict[str, float],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_cols: list[str],
    correlation_threshold: float,
    seed: int,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        pd.DataFrame([{"status": "skipped", "reason": "lime_missing"}]).to_csv(output_csv, index=False)
        return pd.DataFrame()

    proba_models = [name for name in best_params if name != "SVM"]
    if not proba_models:
        pd.DataFrame().to_csv(output_csv, index=False)
        return pd.DataFrame()
    model_name = max(proba_models, key=lambda name: cv_summary.get(name, -np.inf))
    pipe = build_pipeline_from_params(model_name, best_params[model_name], correlation_threshold, seed)
    pipe.fit(X_train, y_train)
    X_train_selected = transformed_selected_space(pipe, X_train)
    X_test_selected = transformed_selected_space(pipe, X_test)
    feature_names = selected_feature_names(pipe, feature_cols)
    model = pipe.named_steps["model"]

    explainer = LimeTabularExplainer(
        training_data=X_train_selected,
        feature_names=list(feature_names),
        class_names=["Normal", "Papilledema"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )
    proba = model.predict_proba(X_test_selected)
    pred = (proba[:, 1] >= 0.5).astype(int)
    positive_correct = np.where((y_test == 1) & (pred == 1))[0]
    positive_any = np.where(y_test == 1)[0]
    explain_index = int(positive_correct[0] if len(positive_correct) else positive_any[0] if len(positive_any) else 0)

    explanation = explainer.explain_instance(
        X_test_selected[explain_index],
        model.predict_proba,
        num_features=min(10, X_test_selected.shape[1]),
        labels=(1,),
    )
    rows = [
        {
            "model": model_name,
            "test_index": explain_index,
            "feature_rule": feature_rule,
            "lime_weight": weight,
        }
        for feature_rule, weight in explanation.as_list(label=1)
    ]
    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)

    plot_df = result.iloc[::-1].copy()
    colors_for_bars = ["#2f7f5f" if value >= 0 else "#9b3d3d" for value in plot_df["lime_weight"]]
    plt.figure(figsize=(9, 5.8))
    plt.barh(plot_df["feature_rule"], plot_df["lime_weight"], color=colors_for_bars)
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.title(f"LIME Local Explanation ({model_name}, test sample {explain_index})")
    plt.xlabel("Contribution to Papilledema probability")
    plt.ylabel("Feature rule")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return result


def run_nested_cross_validation(
    models: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    correlation_threshold: float,
    seed: int,
    outer_folds: int,
    inner_folds: int,
    nested_trials: int,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    if nested_trials <= 0:
        pd.DataFrame().to_csv(output_csv, index=False)
        return pd.DataFrame()

    n_outer = max(2, min(outer_folds, choose_n_splits(pd.Series(groups), pd.Series(y), outer_folds)))
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed + 40)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
        LOGGER.info("Nested CV outer fold %s/%s", fold, n_outer)
        fold_records = []
        for model_name in models:
            record = optimize_model(
                model_name=model_name,
                X=X.iloc[train_idx],
                y=y[train_idx],
                groups=groups[train_idx],
                trials=nested_trials,
                seed=seed + fold,
                correlation_threshold=correlation_threshold,
                cv_folds=inner_folds,
                fast=True,
            )
            fold_records.append(record)

        best_record = max(fold_records, key=lambda item: item["best_macro_f1"])
        best_model = str(best_record["model"])
        pipe = build_pipeline_from_params(best_model, best_record["best_params"], correlation_threshold, seed + fold)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        score = get_binary_score(pipe, X.iloc[test_idx])
        roc_auc = roc_auc_score(y[test_idx], score) if len(np.unique(y[test_idx])) == 2 else np.nan
        pr_auc = average_precision_score(y[test_idx], score) if len(np.unique(y[test_idx])) == 2 else np.nan
        rows.append(
            {
                "fold": fold,
                "selected_model": best_model,
                "inner_best_macro_f1": best_record["best_macro_f1"],
                "outer_accuracy": accuracy_score(y[test_idx], pred),
                "outer_macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
                "outer_balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
                "outer_roc_auc": roc_auc,
                "outer_pr_auc": pr_auc,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)

    plt.figure(figsize=(7.5, 4.8))
    sns.barplot(data=result, x="fold", y="outer_macro_f1", hue="selected_model")
    plt.ylim(0, 1.05)
    plt.title("Nested Cross-Validation Outer Fold Performance")
    plt.xlabel("Outer fold")
    plt.ylabel("Outer Macro-F1")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return result


def make_report_pdf(
    paths: ProjectPaths,
    dataset_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    test_metrics: pd.DataFrame,
    top_features: pd.DataFrame,
) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except ImportError:
        LOGGER.warning("reportlab is not installed; PDF report was skipped.")
        return

    pdf_path = paths.report / "final_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(Paragraph("Radiomics-Based Papilledema Classification", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "This report summarizes a leakage-safe machine learning pipeline for binary classification "
            "of Normal and Papilledema samples using 746 radiomics features.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))

    ds = dataset_summary.iloc[0].to_dict()
    story.append(Paragraph("1. Dataset", styles["Heading1"]))
    story.append(
        Paragraph(
            f"The dataset contains {int(ds['rows'])} samples from {int(ds['patients'])} patients. "
            f"There are {int(ds['normal_rows'])} Normal samples and {int(ds['papilledema_rows'])} "
            f"Papilledema samples. Patient-level grouping was used for all splits.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(make_report_table(split_summary, colors.HexColor("#d9eaf7")))

    story.append(Paragraph("2. Methodology", styles["Heading1"]))
    story.append(
        Paragraph(
            "The pipeline applies median imputation, low-variance filtering, Pearson correlation filtering "
            "(threshold 0.95), RobustScaler normalization, MRMR feature selection, Optuna hyperparameter "
            "optimization with TPE sampling, sigmoid calibration, and a soft-voting ensemble over RF, ET, and GB.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("3. Test Results", styles["Heading1"]))
    display_cols = ["model", "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]
    story.append(make_report_table(test_metrics[display_cols].round(4), colors.HexColor("#e8f2e4")))

    story.append(PageBreak())
    if not top_features.empty:
        story.append(Paragraph("4. Top Features", styles["Heading1"]))
        story.append(make_report_table(top_features.head(10).round(6), colors.HexColor("#f6ead4")))
        story.append(Spacer(1, 14))

    story.append(Paragraph("5. Figures", styles["Heading1"]))
    for figure_name in [
        "roc_curve.png",
        "precision_recall_curve.png",
        "confusion_matrix.png",
        "calibration_curve.png",
        "model_comparison.png",
        "feature_importance.png",
    ]:
        figure_path = paths.figures / figure_name
        if figure_path.exists():
            story.append(
                KeepTogether(
                    [
                        Paragraph(figure_name.replace("_", " ").replace(".png", "").title(), styles["Heading2"]),
                        Image(str(figure_path), width=460, height=300),
                        Spacer(1, 10),
                    ]
                )
            )

    doc.build(story)


def make_report_table(df: pd.DataFrame, header_color: Any) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8b8b8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    paths = ProjectPaths.from_root(Path(args.project_root).resolve())
    paths.ensure()

    sns.set_theme(style="whitegrid", context="notebook")
    data, feature_cols = load_dataset(paths.data_raw)
    write_dataset_summary(data, feature_cols, paths.tables / "dataset_summary.csv")
    dataset_summary = pd.read_csv(paths.tables / "dataset_summary.csv")

    splits = make_patient_level_splits(data, seed=args.seed)
    summarize_split(data, splits, paths.tables / "split_summary.csv")
    split_summary = pd.read_csv(paths.tables / "split_summary.csv")

    X = data[feature_cols]
    y = data["Target"].to_numpy()
    groups = data["GroupID"].to_numpy()

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    train_val_idx = np.concatenate([train_idx, val_idx])

    X_train, y_train, groups_train = X.iloc[train_idx], y[train_idx], groups[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    X_train_val, y_train_val, groups_train_val = X.iloc[train_val_idx], y[train_val_idx], groups[train_val_idx]

    selected_models = args.models or MODEL_ORDER
    best_records = []
    best_params: dict[str, dict[str, Any]] = {}
    for model_name in selected_models:
        LOGGER.info("Optimizing %s with %s trial(s)", model_name, args.trials)
        record = optimize_model(
            model_name=model_name,
            X=X_train,
            y=y_train,
            groups=groups_train,
            trials=args.trials,
            seed=args.seed,
            correlation_threshold=args.correlation_threshold,
            cv_folds=args.cv_folds,
            fast=args.fast,
        )
        best_records.append(record)
        best_params[model_name] = record["best_params"]

    pd.DataFrame(best_records).to_csv(paths.tables / "optuna_best_scores.csv", index=False)
    with (paths.tables / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    fitted_models: dict[str, Any] = {}
    validation_rows = []
    test_rows = []
    for model_name in selected_models:
        LOGGER.info("Fitting calibrated %s", model_name)
        base_pipe = build_pipeline_from_params(model_name, best_params[model_name], args.correlation_threshold, args.seed)
        base_pipe.fit(X_train, y_train)
        calibrated = fit_prefit_calibrator(base_pipe, X_val, y_val)
        fitted_models[model_name] = calibrated
        validation_rows.append(evaluate_model(model_name, calibrated, X_val, y_val, "validation"))
        test_rows.append(evaluate_model(model_name, calibrated, X_test, y_test, "test"))
        save_classification_report(model_name, calibrated, X_test, y_test, paths.tables / f"classification_report_{model_name}.csv")

    ensemble_inputs = {name: fitted_models[name] for name in ENSEMBLE_MEMBERS if name in fitted_models}
    if len(ensemble_inputs) == len(ENSEMBLE_MEMBERS):
        ensemble = FittedSoftVotingEnsemble(ensemble_inputs)
        fitted_models["Ensemble"] = ensemble
        validation_rows.append(evaluate_model("Ensemble", ensemble, X_val, y_val, "validation"))
        test_rows.append(evaluate_model("Ensemble", ensemble, X_test, y_test, "test"))
        save_classification_report("Ensemble", ensemble, X_test, y_test, paths.tables / "classification_report_Ensemble.csv")
    else:
        LOGGER.warning("Ensemble skipped because RF, ET, and GB were not all selected.")

    weighted_ensemble = optimize_ensemble_weights(
        fitted_models=fitted_models,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        output_csv=paths.tables / "ensemble_optimization.csv",
        output_png=paths.figures / "ensemble_optimization.png",
    )
    if weighted_ensemble is not None:
        fitted_models["WeightedEnsemble"] = weighted_ensemble
        validation_rows.append(evaluate_model("WeightedEnsemble", weighted_ensemble, X_val, y_val, "validation"))
        test_rows.append(evaluate_model("WeightedEnsemble", weighted_ensemble, X_test, y_test, "test"))
        save_classification_report(
            "WeightedEnsemble",
            weighted_ensemble,
            X_test,
            y_test,
            paths.tables / "classification_report_WeightedEnsemble.csv",
        )

    validation_metrics = pd.DataFrame(validation_rows)
    test_metrics = pd.DataFrame(test_rows)
    validation_metrics.to_csv(paths.tables / "model_performance_validation.csv", index=False)
    test_metrics.to_csv(paths.tables / "model_performance_test.csv", index=False)
    if "MLP" in test_metrics["model"].values:
        test_metrics.loc[test_metrics["model"] == "MLP"].to_csv(paths.tables / "deep_learning_mlp_results.csv", index=False)

    cv_scores = collect_cv_scores(
        best_params=best_params,
        models=selected_models,
        X=X_train_val,
        y=y_train_val,
        groups=groups_train_val,
        correlation_threshold=args.correlation_threshold,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )
    if "Ensemble" in fitted_models:
        ensemble_rows = []
        pivot_models = [name for name in ENSEMBLE_MEMBERS if name in best_params]
        n_splits = max(2, min(args.cv_folds, choose_n_splits(pd.Series(groups_train_val), pd.Series(y_train_val), args.cv_folds)))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + 20)
        for fold, (tr, va) in enumerate(cv.split(X_train_val, y_train_val, groups_train_val), start=1):
            fold_estimators = {}
            for member in pivot_models:
                pipe = build_pipeline_from_params(member, best_params[member], args.correlation_threshold, args.seed)
                pipe.fit(X_train_val.iloc[tr], y_train_val[tr])
                fold_estimators[member] = pipe
            fold_ensemble = FittedSoftVotingEnsemble(fold_estimators)
            pred = fold_ensemble.predict(X_train_val.iloc[va])
            ensemble_rows.append({"model": "Ensemble", "fold": fold, "macro_f1": f1_score(y_train_val[va], pred, average="macro", zero_division=0)})
        cv_scores = pd.concat([cv_scores, pd.DataFrame(ensemble_rows)], ignore_index=True)
    cv_scores.to_csv(paths.tables / "cv_macro_f1_scores.csv", index=False)
    run_statistical_tests(cv_scores, paths.tables / "statistical_tests.csv")

    cv_summary = {record["model"]: record["best_macro_f1"] for record in best_records}
    stability_model = max(selected_models, key=lambda name: cv_summary.get(name, -np.inf))
    run_feature_stability(
        model_name=stability_model,
        params=best_params[stability_model],
        X=X_train_val,
        y=y_train_val,
        groups=groups_train_val,
        feature_cols=feature_cols,
        correlation_threshold=args.correlation_threshold,
        seed=args.seed,
        cv_folds=args.cv_folds,
        output_csv=paths.tables / "feature_stability.csv",
        output_png=paths.figures / "feature_stability.png",
    )
    run_nested_cross_validation(
        models=selected_models,
        X=X_train_val,
        y=y_train_val,
        groups=groups_train_val,
        correlation_threshold=args.correlation_threshold,
        seed=args.seed,
        outer_folds=args.nested_outer_folds,
        inner_folds=args.nested_inner_folds,
        nested_trials=args.nested_trials,
        output_csv=paths.tables / "nested_cv_results.csv",
        output_png=paths.figures / "nested_cv_results.png",
    )
    run_shap_analysis(
        best_params=best_params,
        cv_summary=cv_summary,
        X_train=X_train_val,
        y_train=y_train_val,
        X_test=X_test,
        feature_cols=feature_cols,
        correlation_threshold=args.correlation_threshold,
        seed=args.seed,
        output_csv=paths.tables / "shap_summary.csv",
        output_png=paths.figures / "shap_summary.png",
    )
    run_lime_analysis(
        best_params=best_params,
        cv_summary=cv_summary,
        X_train=X_train_val,
        y_train=y_train_val,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        correlation_threshold=args.correlation_threshold,
        seed=args.seed,
        output_csv=paths.tables / "lime_explanation.csv",
        output_png=paths.figures / "lime_explanation.png",
    )
    run_threshold_optimization(
        fitted_models=fitted_models,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        output_csv=paths.tables / "threshold_optimization.csv",
        output_png=paths.figures / "threshold_optimization.png",
    )

    top_features = compute_feature_importance(
        best_params=best_params,
        cv_summary=cv_summary,
        X=X_train_val,
        y=y_train_val,
        feature_cols=feature_cols,
        correlation_threshold=args.correlation_threshold,
        seed=args.seed,
        output_csv=paths.tables / "top_features.csv",
        output_png=paths.figures / "feature_importance.png",
    )

    plot_models = fitted_models
    plot_roc_curves(plot_models, X_test, y_test, paths.figures / "roc_curve.png")
    plot_pr_curves(plot_models, X_test, y_test, paths.figures / "precision_recall_curve.png")
    if "Ensemble" in fitted_models:
        plot_confusion_matrix(fitted_models["Ensemble"], X_test, y_test, paths.figures / "confusion_matrix.png")
    else:
        best_name = str(test_metrics.sort_values("macro_f1", ascending=False).iloc[0]["model"])
        plot_confusion_matrix(fitted_models[best_name], X_test, y_test, paths.figures / "confusion_matrix.png")
    plot_calibration_curves(plot_models, X_test, y_test, paths.figures / "calibration_curve.png")
    plot_model_comparison(test_metrics, paths.figures / "model_comparison.png")

    make_report_pdf(paths, dataset_summary, split_summary, test_metrics.round(4), top_features.head(10))
    LOGGER.info("Done. Tables: %s | Figures: %s | Report: %s", paths.tables, paths.figures, paths.report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the radiomics papilledema classification pipeline.")
    parser.add_argument("--project-root", default=".", help="Project root containing data/raw.")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model. Assignment asks for at least 50.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Inner StratifiedGroupKFold count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--correlation-threshold", type=float, default=0.95, help="Pearson correlation drop threshold.")
    parser.add_argument("--nested-outer-folds", type=int, default=3, help="Outer folds for bonus nested cross-validation.")
    parser.add_argument("--nested-inner-folds", type=int, default=3, help="Inner folds for bonus nested cross-validation.")
    parser.add_argument("--nested-trials", type=int, default=5, help="Optuna trials per model inside each nested CV outer fold.")
    parser.add_argument("--fast", action="store_true", help="Use smaller tree search ranges for quick smoke tests.")
    parser.add_argument("--models", nargs="*", choices=MODEL_ORDER, help="Optional subset of models.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
