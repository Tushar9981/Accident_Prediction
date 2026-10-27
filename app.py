from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

CATEGORICAL = ["weather", "road_condition", "time_of_day"]
NUMERIC = ["speed_limit", "driver_age", "visibility_km"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "accident_severity"


# --------------------------------------------------------------------------- #
# Synthetic data (swap this out for your real accident records when you have them)
# --------------------------------------------------------------------------- #
def _risk_score(df: pd.DataFrame, drift: bool = False) -> np.ndarray:
    """Ground-truth risk. drift=True changes the 'rules of the world' (e.g. fog gets deadlier)."""
    weather_w = {"Clear": 0.0, "Rain": 1.2, "Fog": 5.0 if drift else 2.0, "Snow": 2.5}
    road_w = df.road_condition.map({"Dry": 0.0, "Wet": 1.5, "Icy": 3.0}).to_numpy()
    night = (df.time_of_day == "Night").to_numpy()
    age = df.driver_age.to_numpy()
    speed = df.speed_limit.to_numpy()
    vis_gap = 10.0 - df.visibility_km.to_numpy()
    return (
        df.weather.map(weather_w).to_numpy()
        + road_w
        + night * (3.0 if drift else 1.3)
        + speed / (12.0 if drift else 20.0)
        + (age < 22) * 1.8
        + (age > 68) * 1.2
        + vis_gap * 0.4
        + night * vis_gap * 0.15          # darkness makes bad visibility worse
        + (speed / 70.0) * road_w * 0.8   # speed hurts more on slippery roads
    )


def generate_data(n: int = 5000, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weather = rng.choice(
        ["Clear", "Rain", "Fog", "Snow"], n,
        p=[0.40, 0.25, 0.25, 0.10] if drift else [0.50, 0.25, 0.15, 0.10],
    )
    road = np.empty(n, dtype=object)
    road_probs = {
        "Clear": (["Dry", "Wet"], [0.9, 0.1]),
        "Rain": (["Wet", "Dry"], [0.85, 0.15]),
        "Fog": (["Icy", "Wet", "Dry"], [0.6, 0.3, 0.1]),
        "Snow": (["Icy", "Wet", "Dry"], [0.6, 0.3, 0.1]),
    }
    for w, (opts, p) in road_probs.items():
        mask = weather == w
        road[mask] = rng.choice(opts, mask.sum(), p=p)

    df = pd.DataFrame({
        "weather": weather,
        "road_condition": road,
        "time_of_day": rng.choice(["Day", "Night"], n, p=[0.6, 0.4]),
        "speed_limit": rng.choice([25, 35, 45, 55, 65, 70], n),
        "driver_age": rng.integers(18, 75, n),
        "visibility_km": rng.uniform(0.5, 10.0, n),
    })
    # Bernoulli sampling = real-world noise. Same conditions don't always give the same outcome.
    prob = 1.0 / (1.0 + np.exp(-(_risk_score(df, drift) - 7.5)))
    df[TARGET] = rng.binomial(1, prob)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering + pipeline
# --------------------------------------------------------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df[FEATURES].copy()
    slippery = d.road_condition.map({"Dry": 0, "Wet": 1, "Icy": 2}).fillna(0)
    night = (d.time_of_day == "Night").astype(int)
    vis_gap = 10.0 - d.visibility_km
    d["night_x_vis_gap"] = night * vis_gap
    d["speed_x_slippery"] = d.speed_limit * slippery
    d["speed_per_km_vis"] = d.speed_limit / d.visibility_km.clip(lower=0.5)
    d["age_risk_group"] = np.select([d.driver_age < 22, d.driver_age > 68], [1, 2], default=0)
    return d


PARAM_SPACE = {
    "clf__learning_rate": [0.03, 0.05, 0.08, 0.12],
    "clf__max_leaf_nodes": [8, 15, 31],
    "clf__min_samples_leaf": [10, 20, 40],
    "clf__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    "clf__max_depth": [3, 4, 6, None],
}


def _build_pipeline(params: dict | None = None) -> Pipeline:
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(
        max_iter=400, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=42,
    )
    pipe = Pipeline([("features", FunctionTransformer(add_features)), ("prep", prep), ("clf", clf)])
    if params:
        pipe.set_params(**params)
    return pipe


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


# --------------------------------------------------------------------------- #
# Drift metric
# --------------------------------------------------------------------------- #
def psi(ref, new, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 big shift."""
    ref, new = pd.Series(ref), pd.Series(new)
    if pd.api.types.is_numeric_dtype(ref):
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(new, edges)[0] / len(new)
    else:
        cats = sorted(set(ref) | set(new))
        p = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        q = new.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


# --------------------------------------------------------------------------- #
# The adaptive model
# --------------------------------------------------------------------------- #
class AdaptiveAccidentModel:
    def __init__(self, model_dir="models", window=6000, half_life=1000,
                 psi_alert=0.2, loss_alert=1.15, seed=42):
        self.model_dir = Path(model_dir)
        self.window = window          # max rows of history kept for retraining
        self.half_life = half_life    # a row this many steps old counts half as much
        self.psi_alert = psi_alert    # input-drift threshold
        self.loss_alert = loss_alert  # retrain if log-loss is 15% worse than baseline
        self.seed = seed
        self.pipe_ = self.calibrator_ = None
        self.threshold_ = 0.5
        self.best_params_: dict = {}
        self.baseline_: dict = {}
        self.history_ = self.reference_ = None
        self.version_ = 0
        self.log_: list[dict] = []

    # ---- training ---------------------------------------------------------
    def _train(self, df, weights, params=None, tune=False):
        tr, cal, w_tr, w_cal = train_test_split(
            df, weights, test_size=0.15, stratify=df[TARGET], random_state=self.seed)
        pipe = _build_pipeline(params)
        if tune:
            search = RandomizedSearchCV(
                pipe, PARAM_SPACE, n_iter=12, scoring="neg_log_loss",
                cv=StratifiedKFold(3, shuffle=True, random_state=self.seed),
                random_state=self.seed, n_jobs=-1)
            search.fit(tr[FEATURES], tr[TARGET], clf__sample_weight=w_tr)
            pipe, params = search.best_estimator_, search.best_params_
        else:
            pipe.fit(tr[FEATURES], tr[TARGET], clf__sample_weight=w_tr)

        # Platt scaling on held-out rows -> probabilities you can actually trust
        z = _logit(pipe.predict_proba(cal[FEATURES])[:, 1])
        calibrator = LogisticRegression(C=1e6).fit(z, cal[TARGET], sample_weight=w_cal)
        p_cal = calibrator.predict_proba(z)[:, 1]

        # Decision threshold that maximises F1 instead of blindly using 0.5
        grid = np.linspace(0.2, 0.8, 61)
        f1s = [f1_score(cal[TARGET], p_cal >= t, sample_weight=w_cal) for t in grid]
        return pipe, calibrator, float(grid[int(np.argmax(f1s))]), params

    def fit(self, df: pd.DataFrame, tune: bool = True):
        df = df.reset_index(drop=True)
        train_df, hold = train_test_split(
            df, test_size=0.2, stratify=df[TARGET], random_state=self.seed)
        self.pipe_, self.calibrator_, self.threshold_, self.best_params_ = self._train(
            train_df, np.ones(len(train_df)), tune=tune)
        self.baseline_ = self._score(hold)   # what "healthy" looks like
        self.history_ = df.tail(self.window).reset_index(drop=True)
        self.reference_ = self.history_[FEATURES].copy()
        self.version_ = 1
        self._log("fit", rows=len(df), **self.baseline_)
        return self

    # ---- inference --------------------------------------------------------
    def _proba(self, df) -> np.ndarray:
        raw = self.pipe_.predict_proba(df[FEATURES])[:, 1]
        return self.calibrator_.predict_proba(_logit(raw))[:, 1]

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self._proba(df)
        tier = pd.cut(p, [-0.001, 0.33, 0.66, 1.001], labels=["Low", "Medium", "High"])
        return pd.DataFrame({
            "risk_probability": np.round(p, 3),
            "risk_tier": np.asarray(tier),
            "severe_flag": p >= self.threshold_,
        }, index=df.index)

    def _score(self, df) -> dict:
        p, y = self._proba(df), df[TARGET]
        return {
            "auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "f1": f1_score(y, p >= self.threshold_),
            "accuracy": float(((p >= self.threshold_) == y).mean()),
        }

    def evaluate(self, df: pd.DataFrame) -> dict:
        return {k: round(float(v), 4) for k, v in self._score(df).items()}

    def explain(self, df: pd.DataFrame, n_repeats: int = 5) -> pd.Series:
        """How much does log-loss get worse when a column is shuffled? Bigger = more important."""
        r = permutation_importance(
            self.pipe_, df[FEATURES], df[TARGET], scoring="neg_log_loss",
            n_repeats=n_repeats, random_state=self.seed)
        return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)

    # ---- adaptation -------------------------------------------------------
    def check_drift(self, new_df: pd.DataFrame) -> dict:
        psis = {c: psi(self.reference_[c], new_df[c]) for c in FEATURES}
        out = {"psi": psis, "max_psi": max(psis.values())}
        out["covariate_drift"] = out["max_psi"] >= self.psi_alert
        out["concept_drift"] = False
        if TARGET in new_df:   # labels available -> we can measure real degradation
            out["perf"] = self._score(new_df)
            out["loss_ratio"] = out["perf"]["log_loss"] / self.baseline_["log_loss"]
            out["concept_drift"] = out["loss_ratio"] >= self.loss_alert
        out["needs_update"] = out["covariate_drift"] or out["concept_drift"]
        return out

    def update(self, new_df: pd.DataFrame, force: bool = False) -> dict:
        """Feed a fresh labelled batch. Retrains only if drift is detected, and only
        swaps models if the challenger beats the champion on unseen recent data."""
        new_df = new_df.reset_index(drop=True)
        report = self.check_drift(new_df)
        if not (force or report["needs_update"]):
            self._log("kept", reason="no drift", max_psi=report["max_psi"])
            return {"action": "kept (no drift)", "report": report}

        fit_new, hold_new = train_test_split(
            new_df, test_size=0.3, stratify=new_df[TARGET], random_state=self.seed)
        combined = pd.concat([self.history_, fit_new], ignore_index=True).tail(self.window)
        age = np.arange(len(combined))[::-1]               # newest row has age 0
        weights = 0.5 ** (age / self.half_life)            # recent data counts more

        challenger = copy.copy(self)
        (challenger.pipe_, challenger.calibrator_,
         challenger.threshold_, _) = self._train(combined, weights, params=self.best_params_)

        champ, chall = self._score(hold_new), challenger._score(hold_new)
        promote = chall["log_loss"] < champ["log_loss"]
        if promote:
            self.pipe_, self.calibrator_ = challenger.pipe_, challenger.calibrator_
            self.threshold_ = challenger.threshold_
            self.baseline_ = chall
            self.version_ += 1
        self.history_ = pd.concat([self.history_, new_df], ignore_index=True).tail(self.window)
        if promote:
            self.reference_ = self.history_[FEATURES].tail(self.half_life).copy()
        self._log("promoted" if promote else "rejected", champion_loss=champ["log_loss"],
                  challenger_loss=chall["log_loss"], version=self.version_)
        return {
            "action": f"promoted -> v{self.version_}" if promote else "challenger rejected",
            "champion_on_fresh": champ, "challenger_on_fresh": chall, "report": report,
        }

    # ---- persistence ------------------------------------------------------
    def _log(self, event: str, **info):
        self.log_.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **info})

    def save(self, name: str = "accident_model.joblib") -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        versioned = self.model_dir / f"accident_model_v{self.version_}.joblib"
        joblib.dump(self, versioned)
        latest = self.model_dir / name
        shutil.copy2(versioned, latest)
        return latest

    @staticmethod
    def load(path="models/accident_model.joblib") -> "AdaptiveAccidentModel":
        return joblib.load(path)
