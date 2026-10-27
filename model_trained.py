import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_trained import (CATEGORICAL, FEATURES, NUMERIC, TARGET,
                            AdaptiveAccidentModel, generate_data)

train_df = generate_data(6000, seed=42)
test_df = generate_data(2000, seed=99)

# ---- 1. original approach as baseline -------------------------------------
base = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC)])),
    ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                       max_depth=4, random_state=42)),
]).fit(train_df[FEATURES], train_df[TARGET])
p = base.predict_proba(test_df[FEATURES])[:, 1]
base_scores = {
    "auc": roc_auc_score(test_df[TARGET], p),
    "log_loss": log_loss(test_df[TARGET], p),
    "brier": brier_score_loss(test_df[TARGET], p),
    "accuracy": accuracy_score(test_df[TARGET], p >= 0.5),
    "f1": f1_score(test_df[TARGET], p >= 0.5),
}

# ---- 2. adaptive model -----------------------------------------------------
t0 = time.time()
model = AdaptiveAccidentModel().fit(train_df)
print(f"Trained + tuned in {time.time() - t0:.0f}s | best params: {model.best_params_}")
new_scores = model.evaluate(test_df)
model.save()   # keeps v1 on disk, so the later update leaves a rollback point

print("\n=== Same data, same test set (lower log_loss/brier = better, higher auc = better) ===")
print(pd.DataFrame({"original GBC": base_scores, "adaptive model": new_scores}).round(4).to_string())

print("\n=== What drives risk? (permutation importance) ===")
print(model.explain(test_df).round(4).to_string())

# ---- 3. the world changes --------------------------------------------------
drift_test = generate_data(2000, seed=8, drift=True)    # fresh data from the NEW world
new_batch = generate_data(2000, seed=7, drift=True)     # what we "observed" this month

before = model.evaluate(drift_test)
res = model.update(new_batch)
after = model.evaluate(drift_test)

print("\n=== Drift report on the new batch (input PSI: <0.1 stable, >0.2 alert) ===")
print(pd.Series(res["report"]["psi"]).round(3).to_string())
print(f"log-loss vs baseline: x{res['report']['loss_ratio']:.2f}  "
      f"(retrain threshold x{model.loss_alert})")
print(f"covariate drift: {res['report']['covariate_drift']} | "
      f"concept drift: {res['report']['concept_drift']}")
print(f"\nDecision: {res['action']}")
if "champion_on_fresh" in res:
    print(f"  old model on held-out fresh data : log_loss={res['champion_on_fresh']['log_loss']:.4f}")
    print(f"  retrained challenger             : log_loss={res['challenger_on_fresh']['log_loss']:.4f}")

print("\n=== New-world test set: before vs after adapting ===")
print(pd.DataFrame({"before update": before, "after update": after}).to_string())
print("\n(Old-world test set after adapting - some drop is expected, the world changed:)")
print(model.evaluate(test_df))

# ---- 4. use it ---------------------------------------------------------------
scenarios = pd.DataFrame([
    ("Clear", "Dry", "Day", 45, 35, 9.0),
    ("Rain", "Wet", "Night", 55, 20, 3.0),
    ("Fog", "Icy", "Night", 65, 70, 0.8),
    ("Snow", "Icy", "Day", 35, 45, 2.0),
], columns=FEATURES)
print("\n=== Risk predictions ===")
print(pd.concat([scenarios, model.predict_risk(scenarios)], axis=1).to_string(index=False))

path = model.save()
print(f"\nSaved -> {path}  (version {model.version_}; older versions kept in models/)")
