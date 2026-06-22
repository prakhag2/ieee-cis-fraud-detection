"""
Feature audit — why this feature set, and is each member earning its place?
  (1) LightGBM gain + split importance over features.py:COLS
  (2) Leave-one-out ablation: drop each feature, retrain, measure test PR-AUC change
Same leakage-safe builder and predefined split as static_baseline.py. This is the analysis that
flagged dev_fr as harmful and drove the 15 -> 14 trim.
"""
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
import features as F
from pipeline import log, load_static, fit_early_stop

tr, va, te = load_static(); T = F.TARGET
Xtr, (Xva, Xte) = F.build(tr, [va, te])
ytr, yva, yte = tr[T].values, va[T].values, te[T].values

def test_prauc(m, cols=None):
    X = Xte if cols is None else Xte[cols]
    return average_precision_score(yte, m.predict_proba(X)[:, 1])

# --- full model: importance ---
m = fit_early_stop(Xtr, ytr, Xva, yva)
full_pr = test_prauc(m)
full_roc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
log(f"FULL model ({len(F.COLS)} feats): test PR-AUC={full_pr:.4f} ROC={full_roc:.4f} "
    f"({m.best_iteration_} trees)\n")

gain = pd.Series(m.booster_.feature_importance('gain'), index=F.COLS)
split = pd.Series(m.booster_.feature_importance('split'), index=F.COLS)
imp = pd.DataFrame({'gain%': gain / gain.sum() * 100, 'splits': split}).sort_values('gain%', ascending=False)
log("=== LightGBM IMPORTANCE (sorted by gain share) ===")
log(f"{'feature':<30}{'gain%':>8}{'splits':>8}")
for name, r in imp.iterrows():
    log(f"{name:<30}{r['gain%']:>7.1f}%{int(r['splits']):>8d}")

# --- leave-one-out ablation ---
log("\n=== LEAVE-ONE-OUT ABLATION (drop 1 feature, retrain) ===")
log(f"{'dropped feature':<30}{'test PR-AUC':>12}{'delta':>9}")
rows = []
for name in F.COLS:
    cols = [c for c in F.COLS if c != name]
    pr = test_prauc(fit_early_stop(Xtr, ytr, Xva, yva, cols=cols), cols=cols)
    rows.append((name, pr, pr - full_pr))
for name, pr, d in sorted(rows, key=lambda x: x[2]):  # most-harmful-to-drop first
    log(f"{name:<30}{pr:>12.4f}{d:>+9.4f}")
log(f"\n(baseline full PR-AUC={full_pr:.4f}; negative delta = feature was helping)")
