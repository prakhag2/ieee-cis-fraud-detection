"""
BASELINE (THE reference): one static model on the predefined split. Train on train_v2,
early-stop on validation_v2, evaluate on UNSEEN test_v2. Leakage-safe (features fit on
train only). Compare against the rolling models in rolling_retrain.py.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import features as F
from pipeline import log, load_static, fit_early_stop, precision_at_recall as patr

tr, va, te = load_static()
T = F.TARGET
log(f"train={len(tr)} ({tr[T].mean()*100:.2f}%) | val={len(va)} ({va[T].mean()*100:.2f}%) | "
    f"test={len(te)} ({te[T].mean()*100:.2f}%)")

Xtr, (Xva, Xte) = F.build(tr, [va, te])
m = fit_early_stop(Xtr, tr[T].values, Xva, va[T].values)
log(f"trained: {m.best_iteration_} trees (early-stopped on val)")

for nm, X, d in [('VAL', Xva, va), ('TEST(unseen)', Xte, te)]:
    y = d[T].values; p = m.predict_proba(X)[:, 1]
    log(f"[{nm:<12}] PR-AUC={average_precision_score(y,p):.4f} ROC={roc_auc_score(y,p):.4f} | "
        f"P@50={patr(y,p,.5):.0f}% P@60={patr(y,p,.6):.0f}% P@70={patr(y,p,.7):.0f}% "
        f"P@80={patr(y,p,.8):.0f}% P@90={patr(y,p,.9):.0f}%")
