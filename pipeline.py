"""
Shared plumbing for the fraud pipeline: data loading, the rolling-window walk, model
fit/predict, and metrics. Keeps each experiment script thin so it reads as just its
experiment. Feature construction lives in features.py; this module never touches labels
except by delegating to features.build (which is leakage-safe).
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import precision_recall_curve
import features as F

DATA_DIR = "Revised Dataset"
SPLITS = ("train_v2", "validation_v2", "test_v2")
# static-model overrides on top of F.lgb_params(): smaller LR + more trees + early stopping
TUNED = dict(learning_rate=0.03, n_estimators=3000)


def log(msg):
    print(msg); sys.stdout.flush()


def load_static():
    """The predefined (train, validation, test) split, each with derived keys added."""
    return tuple(F.add_keys(pd.read_csv(f"{DATA_DIR}/{n}.csv")) for n in SPLITS)


def load_timeline():
    """All three splits concatenated and sorted by time — one continuous transaction stream."""
    df = pd.concat([pd.read_csv(f"{DATA_DIR}/{n}.csv") for n in SPLITS], ignore_index=True)
    return F.add_keys(df.sort_values(F.TS).reset_index(drop=True))


def rolling_steps(df, win=90, step=14):
    """Start days for each prediction window: train on [t-win, t), predict [t, t+step)."""
    dmin, dmax = df['day'].min(), df['day'].max()
    steps, t = [], dmin + win
    while t + step <= dmax + 0.5:
        steps.append(t); t += step
    return steps


def precision_at_recall(y, p, recall):
    """Precision (%) at the point on the PR curve closest to the target recall."""
    prec, rec, _ = precision_recall_curve(y, p)
    return prec[np.argmin(np.abs(rec - recall))] * 100


def fit_predict(train_df, eval_df):
    """Train one LightGBM on train_df, return fraud probabilities for eval_df.
    Plain fit (no early stopping) — used by the rolling experiments."""
    Xtr, (Xte,) = F.build(train_df, [eval_df])
    m = lgb.LGBMClassifier(**F.lgb_params())
    m.fit(Xtr, train_df[F.TARGET].values)
    return m.predict_proba(Xte)[:, 1]


def fit_early_stop(Xtr, ytr, Xva, yva, cols=None):
    """Train the static reference model: tuned params, early-stopped on the validation set.
    Pass cols to fit on a feature subset (used by the leave-one-out ablation)."""
    if cols is not None:
        Xtr, Xva = Xtr[cols], Xva[cols]
    m = lgb.LGBMClassifier(**dict(F.lgb_params(), **TUNED))
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return m
