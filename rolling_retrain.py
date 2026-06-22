"""
ROLLING-WINDOW retraining, leakage-safe. Walk the timeline in 2-week steps: at each step
train on prior data, predict the next unseen 2 weeks, roll forward. Two schemes on the SAME
future chunks:
  FORGET : train on prior 90 days only (rolling 3-month window)
  EXPAND : train on ALL data up to now (never forget)
Compared against the static baseline (static_baseline.py: PR-AUC 0.535, P@50 58% on test_v2).
"""
import numpy as np
from sklearn.metrics import average_precision_score
import features as F
from pipeline import log, load_timeline, rolling_steps, fit_predict, precision_at_recall as patr

WIN, STEP = 90, 14
df = load_timeline(); T = F.TARGET
steps = rolling_steps(df, WIN, STEP)
log(f"loaded {len(df)} txns, days {df['day'].min():.0f}-{df['day'].max():.0f}")
log(f"=== train prior {WIN}d, predict next {STEP}d, roll {STEP}d. {len(steps)} chunks ===")
log(f"{'pred-days':>14} {'n_te':>6} {'fr%':>4} | {'FORGET(90d)':>26} | {'EXPAND(all)':>26}")

def metrics(y, p):
    return average_precision_score(y, p), patr(y, p, .5), patr(y, p, .8)

forget, expand = [], []
for t in steps:
    te = df[(df.day >= t) & (df.day < t + STEP)]
    if te[T].sum() < 10: continue
    y = te[T].values
    f = metrics(y, fit_predict(df[(df.day >= t - WIN) & (df.day < t)], te))  # forget: last 90d
    e = metrics(y, fit_predict(df[df.day < t], te))                          # expand: all prior
    forget.append(f); expand.append(e)
    log(f"{t:>5.0f}-{t+STEP:<8.0f} {len(te):>6d} {y.mean()*100:>3.1f} | "
        f"PRAUC={f[0]:.3f} P@50={f[1]:>3.0f}% P@80={f[2]:>3.0f}% | "
        f"PRAUC={e[0]:.3f} P@50={e[1]:>3.0f}% P@80={e[2]:>3.0f}%")

f, e = np.array(forget), np.array(expand)
log("=" * 70)
log(f"AVG FORGET(90d): PR-AUC={f[:,0].mean():.3f} P@50={f[:,1].mean():.0f}% P@80={f[:,2].mean():.0f}%")
log(f"AVG EXPAND     : PR-AUC={e[:,0].mean():.3f} P@50={e[:,1].mean():.0f}% P@80={e[:,2].mean():.0f}%")
log(f"BASELINE static: PR-AUC=0.535 P@50=58% P@80=13%")
