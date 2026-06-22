"""
Decisioning tables for the rolling models. Two views:
  1. PER-CHUNK BANDS (FORGET and EXPAND): for each 14-day prediction window, how many txns /
     frauds land in each operating band (auto-block / step-up-verify / allow), with fraud
     caught and legit customers affected. Shows how the bands behave over time.
  2. POOLED ANALYSIS (FORGET): all out-of-sample predictions concatenated, then the score
     sweep / operating curve / band summary used to CHOOSE the cutoffs.
Same leakage-free rolling setup as rolling_retrain.py. FORGET=prior 90d, EXPAND=all prior.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve
import features as F
from pipeline import log, load_timeline, rolling_steps, fit_predict

WIN, STEP = 90, 14
BLOCK, VERIFY = 0.80, 0.30  # from the curve: block where prec~90%, verify down to prec~70%
df = load_timeline(); T = F.TARGET
steps = rolling_steps(df, WIN, STEP)
span_days = df['day'].max() - (df['day'].min() + WIN)

# one pass over chunks: train+predict both schemes on each future window
chunks = []  # (label, y, {'FORGET': p, 'EXPAND': p})
for t in steps:
    te = df[(df.day >= t) & (df.day < t + STEP)]
    if te[T].sum() < 10: continue
    y = te[T].values
    pf = fit_predict(df[(df.day >= t - WIN) & (df.day < t)], te)  # forget: last 90d
    pe = fit_predict(df[df.day < t], te)                          # expand: all prior
    chunks.append((f"{t:.0f}-{t+STEP:.0f}", y, {'FORGET': pf, 'EXPAND': pe}))


def bands(y, p):
    """Counts in the three bands: returns (n, nf, blk(txn,caught,fp), vfy(txn,caught,fp), miss, recall%)."""
    blk = p >= BLOCK
    vfy = (p >= VERIFY) & (p < BLOCK)
    bt, bc = int(blk.sum()), int(y[blk].sum())
    vt, vc = int(vfy.sum()), int(y[vfy].sum())
    nf = int(y.sum())
    miss = nf - bc - vc
    recall = (bc + vc) / nf * 100 if nf else 0
    return len(y), nf, (bt, bc, bt - bc), (vt, vc, vt - vc), miss, recall


# --- view 1: per-chunk bands, one table per scheme ---
for scheme in ('FORGET', 'EXPAND'):
    log(f"\n=== PER-CHUNK BANDS — ROLLING {scheme} "
        f"(BLOCK>={BLOCK}, VERIFY {VERIFY}-{BLOCK}, ALLOW<{VERIFY}) ===")
    hdr = (f"{'pred-days':>12}{'txns':>7}{'fraud':>6} | {'BLK txn':>8}{'caught':>7}{'fp':>5}"
           f" | {'VFY txn':>8}{'caught':>7}{'fp':>5} | {'allow miss':>11}{'recall':>8}")
    log(hdr); log('-' * len(hdr))
    tot = np.zeros(7, dtype=int)  # n, nf, bt, bc, vt, vc, miss
    for label, y, preds in chunks:
        n, nf, (bt, bc, bfp), (vt, vc, vfp), miss, rec = bands(y, preds[scheme])
        tot += np.array([n, nf, bt, bc, vt, vc, miss])
        log(f"{label:>12}{n:>7d}{nf:>6d} | {bt:>8d}{bc:>7d}{bfp:>5d}"
            f" | {vt:>8d}{vc:>7d}{vfp:>5d} | {miss:>11d}{rec:>7.0f}%")
    n, nf, bt, bc, vt, vc, miss = tot
    rec = (bc + vc) / nf * 100 if nf else 0
    log('-' * len(hdr))
    log(f"{'TOTAL':>12}{n:>7d}{nf:>6d} | {bt:>8d}{bc:>7d}{bt-bc:>5d}"
        f" | {vt:>8d}{vc:>7d}{vt-vc:>5d} | {miss:>11d}{rec:>7.0f}%")
    bp = bc / bt * 100 if bt else 0; vp = vc / vt * 100 if vt else 0
    log(f"  block precision={bp:.0f}%  verify-band precision={vp:.0f}%  "
        f"act-on recall={rec:.0f}%  allow lets {(n-bt-vt)/n*100:.1f}% of txns through")

# ===================== pooled analysis on FORGET (cutoff selection) =====================
p = np.concatenate([preds['FORGET'] for _, _, preds in chunks])
y = np.concatenate([yc for _, yc, _ in chunks])
n, nf = len(y), int(y.sum())
log(f"\n\npooled out-of-sample (FORGET): {n} txns, {nf} fraud ({y.mean()*100:.2f}%)\n")

# --- table A: act on everything above a score threshold ---
log("=== ACT ON ALL TXNS WITH score >= threshold ===")
log(f"{'thresh':>7} {'flagged':>8} {'flagged%':>9} {'recall':>7} {'precision':>10} {'false+/day*':>11}")
for thr in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    sel = p >= thr; fl = int(sel.sum()); tp = int(y[sel].sum()); fp = fl - tp
    rec = tp / nf * 100; prec = tp / fl * 100 if fl else 0
    log(f"{thr:>7.2f} {fl:>8d} {fl/n*100:>8.1f}% {rec:>6.1f}% {prec:>9.1f}% {fp/max(span_days,1):>11.0f}")

# --- table B: precision at each recall milestone (the operating curve) ---
log("\n=== PRECISION AT EACH RECALL LEVEL (the operating curve) ===")
prec, rec, thr = precision_recall_curve(y, p)
log(f"{'recall':>7} {'precision':>10} {'~score cut':>11} {'%txns flagged':>14}")
for target in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    i = np.argmin(np.abs(rec - target))
    cut = thr[min(i, len(thr) - 1)]
    log(f"{rec[i]*100:>6.0f}% {prec[i]*100:>9.1f}% {cut:>11.3f} {(p>=cut).mean()*100:>13.1f}%")

# --- table C: the proposed 3 bands ---
log("\n=== PROPOSED DECISION BANDS ===")
def cumulative(thr):  # act on all score >= thr
    sel = p >= thr; fl = int(sel.sum()); tp = int(y[sel].sum())
    return dict(recall=tp / nf, prec=(tp / fl if fl else 0), vol=fl / n, tp=tp, fl=fl)
b, v = cumulative(BLOCK), cumulative(VERIFY)
band_fl = v['fl'] - b['fl']; band_tp = v['tp'] - b['tp']  # step-up band = VERIFY <= score < BLOCK
band_prec = band_tp / band_fl if band_fl else 0
log(f"AUTO-BLOCK    : score>={BLOCK:.2f}            recall={b['recall']*100:4.0f}%  precision={b['prec']*100:4.0f}%  vol={b['vol']*100:4.1f}% of txns")
log(f"STEP-UP VERIFY: {VERIFY:.2f}<=score<{BLOCK:.2f}      +{(v['recall']-b['recall'])*100:.0f}% recall (cum {v['recall']*100:.0f}%)  band precision={band_prec*100:4.0f}%  vol={band_fl/n*100:4.1f}% of txns")
log(f"ALLOW+MONITOR : score<{VERIFY:.2f}             allows {(1-v['vol'])*100:.1f}% of txns; misses {(1-v['recall'])*100:.0f}% of fraud -> account-monitoring + chargeback recovery + retrain")
log("\n* false+/day illustrative (pooled period spans ~%d days)" % span_days)
