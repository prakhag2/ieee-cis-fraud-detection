# Fraud Detection -- Minimal Pipeline

Kaggle IEEE-CIS fraud dataset (renamed columns), e-commerce card-not-present transactions,
~3.5% fraud. This directory holds the final, cleaned, leakage-free pipeline.

## Files
- `features.py` -- shared 14-feature builder (leakage-safe). Single source of truth.
- `pipeline.py` -- shared plumbing: data loading, the rolling-window walk, model fit/predict, metrics.
- `static_baseline.py` -- static model: train on `train_v2`, validate on `validation_v2`, test on unseen `test_v2`.
- `rolling_retrain.py` -- rolling retraining: every 2 weeks, predict next 2 weeks. Compares
  FORGET (last 90 days) vs EXPAND (all history), against the static baseline.
- `decision_bands.py` -- auto-block / step-up-verify / allow operating bands: per-chunk breakdown
  (FORGET & EXPAND, txns, fraud, caught, false-positives, recall over time) + pooled cutoff-selection tables.
- `feature_audit.py` -- feature audit: LightGBM gain/split importance + leave-one-out ablation.
  This is what flagged `dev_fr` as harmful and drove the 15->14 trim.
- `Revised Dataset/{train_v2,validation_v2,test_v2}.csv` -- the data (temporal split, no overlap).

## Data
The CSVs are **not** included in this repo (Kaggle IEEE-CIS license -- not redistributed).
Obtain the IEEE-CIS Fraud Detection dataset from Kaggle, apply the column renaming this code
expects, and place the temporal split at `Revised Dataset/{train_v2,validation_v2,test_v2}.csv`.

## Run
Run from the repository root (scripts use the relative path `Revised Dataset/`):
```
python3 static_baseline.py
python3 rolling_retrain.py
python3 decision_bands.py
python3 feature_audit.py
```
Requires: pandas, numpy, lightgbm, scikit-learn.

## The 14 features (all in `features.py:COLS`)
Built from a synthetic **customer id** = `card1 + billing_zip + account-open-day`
(`account-open-day = floor(day - days_since_first_txn_card)`).

| Group | Features |
|---|---|
| Identity / guilt-by-association | `cust_fr`, `card2_fr`, `em_fr` (smoothed target-encoded fraud rates) |
| Behavioral deviation | `card_z`, `zip_z` (amount vs entity's normal) |
| Spend profile | `card_m`, `card_s`, `uid_s` |
| Ring / takeover | `dev_per_card` |
| Tenure | `dscard_norm` |
| Velocity / money-flow (causal passthrough) | `count_txns_on_card`, `count_txns_card_addr_email`, `count_txns_email_domain`, `v_card_addr_all_amount_258` |

> **Feature audit (was 15, now 14).** `dev_fr` (device fraud-rate) was dropped. A leave-one-out
> ablation + rolling test showed it consistently *hurt* held-out PR-AUC -- removing it gained
> **+0.013** static / **+0.006** rolling -- because re-encoded device rates drift faster than they
> inform. Importance is heavily concentrated: `cust_fr` (~63% gain) + `count_txns_on_card` carry
> most of the signal; the bottom ~5 features add little PR-AUC but firm up the high-precision
> operating point, so they're kept for the deployable band.

## Leakage controls (in `features.py:build`)
- All statistics fit on the **train frame only**; eval rows only look them up via `.map()`.
- The 3 fraud-rate features use **5-fold out-of-fold** encoding on train rows (no row sees its
  own label) and full-train encoding on eval rows.
- Time discipline lives in the callers: baseline uses the predefined splits; rolling uses
  strictly-past windows (`day < t`). No future data reaches training.

## Results (unseen test, leakage-free)
| Approach | PR-AUC | Precision @ 50% recall | Precision @ 80% recall |
|---|---|---|---|
| Baseline (static) | 0.535 | 58% | 13% |
| Rolling -- forget 90d | 0.662 | 86% | 23% |
| Rolling -- expand (no forget) | 0.670 | 85% | 24% |

## Takeaways
1. **Retraining every 2 weeks is the decisive lever** (58% -> 86% precision @ 50% recall). Fraud
   drifts; a static model decays. Forget(90d) ~ Expand, so the cheaper 90-day window is fine.
2. **High recall is data-limited** (~23% precision @ 80% recall regardless of model/features) --
   the hard half of fraud looks like legit traffic without IP / session / device-fingerprint /
   recipient-graph signals this dataset lacks.
3. **Deploy at ~50% recall** (auto-block tier, ~85% precision) and route the rest to step-up
   auth / manual review / chargeback recovery.
