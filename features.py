"""
Shared 14-feature builder for the fraud model. Leakage-safe: every statistic is fit on
the training frame only, then mapped onto any eval frame. The customer fraud-rate uses
out-of-fold encoding on train so a training row never sees its own label.

Used by:  static_baseline.py  (static train/val/test)
          rolling_retrain.py  (rolling-window retraining)
"""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

TARGET='is_fraud'; CARD='card_number_hash1'; TS='txn_timestamp_sec'

# the 14 features, in order. dev_fr (device fraud-rate) was dropped: an audit
# (feature_audit.py + rolling test) showed it consistently HURTS held-out PR-AUC
# (+0.013 static, +0.006 rolling when removed) — re-encoded device rates drift faster
# than they inform. All other features earn their place or are group-coverage free-riders.
COLS=['cust_fr','card2_fr','v_card_addr_all_amount_258','count_txns_on_card',
      'count_txns_card_addr_email','card_m','dev_per_card','uid_s','dscard_norm',
      'em_fr','card_z','zip_z','card_s','count_txns_email_domain']

def add_keys(df):
    """Add derived keys (day, reg_card, uid) needed for feature construction. In place-safe copy."""
    df=df.copy()
    df['day']=df[TS]/86400.0
    df['reg_card']=np.floor(df['day']-df['days_since_first_txn_card'])
    df['uid']=df[CARD].astype(str)+'_'+df['billing_zip_region'].astype(str)+'_'+df['reg_card'].astype(str)
    return df

# label-derived fraud-rate features: (output column, grouping key). These need OOF on
# train so a training row never sees its own label.
FRATE=[('cust_fr','uid'),('card2_fr','card_number_hash2'),
       ('em_fr','purchaser_email_domain')]

def _smooth_rate(sum_, count_, gp, SM=20.0):
    return (sum_+SM*gp)/(count_+SM)

def build(train_df, eval_frames):
    """
    Fit all 14 features on train_df ONLY; return (X_train, [X_eval for each eval frame]).
    train_df and eval_frames must already have keys (call add_keys first).
    Leakage-safe: non-label stats fit on train and mapped to eval; the 3 fraud-rate
    features use full-train encoding for EVAL rows and out-of-fold (OOF) encoding for
    TRAIN rows, so no training row ever sees its own label.
    """
    gp=train_df[TARGET].mean()
    a_card=train_df.groupby(CARD)['txn_amount_usd'].agg(['mean','std'])
    a_zip_s=train_df.groupby('billing_zip_region')['txn_amount_usd'].std()
    a_zip_m=train_df.groupby('billing_zip_region')['txn_amount_usd'].mean()
    a_uid_s=train_df.groupby('uid')['txn_amount_usd'].std()
    devpc=train_df.groupby(CARD)['device_model'].nunique()
    # full-train fraud-rate encoders (used for EVAL frames): key value -> smoothed rate
    enc={}
    for col,key in FRATE:
        g=train_df.groupby(key)[TARGET].agg(['sum','count'])
        enc[col]=_smooth_rate(g['sum'],g['count'],gp)

    def feats(d, rate_src):
        """Assemble the feature frame for rows d. rate_src[col] -> Series of fraud rates
        for the 3 fraud-rate features (OOF on train rows); empty -> use full-train enc."""
        out={}; cm=d[CARD].map(a_card['mean']); cs=d[CARD].map(a_card['std'])
        zs=d['billing_zip_region'].map(a_zip_s); zm=d['billing_zip_region'].map(a_zip_m)
        out['v_card_addr_all_amount_258']=d['v_card_addr_all_amount_258'].values
        out['count_txns_on_card']=d['count_txns_on_card'].values
        out['count_txns_card_addr_email']=d['count_txns_card_addr_email'].values
        out['count_txns_email_domain']=d['count_txns_email_domain'].values
        out['card_m']=cm.values; out['card_s']=cs.values
        out['dev_per_card']=d[CARD].map(devpc).fillna(0).values
        out['uid_s']=d['uid'].map(a_uid_s).values
        out['dscard_norm']=(d['days_since_first_txn_card']-d['day']).values
        out['card_z']=((d['txn_amount_usd']-cm)/cs.replace(0,np.nan)).values
        out['zip_z']=((d['txn_amount_usd']-zm)/zs.replace(0,np.nan)).values
        for col,key in FRATE:
            out[col]=rate_src[col].values if col in rate_src else d[key].map(enc[col]).fillna(gp).values
        return pd.DataFrame(out)[COLS]

    # ---- TRAIN rows: OOF for all 3 fraud-rate features (no row sees its own label) ----
    oof={col: np.full(len(train_df),gp) for col,_ in FRATE}
    yv=train_df[TARGET].values
    for trn,vld in KFold(5,shuffle=True,random_state=0).split(np.arange(len(train_df))):
        for col,key in FRATE:
            kv=train_df[key].values
            ss=pd.DataFrame({'k':kv[trn],'y':yv[trn]}).groupby('k')['y'].agg(['sum','count'])
            e=_smooth_rate(ss['sum'],ss['count'],gp)
            oof[col][vld]=pd.Series(kv[vld]).map(e).fillna(gp).values
    Xtr=feats(train_df, rate_src={col: pd.Series(oof[col]) for col,_ in FRATE})
    # ---- EVAL rows: full-train encoding (clean by construction) ----
    Xevs=[feats(e, rate_src={}) for e in eval_frames]
    return Xtr, Xevs


def lgb_params():
    return dict(objective='binary',metric='auc',learning_rate=0.05,num_leaves=64,
                min_child_samples=60,subsample=0.8,colsample_bytree=0.9,reg_lambda=8,
                n_estimators=900,n_jobs=-1,verbosity=-1)
