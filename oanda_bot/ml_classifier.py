#!/usr/bin/env python3
"""ML trade-selection classifier on the MS-ORB feature dataset.

Predicts P(win @ 3:1 RR) from PRE-TRADE features only (no outcome leakage),
using walk-forward time-series validation (no shuffling — critical for trading).
Then uses the model to SELECT the highest-probability trades and measures the
resulting out-of-sample win rate / profit factor / expectancy vs. taking all
trades. The goal is the optimal selection model, validated honestly.

Label: win = (y_atr15 == 1), i.e. price hit +3R before -1R with a 1.5xATR stop.
Run:   python ml_classifier.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

# round-trip cost (spread+slippage) in PRICE units per instrument (estimates)
COST_PRICE = {"XAU_USD": 0.35, "XAG_USD": 0.02, "NAS100_USD": 1.5, "SPX500_USD": 0.5,
              "US30_USD": 2.5, "JP225_USD": 10.0, "BCO_USD": 0.04, "EUR_USD": 0.00008,
              "GBP_USD": 0.00010, "USD_JPY": 0.010, "AUD_USD": 0.00008, "USD_CAD": 0.00010}

CSV = "cache/research_signals.csv"
# round-trip cost as a fraction of the 1.5xATR initial risk (estimate, liquid M5)
COST_R = 0.05

# PRE-TRADE features only. Everything outcome/post-entry/price-scale is excluded.
CAT = ["instrument", "session", "direction"]
NUM = ["hour", "dow", "seq", "range_atr", "range_pctile", "atr_pct", "body_ratio",
       "vol_ratio", "adx", "cvd_aligned", "cvd_norm", "ramp", "trend_fast",
       "trend_htf", "dist_beyond_r", "dist_beyond_atr", "rvol", "news", "clean"]
LEAK = {"y_swing3d", "y_swing5d", "y_candle", "maxR", "y_range", "y_atr10", "y_atr15",
        "y_atr50", "y_atr75", "y_atr125", "y_atr200", "risk_price", "risk_atr",
        "r_k15", "r_k20", "r_k25", "r_k30", "r_atrstop", "r_runner", "atr_price",
        "r_a75_k20", "r_a75_k30", "r_a100_k20", "r_a100_k30", "r_a150_k20", "r_a150_k30", "date"}


def load():
    df = pd.read_csv(CSV)
    df = df[df["y_atr15"].isin([0, 1, -1])].copy()
    df["win"] = (df["y_atr15"] == 1).astype(int)       # win @ 3:1 (1.5xATR stop)
    df = df.sort_values("date").reset_index(drop=True)  # TIME ORDER
    assert not (LEAK & set(NUM + CAT)), "leakage in feature list!"
    return df


def featurize(df):
    X = pd.get_dummies(df[CAT + NUM], columns=CAT, drop_first=False)
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(X.median(numeric_only=True))


def trade_stats(win, df_sub, sel_mask):
    """WR@3R, PF, gross & net expectancy on selected trades."""
    w = win[sel_mask]
    n = len(w)
    if n == 0:
        return None
    wr = w.mean()
    pf = (wr * 3) / ((1 - wr) * 1) if wr < 1 else 99
    exp = wr * 3 - (1 - wr) * 1
    exp_net = exp - COST_R
    return dict(n=n, wr=wr, pf=pf, exp=exp, exp_net=exp_net)


def walk_forward(df, X, y, n_folds=5):
    """Expanding-window time-series CV: predict each later block from earlier ones."""
    idx = np.arange(len(df))
    bounds = np.linspace(0, len(df), n_folds + 1).astype(int)
    oof = np.full(len(df), np.nan)
    aucs = []
    for i in range(1, n_folds):
        tr = idx[bounds[0]:bounds[i]]
        te = idx[bounds[i]:bounds[i + 1]]
        if len(te) == 0 or y[tr].nunique() < 2:
            continue
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                       min_samples_leaf=60, subsample=0.8, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        try:
            aucs.append(roc_auc_score(y.iloc[te], p))
        except ValueError:
            pass
    return oof, aucs


def main():
    df = load()
    X = featurize(df)
    y = df["win"]
    win = y.values
    print(f"Loaded {len(df)} signals | win@3R base rate = {win.mean()*100:.1f}% "
          f"| {X.shape[1]} features | {df['instrument'].nunique()} instruments")

    # ---- walk-forward out-of-sample predictions ----
    oof, aucs = walk_forward(df, X, y, n_folds=5)
    mask = ~np.isnan(oof)
    print(f"\nWalk-forward OOS: {mask.sum()} predictions | mean fold AUC = {np.mean(aucs):.3f} "
          f"(0.5 = no skill)")
    base = trade_stats(win, df, mask)
    print(f"  BASELINE (take all OOS trades): n={base['n']}  WR={base['wr']*100:.1f}%  "
          f"PF={base['pf']:.2f}  expR={base['exp']:+.3f}  net={base['exp_net']:+.3f}")

    # ---- trade selection by model probability ----
    print("\n  MODEL-SELECTED TRADES (keep only high P(win)):")
    print("    selection            n     WR@3R    PF     expR    netR")
    p_oof = oof[mask]
    win_oof = win[mask]
    sub = df[mask]
    for q in (0.5, 0.6, 0.7, 0.8, 0.9):
        thr = np.quantile(p_oof, q)
        sm = p_oof >= thr
        s = trade_stats(win_oof, sub, sm)
        if s:
            print(f"    top-{int((1-q)*100):>2d}% (P>={thr:.2f})   {s['n']:5d}  {s['wr']*100:5.1f}%  "
                  f"{s['pf']:5.2f}  {s['exp']:+.3f}  {s['exp_net']:+.3f}")

    # ---- feature importance (full-data fit, for interpretation only) ----
    m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   min_samples_leaf=60, subsample=0.8, random_state=0)
    m.fit(X, y)
    imp = sorted(zip(X.columns, m.feature_importances_), key=lambda t: -t[1])[:12]
    print("\n  TOP FEATURES (gradient boosting importance):")
    for name, val in imp:
        print(f"    {name:22s} {val:.3f}")

    runner_regression(df, X)


def runner_regression(df, X):
    """Predict the no-target runner's realized R (1.0xATR stop, 2.0 trail) and
    select trades to maximize OOS net total R — the 'optimal trading result'."""
    print("\n" + "=" * 70)
    print("  ML REGRESSION — predict runner realized R, select best trades (NET)")
    print("=" * 70)
    d = df[pd.to_numeric(df["r_a100_k20"], errors="coerce").notna()
           & pd.to_numeric(df["atr_price"], errors="coerce").notna()].copy()
    Xr = X.loc[d.index]
    r = pd.to_numeric(d["r_a100_k20"], errors="coerce").values          # gross realized R
    atr = pd.to_numeric(d["atr_price"], errors="coerce").values
    cost = d["instrument"].map(COST_PRICE).fillna(0).values
    net = r - cost / np.maximum(atr, 1e-9)                              # net R (cost / 1.0xATR risk)

    idx = np.arange(len(d))
    bounds = np.linspace(0, len(d), 6).astype(int)
    pred = np.full(len(d), np.nan)
    for i in range(1, 5):
        tr, te = idx[:bounds[i]], idx[bounds[i]:bounds[i + 1]]
        if len(te) == 0:
            continue
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      min_samples_leaf=60, subsample=0.8, random_state=0)
        m.fit(Xr.iloc[tr], r[tr])
        pred[te] = m.predict(Xr.iloc[te])
    mask = ~np.isnan(pred)
    p, nm = pred[mask], net[mask]
    def stats(sel):
        x = nm[sel]; n = len(x)
        if n == 0: return None
        wins = x[x > 0]; losses = x[x <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else 99
        sd = x.std() or 1
        return n, (x > 0).mean(), x.mean(), x.sum(), pf, x.mean()/sd*np.sqrt(n)
    b = stats(np.ones(len(nm), bool))
    print(f"  ALL runner trades (OOS, net): n={b[0]} win={b[1]*100:.0f}% expR={b[2]:+.3f} "
          f"totalR={b[3]:+.0f} PF={b[4]:.2f} Sharpe~{b[5]:.2f}")
    print("    selection           n     win%   expR    totalR   PF    Sharpe")
    for q in (0.5, 0.7, 0.8, 0.9):
        s = stats(p >= np.quantile(p, q))
        if s:
            print(f"    top-{int((1-q)*100):>2d}% by pred-R  {s[0]:5d}  {s[1]*100:4.0f}%  {s[2]:+.3f}  "
                  f"{s[3]:+6.0f}  {s[4]:.2f}  {s[5]:.2f}")

    # focused, cost-viable universe (indices+metals), Tokyo session: ML vs take-all
    print("\n  COST-VIABLE FOCUS (indices+metals, Tokyo): does ML selection beat take-all?")
    good = {"XAU_USD", "XAG_USD", "NAS100_USD", "SPX500_USD", "US30_USD", "JP225_USD", "BCO_USD"}
    dm = d[mask].reset_index(drop=True)
    pm, nmm = p, nm
    foc = ((dm["session"] == "TOKYO") & dm["instrument"].isin(good)).values
    if foc.sum() > 50:
        s_all = stats(foc)
        print(f"    take-all (Tokyo+viable):  n={s_all[0]:4d} win={s_all[1]*100:.0f}% "
              f"expR={s_all[2]:+.3f} totalR={s_all[3]:+.0f} PF={s_all[4]:.2f}")
        thr = np.quantile(pm[foc], 0.5)
        sel = foc & (pm >= thr)
        s_ml = stats(sel)
        if s_ml:
            print(f"    ML top-50% within focus:  n={s_ml[0]:4d} win={s_ml[1]*100:.0f}% "
                  f"expR={s_ml[2]:+.3f} totalR={s_ml[3]:+.0f} PF={s_ml[4]:.2f}")


if __name__ == "__main__":
    main()
