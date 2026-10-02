"""Race-pace prediction from practice data - dataset builder + backtest.

Question: which observable practice signals best predict who is fastest on
race pace (we can't see fuel/engine, so we test what the stopwatch does see)?

Per driver per event:
  features (all as % deltas to the field best/reference that session):
    fp3_best_pct      - best FP3 lap vs session best (the "headline" lap)
    fp2_best_pct      - best FP2 lap vs session best
    fp2_lr_median_pct - FP2 long-run median vs best long-run median in field
    fp2_deg_slope     - s/lap drift within FP2 long runs (tyre deg proxy)
    fp2_lr_resid_std  - lap scatter around each long run's trend (clustering)
    fp2_lr_laps       - number of FP2 long-run laps (sample confidence)
  target:
    race_pace_pct     - median clean race lap vs field best median (lower=faster)

Long run = FP2 stint with >= MIN_STINT timed green-flag laps (in/out laps and
laps > 107% of the stint median dropped before fitting).

Backtest: chronological walk-forward - train on all earlier events, predict
each next event, score Spearman rank correlation vs actual race pace, plus
single-feature baselines (answers "which factor correlates best" out of
sample). Feature importance via permutation on the final model.

Usage:  python build_race_pace_model.py [--build] [--analyze]
        (no args = both; --analyze reuses data/race_pace_dataset.csv)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
CACHE = HERE / "data" / "fastf1_cache"
DATASET = HERE / "data" / "race_pace_dataset.csv"
SEASONS = [2023, 2024, 2025, 2026]
MIN_STINT = 5          # laps for a stint to count as a long run
QUICK_CUT = 1.07       # within-stint outlier cut vs stint median
MIN_TRAIN_EVENTS = 8   # events before walk-forward predictions start

FEATURES = ["fp3_best_pct", "fp2_best_pct", "fp2_lr_median_pct",
            "fp2_deg_slope", "fp2_lr_resid_std", "fp2_lr_laps"]


# ------------------------------------------------------------------ build ---

def lap_seconds(laps):
    return laps["LapTime"].dt.total_seconds()


def clean_green(laps):
    ok = laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    ok &= laps["TrackStatus"].astype(str) == "1"
    ok &= laps["LapTime"].notna()
    return laps[ok]


def best_quick_pct(laps):
    """Per-driver best lap as % delta to session best."""
    sec = clean_green(laps).groupby("Driver")["LapTime"].min().dt.total_seconds()
    return (sec / sec.min() - 1) * 100


def long_run_features(laps):
    """Per-driver long-run median %, deg slope, residual std, lap count."""
    rows = {}
    laps = clean_green(laps)
    for drv, dl in laps.groupby("Driver"):
        stints = []
        for _, stint in dl.groupby("Stint"):
            sec = lap_seconds(stint)
            sec = sec[sec <= sec.median() * QUICK_CUT]
            if len(sec) < MIN_STINT:
                continue
            x = np.arange(len(sec))
            slope, intercept = np.polyfit(x, sec.values, 1)
            resid = sec.values - (slope * x + intercept)
            stints.append({"n": len(sec), "median": sec.median(),
                           "slope": slope, "resid_std": float(np.std(resid))})
        if not stints:
            continue
        n = sum(s["n"] for s in stints)
        rows[drv] = {
            "lr_median": min(s["median"] for s in stints),
            "lr_slope": sum(s["slope"] * s["n"] for s in stints) / n,
            "lr_resid_std": sum(s["resid_std"] * s["n"] for s in stints) / n,
            "lr_laps": n,
        }
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).T
    df["lr_median_pct"] = (df["lr_median"] / df["lr_median"].min() - 1) * 100
    return df


def race_pace_target(laps):
    laps = clean_green(laps)
    med = laps.groupby("Driver")["LapTime"].median().dt.total_seconds()
    counts = laps.groupby("Driver")["LapTime"].count()
    med = med[counts >= 10]  # enough racing laps to mean something
    return (med / med.min() - 1) * 100


def load_session(fastf1, year, rnd, name):
    try:
        ses = fastf1.get_session(year, rnd, name)
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
        if ses.laps is None or len(ses.laps) == 0:
            return None
        return ses
    except Exception as exc:
        print(f"    {name}: unavailable ({type(exc).__name__}: {exc})")
        return None


def build():
    import fastf1
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    fastf1.set_log_level("ERROR")

    all_rows = []
    today = pd.Timestamp.now()
    for year in SEASONS:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        sched = sched[pd.to_datetime(sched["EventDate"]) < today - pd.Timedelta(days=1)]
        for _, ev in sched.iterrows():
            rnd, name = int(ev["RoundNumber"]), ev["EventName"]
            print(f"[{year} R{rnd:02d}] {name}", flush=True)
            race = load_session(fastf1, year, rnd, "R")
            if race is None:
                continue
            target = race_pace_target(race.laps)
            if target.empty:
                continue
            teams = race.laps.groupby("Driver")["Team"].first()

            fp2 = load_session(fastf1, year, rnd, "FP2")
            fp3 = load_session(fastf1, year, rnd, "FP3")
            fp2_best = best_quick_pct(fp2.laps) if fp2 else pd.Series(dtype=float)
            fp3_best = best_quick_pct(fp3.laps) if fp3 else pd.Series(dtype=float)
            lr = long_run_features(fp2.laps) if fp2 else pd.DataFrame()

            for drv, tgt in target.items():
                row = {"season": year, "round": rnd, "event": name,
                       "driver": drv, "team": teams.get(drv, ""),
                       "race_pace_pct": tgt,
                       "fp3_best_pct": fp3_best.get(drv, np.nan),
                       "fp2_best_pct": fp2_best.get(drv, np.nan)}
                if not lr.empty and drv in lr.index:
                    row.update({
                        "fp2_lr_median_pct": lr.loc[drv, "lr_median_pct"],
                        "fp2_deg_slope": lr.loc[drv, "lr_slope"],
                        "fp2_lr_resid_std": lr.loc[drv, "lr_resid_std"],
                        "fp2_lr_laps": lr.loc[drv, "lr_laps"]})
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET, index=False)
    print(f"\nwrote {len(df)} driver-event rows across "
          f"{df.groupby(['season', 'round']).ngroups} events -> {DATASET}")


# ---------------------------------------------------------------- analyze ---

def analyze():
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    df = pd.read_csv(DATASET)
    events = (df[["season", "round", "event"]].drop_duplicates()
              .sort_values(["season", "round"]).reset_index(drop=True))
    print(f"dataset: {len(df)} rows, {len(events)} events\n")

    # -- 1. raw correlation of each feature with race pace (pooled, in-sample)
    print("== Spearman correlation vs race pace (pooled, all events) ==")
    full = df.dropna(subset=["race_pace_pct"])
    for f in FEATURES:
        sub = full.dropna(subset=[f])
        rho, _ = spearmanr(sub[f], sub["race_pace_pct"])
        print(f"  {f:18s} rho={rho:+.3f}   (n={len(sub)})")

    # -- 2. walk-forward backtest
    model_feats = FEATURES
    complete = df.dropna(subset=model_feats + ["race_pace_pct"]).copy()
    ev_key = complete["season"] * 100 + complete["round"]
    events_sorted = sorted(ev_key.unique())

    def event_spearman(pred_col, frame):
        out = []
        for _, g in frame.groupby(["season", "round"]):
            if len(g) >= 8:
                rho, _ = spearmanr(g[pred_col], g["race_pace_pct"])
                out.append(rho)
        return np.mean(out), len(out)

    preds = []
    for i, ek in enumerate(events_sorted):
        if i < MIN_TRAIN_EVENTS:
            continue
        train = complete[ev_key < ek]
        test = complete[ev_key == ek].copy()
        if len(test) < 8:
            continue
        m = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                      learning_rate=0.05, random_state=0)
        m.fit(train[model_feats], train["race_pace_pct"])
        test["pred"] = m.predict(test[model_feats])
        preds.append(test)
    bt = pd.concat(preds)
    rho_model, n_ev = event_spearman("pred", bt)
    print(f"\n== walk-forward backtest ({n_ev} events) ==")
    print(f"  model (all features)      mean per-event Spearman = {rho_model:.3f}")
    for f in model_feats:
        rho_f, _ = event_spearman(f, bt)
        print(f"  baseline: {f:18s} mean per-event Spearman = {rho_f:.3f}")

    hit = (bt.sort_values("pred").groupby(["season", "round"]).first()["race_pace_pct"]
           .rank().mean() if False else None)
    top_hits = []
    for _, g in bt.groupby(["season", "round"]):
        picked = g.loc[g["pred"].idxmin()]
        top_hits.append(picked["race_pace_pct"] == g["race_pace_pct"].min())
    print(f"  model picked the race-pace leader in {sum(top_hits)}/{len(top_hits)} events")

    # -- 3. permutation importance on a model fit to everything
    m = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                  learning_rate=0.05, random_state=0)
    m.fit(complete[model_feats], complete["race_pace_pct"])
    imp = permutation_importance(m, complete[model_feats],
                                 complete["race_pace_pct"],
                                 n_repeats=20, random_state=0)
    print("\n== permutation feature importance (full fit) ==")
    order = np.argsort(-imp.importances_mean)
    for i in order:
        print(f"  {model_feats[i]:18s} {imp.importances_mean[i]:.4f} "
              f"+/- {imp.importances_std[i]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.build or not args.analyze:
        build()
    if args.analyze or not args.build:
        analyze()
