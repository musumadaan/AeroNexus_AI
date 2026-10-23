# """Predictive maintenance on NASA C-MAPSS: remaining-useful-life (RUL) with calibrated intervals,
# plus a health/anomaly score — trained on the real public dataset (all four subsets).

# Method (standard, reproducible, CPU-only; the deep-learning variant is notebooks/05 on TPU):
#   1. Operating-condition normalisation: FD002/FD004 run under 6 regimes. Regimes are recovered by
#      rounding the 3 operational settings, and each sensor is z-scored *per regime* using training
#      statistics only — otherwise the model learns the regime, not the wear.
#   2. Drop near-constant sensors (computed on train).
#   3. Piece-wise linear RUL target capped at 125 cycles (the usual convention: early-life wear is
#      not observable, so the target is flat until degradation starts).
#   4. Window features at every cycle: last value, mean over last 5/15/30 cycles, slope (OLS) and
#      std over last 30, cycle count. Engines are split by *unit id* (never by row) for validation.
#   5. Models compared on held-out units: mean baseline, gradient-boosted trees, MLP neural net.
#   6. Split-conformal 90% interval on held-out units at random truncation points (mimicking how
#      the official test set truncates engines), so the interval matches test-time conditions.
#   7. Official test: prediction at each test engine's last cycle vs RUL_FDxxx; RMSE (vs true RUL
#      capped at 125, the convention most papers report — uncapped also given) and the PHM08 score.
#   8. Health index: IsolationForest fitted on early-life (healthy) windows only; AUROC of early vs
#      late life on held-out units shows whether the score actually tracks degradation.
# """
# from __future__ import annotations

# import json
# import math
# import time
# from dataclasses import dataclass
# from typing import Any

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
# from sklearn.metrics import roc_auc_score
# from sklearn.neural_network import MLPRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# from ..config import DATA, MODELS
# from ..public_data import REGISTRY

# COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
# SENSORS = [f"s{i}" for i in range(1, 22)]
# CAP = 125
# WINDOWS = (5, 15, 30)


# def load(subset: str, split: str) -> pd.DataFrame:
#     path = REGISTRY["cmapss"].folder / f"{split}_{subset}.txt"
#     df = pd.read_csv(path, sep=r"\s+", header=None)
#     df = df.iloc[:, : len(COLS)]
#     df.columns = COLS
#     return df


# def load_rul(subset: str) -> np.ndarray:
#     return pd.read_csv(REGISTRY["cmapss"].folder / f"RUL_{subset}.txt", header=None).iloc[:, 0].to_numpy(float)


# def add_target(df: pd.DataFrame) -> pd.DataFrame:
#     life = df.groupby("unit")["cycle"].transform("max")
#     df = df.copy()
#     df["rul"] = np.minimum(life - df["cycle"], CAP)
#     return df


# @dataclass
# class Normalizer:
#     sensors: list[str]
#     stats: dict[str, tuple[np.ndarray, np.ndarray]]

#     @staticmethod
#     def regime(df: pd.DataFrame) -> pd.Series:
#         return (df["os1"].round(0).astype(int).astype(str) + "_" + df["os2"].round(2).astype(str) + "_"
#                 + df["os3"].round(0).astype(int).astype(str))

#     @classmethod
#     def fit(cls, train: pd.DataFrame) -> "Normalizer":
#         reg = cls.regime(train)
#         # keep sensors that still vary after per-regime normalisation (drops the flat ones)
#         keep = []
#         for s in SENSORS:
#             within = train.groupby(reg)[s].std().fillna(0)
#             if within.mean() > 1e-4:
#                 keep.append(s)
#         stats = {}
#         for r, g in train.groupby(reg):
#             stats[r] = (g[keep].mean().to_numpy(), g[keep].std().replace(0, 1).fillna(1).to_numpy())
#         return cls(keep, stats)

#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         reg = self.regime(df)
#         out = df[["unit", "cycle"]].copy()
#         X = df[self.sensors].to_numpy(float)
#         Z = np.zeros_like(X)
#         glob_mu = np.mean([m for m, _ in self.stats.values()], axis=0)
#         glob_sd = np.mean([s for _, s in self.stats.values()], axis=0)
#         for r in reg.unique():
#             idx = (reg == r).to_numpy()
#             mu, sd = self.stats.get(r, (glob_mu, glob_sd))  # unseen regime → global stats
#             Z[idx] = (X[idx] - mu) / sd
#         out[self.sensors] = Z
#         return out


# def window_features(z: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
#     """Features for every row using only that engine's history up to the row (no look-ahead)."""
#     parts = []
#     for _, g in z.groupby("unit", sort=False):
#         g = g.sort_values("cycle")
#         feats = {"cycle": g["cycle"].to_numpy(float)}
#         for s in sensors:
#             x = g[s]
#             feats[f"{s}_last"] = x.to_numpy()
#             for w in WINDOWS:
#                 feats[f"{s}_m{w}"] = x.rolling(w, min_periods=1).mean().to_numpy()
#             feats[f"{s}_sd30"] = x.rolling(30, min_periods=2).std().fillna(0).to_numpy()
#             # rolling OLS slope over last 30 cycles
#             t = np.arange(len(x), dtype=float)
#             xs = x.to_numpy()
#             slope = np.zeros(len(x))
#             for i in range(len(x)):
#                 lo = max(0, i - 29)
#                 if i - lo >= 2:
#                     tt, yy = t[lo: i + 1], xs[lo: i + 1]
#                     tt = tt - tt.mean()
#                     slope[i] = float((tt * (yy - yy.mean())).sum() / (tt * tt).sum())
#             feats[f"{s}_slope30"] = slope
#         f = pd.DataFrame(feats, index=g.index)
#         f["unit"] = g["unit"].to_numpy()
#         parts.append(f)
#     return pd.concat(parts).loc[z.index]


# def phm_score(pred: np.ndarray, true: np.ndarray) -> float:
#     d = pred - true
#     return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# def rmse(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.sqrt(np.mean((a - b) ** 2)))


# def truncation_sample(df_units: pd.DataFrame, rng: np.random.Generator, per_unit: int = 5) -> pd.Index:
#     """Random truncation points per held-out unit, like the official test set."""
#     idx = []
#     for _, g in df_units.groupby("unit"):
#         n = len(g)
#         cuts = rng.integers(low=min(31, n - 1), high=n, size=min(per_unit, max(1, n - 31)))
#         idx.extend(g.index[cuts])
#     return pd.Index(idx)


# def train_subset(subset: str, seed: int = 7, alpha: float = 0.1) -> dict[str, Any]:
#     t0 = time.time()
#     rng = np.random.default_rng(seed)
#     tr = add_target(load(subset, "train"))
#     norm = Normalizer.fit(tr)
#     feats = window_features(norm.transform(tr), norm.sensors)
#     fcols = [c for c in feats.columns if c != "unit"]
#     y = tr["rul"].to_numpy(float)
#     units = tr["unit"].unique()
#     rng.shuffle(units)
#     n_cal = max(10, int(0.2 * len(units)))
#     cal_units, fit_units = set(units[:n_cal]), set(units[n_cal:])
#     fit_mask = tr["unit"].isin(fit_units).to_numpy()
#     X = feats[fcols].to_numpy(float)

#     cal_idx = truncation_sample(tr[tr["unit"].isin(cal_units)], rng)
#     cal_pos = tr.index.get_indexer(cal_idx)

#     models = {
#         "gbt": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
#                                              random_state=seed),
#         "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
#                                                             max_iter=60, early_stopping=True, random_state=seed)),
#     }
#     val = {"mean_baseline": rmse(np.full(len(cal_pos), y[fit_mask].mean()), y[cal_pos])}
#     for name, m in models.items():
#         m.fit(X[fit_mask], y[fit_mask])
#         val[name] = rmse(np.clip(m.predict(X[cal_pos]), 0, CAP), y[cal_pos])
#     best = min(models, key=lambda k: val[k])
#     model = models[best]
#     res = np.abs(np.clip(model.predict(X[cal_pos]), 0, CAP) - y[cal_pos])
#     k = min(len(res) - 1, math.ceil((len(res) + 1) * (1 - alpha)) - 1)
#     q = float(np.sort(res)[k])
#     model.fit(X, y)  # refit on all training units for the final model

#     # official test: last cycle of each test engine
#     te = load(subset, "test")
#     te_feats = window_features(norm.transform(te), norm.sensors)
#     last = te.groupby("unit")["cycle"].idxmax()
#     Xte = te_feats.loc[last, fcols].to_numpy(float)
#     true = load_rul(subset)
#     pred = np.clip(model.predict(Xte), 0, CAP)
#     true_cap = np.minimum(true, CAP)
#     covered = float(np.mean(np.abs(pred - true_cap) <= q))

#     # health index: fit on early-life windows, check it separates early vs late life on held-out units
#     early = (tr["cycle"] <= 30).to_numpy() & fit_mask
#     iso = IsolationForest(n_estimators=200, random_state=seed).fit(X[early][:, 1:])
#     hold = ~fit_mask
#     life = tr.groupby("unit")["cycle"].transform("max").to_numpy()
#     lab_early = hold & (tr["cycle"].to_numpy() <= 30)
#     lab_late = hold & (life - tr["cycle"].to_numpy() <= 30)
#     scores = -iso.score_samples(X[:, 1:])
#     health_threshold = float(np.quantile(scores[early], 0.99))  # 99th pct of healthy-engine scores
#     auroc = float(roc_auc_score(np.r_[np.zeros(lab_early.sum()), np.ones(lab_late.sum())],
#                                 np.r_[scores[lab_early], scores[lab_late]]))

#     metrics = {
#         "subset": subset, "n_train_units": int(len(units)), "n_test_units": int(len(true)), "sensors_used": norm.sensors,
#         "validation_rmse": {k_: round(v, 3) for k_, v in val.items()}, "selected_model": best,
#         "test_rmse_capped": round(rmse(pred, true_cap), 3), "test_rmse_uncapped": round(rmse(pred, true), 3),
#         "test_phm_score": round(phm_score(pred, true_cap), 1), "interval_halfwidth_cycles": round(q, 2),
#         "interval_coverage_test": round(covered, 4), "target_coverage": 1 - alpha,
#         "health_index_auroc_early_vs_late": round(auroc, 4),
#         "health_threshold": round(health_threshold, 4), "train_seconds": round(time.time() - t0, 1),
#         "data": "NASA C-MAPSS (real public dataset, simulated engines)",
#     }
#     fleet = [{"unit": int(u), "cycles_flown": int(te.loc[last.loc[u], "cycle"]), "pred_rul": round(float(p), 1),
#               "low": round(max(0.0, float(p) - q), 1), "high": round(min(CAP, float(p) + q), 1), "true_rul": float(t),
#               "health": round(float(-iso.score_samples(Xte[i:i + 1, 1:])[0]), 4)}
#              for i, (u, p, t) in enumerate(zip(last.index, pred, true))]
#     bundle = {"subset": subset, "normalizer": norm, "model": model, "q": q, "features": fcols, "iso": iso, "metrics": metrics,
#               "health_threshold": health_threshold}
#     return {"metrics": metrics, "fleet": fleet, "bundle": bundle}


# def train_all(subsets: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")) -> dict[str, Any]:
#     import joblib

#     MODELS.mkdir(parents=True, exist_ok=True)
#     out = {}
#     for s in subsets:
#         r = train_subset(s)
#         joblib.dump(r["bundle"], MODELS / f"pdm_{s}.joblib", compress=3)
#         out[s] = {"metrics": r["metrics"], "fleet": r["fleet"]}
#         print(s, json.dumps({k: v for k, v in r["metrics"].items() if k != "sensors_used"}))
#     (DATA / "reports").mkdir(parents=True, exist_ok=True)
#     (DATA / "reports" / "pdm_cmapss.json").write_text(json.dumps(out, indent=1))
#     return out


# def predict_engine(bundle: dict[str, Any], history: pd.DataFrame) -> dict[str, float]:
#     """RUL + interval + health for one engine's sensor history (columns as in C-MAPSS)."""
#     z = bundle["normalizer"].transform(history)
#     f = window_features(z, bundle["normalizer"].sensors)
#     x = f[bundle["features"]].to_numpy(float)[-1:]
#     p = float(np.clip(bundle["model"].predict(x)[0], 0, CAP))
#     q = bundle["q"]
#     return {"pred_rul": round(p, 1), "low": round(max(0.0, p - q), 1), "high": round(min(CAP, p + q), 1),
#             "health": round(float(-bundle["iso"].score_samples(x[:, 1:])[0]), 4)}


# if __name__ == "__main__":
#     train_all()

"""Predictive maintenance on NASA C-MAPSS.

Predicts remaining useful life (RUL), uncertainty intervals, and a health score
for FD001, FD002, FD003, and FD004.

Design:
  1. Split training engines into fit, validation, and calibration groups.
  2. Fit normalization statistics on fit engines only.
  3. Compare GBT and MLP on one random truncation point per validation engine.
  4. Refit the selected model on fit + validation engines.
  5. Calibrate intervals using one truncation point per calibration engine.
  6. Evaluate once on the official test set.
  7. Fit the health model on early-life fit-engine windows only.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DATA, MODELS
from ..public_data import REGISTRY

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [
    f"s{i}" for i in range(1, 22)
]
SENSORS = [f"s{i}" for i in range(1, 22)]
CAP = 125
WINDOWS = (5, 15, 30)


def load(subset: str, split: str) -> pd.DataFrame:
    path = REGISTRY["cmapss"].folder / f"{split}_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    return df


def load_rul(subset: str) -> np.ndarray:
    path = REGISTRY["cmapss"].folder / f"RUL_{subset}.txt"
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy(float)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Capped RUL targets for the complete training trajectories."""
    out = df.copy()
    life = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = np.minimum(life - out["cycle"], CAP)
    return out


@dataclass
class Normalizer:
    sensors: list[str]
    stats: dict[str, tuple[np.ndarray, np.ndarray]]

    @staticmethod
    def regime(df: pd.DataFrame) -> pd.Series:
        return (
            df["os1"].round(0).astype(int).astype(str)
            + "_"
            + df["os2"].round(2).astype(str)
            + "_"
            + df["os3"].round(0).astype(int).astype(str)
        )

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "Normalizer":
        """Fit sensor selection and statistics using fit engines only."""
        reg = cls.regime(train)

        keep = []
        for sensor in SENSORS:
            within_regime_std = train.groupby(reg)[sensor].std().fillna(0)
            if within_regime_std.mean() > 1e-4:
                keep.append(sensor)

        if not keep:
            raise ValueError("No usable C-MAPSS sensors remain after filtering.")

        stats = {}
        for regime, group in train.groupby(reg):
            means = group[keep].mean().to_numpy(float)
            stds = (
                group[keep].std()
                .replace(0, 1)
                .fillna(1)
                .to_numpy(dtype=float, copy=True)
            )
            stds[stds < 1e-8] = 1.0
            stats[regime] = (means, stds)

        return cls(sensors=keep, stats=stats)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        reg = self.regime(df)
        out = df[["unit", "cycle"]].copy()
        X = df[self.sensors].to_numpy(float)
        Z = np.zeros_like(X)

        global_mu = np.mean(
            [means for means, _ in self.stats.values()], axis=0
        )
        global_sd = np.mean(
            [stds for _, stds in self.stats.values()], axis=0
        )
        global_sd = np.maximum(global_sd, 1e-8)

        for regime in reg.unique():
            mask = (reg == regime).to_numpy()
            means, stds = self.stats.get(
                regime, (global_mu, global_sd)
            )
            Z[mask] = (X[mask] - means) / np.maximum(stds, 1e-8)

        out[self.sensors] = Z
        return out


def window_features(
    z: pd.DataFrame, sensors: list[str]
) -> pd.DataFrame:
    """Build causal features using only each engine's history up to each cycle."""
    parts = []

    for _, group in z.groupby("unit", sort=False):
        group = group.sort_values("cycle")
        features = {"cycle": group["cycle"].to_numpy(float)}

        for sensor in sensors:
            series = group[sensor]
            values = series.to_numpy(float)

            features[f"{sensor}_last"] = values

            for window in WINDOWS:
                features[f"{sensor}_m{window}"] = (
                    series.rolling(window, min_periods=1)
                    .mean()
                    .to_numpy()
                )

            features[f"{sensor}_sd30"] = (
                series.rolling(30, min_periods=2)
                .std()
                .fillna(0)
                .to_numpy()
            )

            # OLS slope over the preceding 30 observations.
            slope = np.zeros(len(values), dtype=float)
            time_index = np.arange(len(values), dtype=float)

            for i in range(len(values)):
                lo = max(0, i - 29)
                if i - lo >= 2:
                    t = time_index[lo:i + 1]
                    y = values[lo:i + 1]
                    t_centered = t - t.mean()
                    denominator = np.sum(t_centered ** 2)

                    if denominator > 0:
                        slope[i] = np.sum(
                            t_centered * (y - y.mean())
                        ) / denominator

            features[f"{sensor}_slope30"] = slope

        frame = pd.DataFrame(features, index=group.index)
        frame["unit"] = group["unit"].to_numpy()
        parts.append(frame)

    return pd.concat(parts).loc[z.index]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def phm_score(pred: np.ndarray, true: np.ndarray) -> float:
    difference = pred - true
    penalties = np.where(
        difference < 0,
        np.exp(-difference / 13) - 1,
        np.exp(difference / 10) - 1,
    )
    return float(np.sum(penalties))


def truncation_sample(
    df_units: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Index:
    """Select exactly one random truncation point per engine."""
    indices = []

    for _, group in df_units.groupby("unit", sort=True):
        group = group.sort_values("cycle")
        n = len(group)

        if n == 0:
            continue

        # Prefer truncations after 30 cycles where possible.
        low = min(30, n - 1)
        position = int(rng.integers(low=low, high=n))
        indices.append(group.index[position])

    return pd.Index(indices)


def conformal_quantile(
    residuals: np.ndarray, alpha: float
) -> float:
    """Finite-sample split-conformal absolute-residual quantile."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) == 0:
        raise ValueError("No valid calibration residuals were produced.")

    rank = math.ceil((len(residuals) + 1) * (1 - alpha))

    # If the finite-sample rank exceeds n, the formal finite-sample
    # threshold is infinite. Do not silently substitute a smaller rank.
    if rank > len(residuals):
        return float("inf")

    return float(np.sort(residuals)[rank - 1])


def train_subset(
    subset: str, seed: int = 7, alpha: float = 0.1
) -> dict[str, Any]:
    t0 = time.time()
    rng = np.random.default_rng(seed)

    tr = add_target(load(subset, "train"))
    units = np.asarray(sorted(tr["unit"].unique()))
    rng.shuffle(units)

    n_units = len(units)
    n_val = max(1, int(round(0.2 * n_units)))
    n_cal = max(1, int(round(0.2 * n_units)))

    if n_units - n_val - n_cal < 1:
        raise ValueError(
            f"{subset}: not enough engines for fit/validation/calibration."
        )

    val_units = set(units[:n_val])
    cal_units = set(units[n_val:n_val + n_cal])
    fit_units = set(units[n_val + n_cal:])

    fit_mask = tr["unit"].isin(fit_units).to_numpy()
    val_mask = tr["unit"].isin(val_units).to_numpy()
    cal_mask = tr["unit"].isin(cal_units).to_numpy()
    fit_val_mask = fit_mask | val_mask

    # Preprocessing is fitted exclusively on model-fitting engines.
    norm = Normalizer.fit(tr.loc[fit_mask])
    features = window_features(norm.transform(tr), norm.sensors)
    feature_cols = [c for c in features.columns if c != "unit"]

    X = features[feature_cols].to_numpy(float)
    y = tr["rul"].to_numpy(float)

    val_indices = truncation_sample(tr.loc[val_mask], rng)
    val_positions = tr.index.get_indexer(val_indices)

    if len(val_positions) == 0:
        raise ValueError(f"{subset}: no validation truncation points.")

    models = {
        "gbt": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=100,
                # Avoid sklearn's row-level internal validation split.
                # Engine-level validation is handled explicitly above.
                early_stopping=False,
                random_state=seed,
            ),
        ),
    }

    metrics_val = {
        "mean_baseline": rmse(
            y[val_positions],
            np.full(len(val_positions), float(y[fit_mask].mean())),
        )
    }

    for name, candidate in models.items():
        candidate.fit(X[fit_mask], y[fit_mask])
        predictions = np.clip(
            candidate.predict(X[val_positions]), 0, CAP
        )
        metrics_val[name] = rmse(y[val_positions], predictions)

    selected_name = min(
        models, key=lambda name: metrics_val[name]
    )

    # Refit on fit + validation engines. Calibration engines remain unseen.
    model = models[selected_name]
    model.fit(X[fit_val_mask], y[fit_val_mask])

    # Calibrate with exactly one truncation point per calibration engine.
    # The fitted model is not refitted after this step.
    cal_indices = truncation_sample(tr.loc[cal_mask], rng)
    cal_positions = tr.index.get_indexer(cal_indices)

    if len(cal_positions) < 2:
        raise ValueError(
            f"{subset}: too few calibration engines for interval calibration."
        )

    cal_predictions = np.clip(
        model.predict(X[cal_positions]), 0, CAP
    )
    cal_residuals = np.abs(y[cal_positions] - cal_predictions)
    q = conformal_quantile(cal_residuals, alpha)

    # Official test set: one prediction at each engine's final observed cycle.
    te = load(subset, "test")
    test_features = window_features(norm.transform(te), norm.sensors)
    last_rows = te.groupby("unit")["cycle"].idxmax()

    X_test = test_features.loc[last_rows, feature_cols].to_numpy(float)
    true_rul = load_rul(subset)

    if len(X_test) != len(true_rul):
        raise ValueError(
            f"{subset}: {len(X_test)} test engines but "
            f"{len(true_rul)} RUL labels."
        )

    pred = np.clip(model.predict(X_test), 0, CAP)
    true_capped = np.minimum(true_rul, CAP)

    covered = float(
        np.mean(np.abs(pred - true_capped) <= q)
    )

    # Fit the health model using early-life fit-engine windows only.
    early_fit = (
        (tr["cycle"].to_numpy() <= 30) & fit_mask
    )

    if not early_fit.any():
        raise ValueError(f"{subset}: no early-life fit windows.")

    # The first feature column is cycle, so exclude it for the health model.
    iso = IsolationForest(
        n_estimators=200,
        random_state=seed,
    )
    iso.fit(X[early_fit][:, 1:])

    life = tr.groupby("unit")["cycle"].transform("max").to_numpy()
    cycle = tr["cycle"].to_numpy()
    held_out = tr["unit"].isin(val_units | cal_units).to_numpy()

    early_eval = held_out & (cycle <= 30)
    late_eval = held_out & ((life - cycle) <= 30)

    if early_eval.any() and late_eval.any():
        early_scores = -iso.score_samples(X[early_eval][:, 1:])
        late_scores = -iso.score_samples(X[late_eval][:, 1:])
        labels = np.r_[
            np.zeros(len(early_scores)),
            np.ones(len(late_scores)),
        ]
        scores = np.r_[early_scores, late_scores]
        health_auroc = float(roc_auc_score(labels, scores))
    else:
        health_auroc = float("nan")

    healthy_scores = -iso.score_samples(X[early_fit][:, 1:])
    health_threshold = float(np.quantile(healthy_scores, 0.99))

    fleet = []
    for i, (unit, prediction, truth) in enumerate(
        zip(last_rows.index, pred, true_rul)
    ):
        fleet.append({
            "unit": int(unit),
            "cycles_flown": int(te.loc[last_rows.loc[unit], "cycle"]),
            "pred_rul": round(float(prediction), 1),
            "low": round(max(0.0, float(prediction) - q), 1),
            "high": round(min(CAP, float(prediction) + q), 1),
            "true_rul": float(truth),
            "health": round(
                float(-iso.score_samples(X_test[i:i + 1, 1:])[0]), 4
            ),
        })

    metrics = {
        "subset": subset,
        "n_train_units": int(n_units),
        "n_fit_units": int(len(fit_units)),
        "n_validation_units": int(len(val_units)),
        "n_calibration_units": int(len(cal_units)),
        "n_test_units": int(len(true_rul)),
        "sensors_used": norm.sensors,
        "validation_rmse": {
            name: round(float(value), 3)
            for name, value in metrics_val.items()
        },
        "selected_model": selected_name,
        "calibration_points": int(len(cal_positions)),
        "calibration_residual_q": (
            round(q, 3) if np.isfinite(q) else None
        ),
        "test_rmse_capped": round(rmse(pred, true_capped), 3),
        "test_rmse_uncapped": round(rmse(pred, true_rul), 3),
        "test_phm_score": round(phm_score(pred, true_capped), 1),
        "interval_halfwidth_cycles": (
            round(q, 2) if np.isfinite(q) else None
        ),
        "interval_coverage_test": round(covered, 4),
        "target_coverage": 1 - alpha,
        "health_index_auroc_early_vs_late": (
            round(health_auroc, 4)
            if np.isfinite(health_auroc) else None
        ),
        "health_threshold": round(health_threshold, 4),
        "train_seconds": round(time.time() - t0, 1),
        "data": "NASA C-MAPSS public dataset (simulated engines)",
    }

    bundle = {
        "subset": subset,
        "normalizer": norm,
        "model": model,
        "q": q,
        "features": feature_cols,
        "iso": iso,
        "metrics": metrics,
        "health_threshold": health_threshold,
    }

    return {
        "metrics": metrics,
        "fleet": fleet,
        "bundle": bundle,
    }


def train_all(
    subsets: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
) -> dict[str, Any]:
    import joblib

    MODELS.mkdir(parents=True, exist_ok=True)
    results = {}

    for subset in subsets:
        result = train_subset(subset)
        joblib.dump(
            result["bundle"],
            MODELS / f"pdm_{subset}.joblib",
            compress=3,
        )
        results[subset] = {
            "metrics": result["metrics"],
            "fleet": result["fleet"],
        }
        printable = {
            key: value
            for key, value in result["metrics"].items()
            if key != "sensors_used"
        }
        print(subset, json.dumps(printable))

    report_dir = DATA / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "pdm_cmapss.json").write_text(
        json.dumps(results, indent=1, default=str)
    )
    return results


def predict_engine(
    bundle: dict[str, Any], history: pd.DataFrame
) -> dict[str, float]:
    """Predict RUL, interval, and health from one engine's history."""
    z = bundle["normalizer"].transform(history)
    features = window_features(z, bundle["normalizer"].sensors)
    X = features[bundle["features"]].to_numpy(float)[-1:]

    prediction = float(
        np.clip(bundle["model"].predict(X)[0], 0, CAP)
    )
    q = bundle["q"]
    health = float(
        -bundle["iso"].score_samples(X[:, 1:])[0]
    )

    return {
        "pred_rul": round(prediction, 1),
        "low": round(max(0.0, prediction - q), 1),
        "high": round(min(CAP, prediction + q), 1),
        "health": round(health, 4),
    }


if __name__ == "__main__":
    train_all()
