"""
src/preprocessing/preprocess.py
--------------------------------
Handles imputation, normalisation, SOFA score computation,
and feature engineering for PhysioNet 2019 ICU data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


# ─── Feature groups ──────────────────────────────────────────────────────────

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
ALL_FEATURES = VITALS + LABS + DEMOGRAPHICS


# ─── SOFA score components ────────────────────────────────────────────────────

def compute_sofa_score(df: pd.DataFrame) -> pd.Series:
    """Approximate SOFA score from available PhysioNet 2019 features."""
    score = pd.Series(0.0, index=df.index)

    # Respiratory: PaO2/FiO2 (use O2Sat as proxy)
    if "O2Sat" in df.columns and "FiO2" in df.columns:
        pf = df["O2Sat"] / df["FiO2"].clip(lower=0.21)
        score += pd.cut(pf, bins=[-np.inf, 100, 200, 300, 400, np.inf],
                        labels=[4, 3, 2, 1, 0]).astype(float).fillna(0)

    # Coagulation: Platelets
    if "Platelets" in df.columns:
        score += pd.cut(df["Platelets"],
                        bins=[-np.inf, 20, 50, 100, 150, np.inf],
                        labels=[4, 3, 2, 1, 0]).astype(float).fillna(0)

    # Liver: Bilirubin
    if "Bilirubin_total" in df.columns:
        score += pd.cut(df["Bilirubin_total"],
                        bins=[-np.inf, 1.2, 2, 6, 12, np.inf],
                        labels=[0, 1, 2, 3, 4]).astype(float).fillna(0)

    # Cardiovascular: MAP
    if "MAP" in df.columns:
        score += (df["MAP"] < 70).astype(float)

    # Renal: Creatinine
    if "Creatinine" in df.columns:
        score += pd.cut(df["Creatinine"],
                        bins=[-np.inf, 1.2, 2, 3.5, 5, np.inf],
                        labels=[0, 1, 2, 3, 4]).astype(float).fillna(0)

    return score.clip(0, 24)


# ─── Engineered features ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Shock index
    if "HR" in df.columns and "SBP" in df.columns:
        df["shock_index"] = df["HR"] / df["SBP"].replace(0, np.nan)

    # MAP × HR product (cardiovascular stress)
    if "MAP" in df.columns and "HR" in df.columns:
        df["map_hr_product"] = df["MAP"] * df["HR"]

    # Lactate 3h trend (rolling slope proxy)
    if "Lactate" in df.columns:
        df["lactate_trend_3h"] = df.groupby("patient_id")["Lactate"].transform(
            lambda s: s.rolling(3, min_periods=1).mean().diff()
        )

    # WBC 6h trend
    if "WBC" in df.columns:
        df["wbc_trend_6h"] = df.groupby("patient_id")["WBC"].transform(
            lambda s: s.rolling(6, min_periods=1).mean().diff()
        )

    # SOFA score
    df["sofa_score"] = compute_sofa_score(df)

    return df


# ─── Imputation ───────────────────────────────────────────────────────────────

def impute(df: pd.DataFrame, method: str = "forward_fill",
           knn_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    if method == "forward_fill":
        df[feature_cols] = (
            df.groupby("patient_id")[feature_cols]
            .transform(lambda s: s.ffill().bfill())
        )
        # Fill remaining NaN with column median
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    elif method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

    elif method == "median":
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return df


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalise(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str = "zscore",
    feature_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    if feature_cols is None:
        feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]

    scaler = StandardScaler() if method == "zscore" else MinMaxScaler()
    scaler.fit(train_df[feature_cols])

    for split in (train_df, val_df, test_df):
        split[feature_cols] = scaler.transform(split[feature_cols])

    return train_df, val_df, test_df, scaler


# ─── Windowing ────────────────────────────────────────────────────────────────

def create_sequences(
    df: pd.DataFrame,
    lookback: int = 48,
    horizon: int = 6,
    feature_cols: List[str] | None = None,
    label_col: str = "SepsisLabel",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X      : (N, lookback, F) float32 — input sequences
        X_cnn  : (N, lookback_cnn, F_vitals) float32 — short waveform window
        y      : (N,) float32 — label at t+horizon
    """
    if feature_cols is None:
        engineered = ["sofa_score", "shock_index", "lactate_trend_3h",
                      "map_hr_product", "wbc_trend_6h"]
        feature_cols = [c for c in ALL_FEATURES + engineered if c in df.columns]

    vital_cols = [c for c in VITALS if c in df.columns]
    cnn_window = 6  # hours for CNN branch

    Xs, Xc, ys = [], [], []

    for pid, pdata in df.groupby("patient_id"):
        pdata = pdata.sort_values("hour").reset_index(drop=True)
        n = len(pdata)

        for t in range(lookback, n - horizon + 1):
            window = pdata.iloc[t - lookback:t][feature_cols].values
            cnn_win = pdata.iloc[t - cnn_window:t][vital_cols].values
            label = pdata.iloc[t + horizon - 1][label_col]

            Xs.append(window)
            Xc.append(cnn_win)
            ys.append(label)

    X = np.array(Xs, dtype=np.float32)
    X_cnn = np.array(Xc, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)

    logger.info(f"Sequences: X={X.shape}, X_cnn={X_cnn.shape}, y={y.shape}, "
                f"pos_rate={y.mean():.3f}")
    return X, X_cnn, y


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_pipeline(cfg) -> None:
    from src.utils.io import load_physionet, save_splits, split_patients

    logger.info("Loading raw PhysioNet 2019 data …")
    df = load_physionet(cfg.data.raw_dir)

    logger.info("Imputing missing values …")
    df = impute(df, method=cfg.preprocessing.imputation,
                knn_neighbors=cfg.preprocessing.knn_neighbors)

    logger.info("Engineering features …")
    df = engineer_features(df)

    logger.info("Splitting patients …")
    train_df, val_df, test_df = split_patients(
        df,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
    )

    logger.info("Normalising …")
    train_df, val_df, test_df, scaler = normalise(
        train_df, val_df, test_df, method=cfg.preprocessing.normalisation
    )

    logger.info("Creating sequences …")
    datasets = {}
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, X_cnn, y = create_sequences(split, cfg.data.lookback, cfg.data.horizon)
        datasets[name] = (X, X_cnn, y)

    save_splits(datasets, scaler, cfg.data.splits_dir)
    logger.info("Preprocessing complete ✓")
