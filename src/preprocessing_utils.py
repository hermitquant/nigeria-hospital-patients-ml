"""
Reusable, heavily commented utilities for loading data from Hugging Face, computing
basic statistics, generating EDA visualizations, and building a robust
scikit-learn preprocessing pipeline.

All functions are written to be composable and side-effect-light. Plot functions
save figures to disk instead of showing GUI windows.
"""
from __future__ import annotations

# Standard library imports
import os
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from global_gender_predictor import GlobalGenderPredictor

# Use a non-interactive backend so plots can be saved without a GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


# ------------------------------
# General utilities
# ------------------------------

def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists and return it as a Path.

    This is a tiny convenience for consistent directory creation in one line.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_plot_style() -> None:
    """Set a clean, publication-friendly seaborn/matplotlib style.

    This keeps visualizations consistent and legible across runs.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    })


# ------------------------------
# Data loading and schema helpers
# ------------------------------

def load_hf_parquet(
    hf_path: str,
    columns: Optional[List[str]] = None,
    storage_options: Optional[Dict] = None,
    engine: str = "pyarrow",
) -> pd.DataFrame:
    """Load a parquet file directly from a Hugging Face dataset via fsspec.

    Parameters
    ----------
    hf_path : str
        Path like "hf://datasets/<org_or_user>/<repo>/<path_in_repo>.parquet".
    columns : Optional[List[str]]
        Optional subset of columns to read for efficiency.
    storage_options : Optional[Dict]
        Extra keyword args forwarded to fsspec. If `HF_TOKEN` is set in the
        environment, it will be attached automatically to support private repos.
    engine : str
        Parquet engine, defaults to "pyarrow".

    Returns
    -------
    pd.DataFrame
    """
    opts = dict(storage_options or {})
    # Attach HF token automatically if present (harmless for public datasets)
    if os.environ.get("HF_TOKEN") and "token" not in opts:
        opts["token"] = os.environ["HF_TOKEN"]

    # pandas >= 1.2 supports storage_options with remote filesystems
    df = pd.read_parquet(hf_path, columns=columns, storage_options=opts, engine=engine)
    return df


def export_basic_tables(df: pd.DataFrame, reports_dir: Path) -> None:
    """Export basic CSV tables for quick inspection and versioned artifacts.

    Files written:
    - schema_dtypes.csv: column -> dtype
    - missing_values.csv: column, missing_count, missing_pct
    - numeric_describe.csv: describe(include=[np.number])
    - categorical_cardinality.csv: column, n_unique (object, category, bool)
    """
    ensure_dir(reports_dir)

    # Schema / dtypes
    schema_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    schema_df.to_csv(reports_dir / "schema_dtypes.csv", index=False)

    # Missingness
    miss = df.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
    miss["missing_pct"] = (miss["missing_count"] / len(df) * 100.0).round(2)
    miss.to_csv(reports_dir / "missing_values.csv", index=False)

    # Numeric summary
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        num_desc = df.select_dtypes(include=[np.number]).describe().T
        num_desc.to_csv(reports_dir / "numeric_describe.csv")

    # Categorical cardinality
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        card = pd.DataFrame({
            "column": cat_cols,
            "n_unique": [df[c].nunique(dropna=True) for c in cat_cols],
        })
        card.to_csv(reports_dir / "categorical_cardinality.csv", index=False)


# ------------------------------
# Type inference and coercion
# ------------------------------

def infer_datetime_columns(df: pd.DataFrame, max_cols: int = 50) -> List[str]:
    """Heuristically detect datetime-like columns and parse them.

    Strategy:
    - Any column already of datetime dtype is accepted
    - For object columns, try parsing with pandas. If >= 80% of non-nulls
      parse successfully, treat as datetime.
    The function returns the list of columns that look datetime-like (parsed or not).
    """
    dt_cols: List[str] = []
    # Already datetime69 typed
    preset = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    dt_cols.extend(preset)

    # Try parsing object columns conservatively
    candidates = [c for c in df.columns if df[c].dtype == "object" and c not in preset]
    for c in candidates[:max_cols]:
        s = df[c].dropna()
        if s.empty:
            continue
        parsed = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        success_ratio = parsed.notna().mean()
        if success_ratio >= 0.8:
            dt_cols.append(c)
    return sorted(set(dt_cols))


def coerce_datetimes(df: pd.DataFrame, datetime_cols: Iterable[str]) -> pd.DataFrame:
    """Return a copy of df with provided columns converted to pandas datetime.

    Columns that fail to parse remain unchanged to avoid destructive behavior.
    """
    df2 = df.copy()
    for c in datetime_cols:
        try:
            df2[c] = pd.to_datetime(df2[c], errors="coerce", infer_datetime_format=True)
        except Exception:
            # Keep original if coercion fails catastrophically
            pass
    return df2


def infer_column_types(df: pd.DataFrame, explicit_datetime: Optional[str] = None) -> Dict[str, List[str]]:
    """Infer basic semantic column groups used by the preprocessing pipeline.

    Returns a dict with keys: numeric, categorical, datetime.
    - Numeric: pandas numeric dtypes (int, float)
    - Categorical: object, category, bool
    - Datetime: inferred via `infer_datetime_columns` plus `explicit_datetime` if provided
    """
    datetime_cols = infer_datetime_columns(df)
    if explicit_datetime and explicit_datetime not in datetime_cols and explicit_datetime in df.columns:
        datetime_cols.append(explicit_datetime)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Exclude datetime columns from cat bucket if any overlap
    cat_cols = [c for c in cat_cols if c not in datetime_cols]

    return {"numeric": num_cols, "categorical": cat_cols, "datetime": datetime_cols}

def basic_overview(df: pd.DataFrame) -> Dict[str, object]:
    """Compute basic, human-friendly dataset overview information.

    Returns a dict so callers can both log and serialize to JSON if desired.
    """
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100.0).round(2)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    overview = {
        "shape": {"rows": int(n_rows), "columns": int(n_cols)},
        "memory_mb": float(round(memory_mb, 3)),
        "dtypes": dtypes,
        "missing_counts": missing_counts.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "n_unique_per_col": df.nunique(dropna=True).to_dict(),
    }
    return overview

# ------------------------------
# EDA plotters (saved to disk)
# ------------------------------

def plot_numeric_distributions(df: pd.DataFrame, outdir: Path, sample_n: Optional[int] = 5000) -> None:
    """Plot and save histograms and boxplots for numeric columns.

    `sample_n` caps the number of rows used for plotting to keep runtime reasonable.
    """
    ensure_dir(outdir)
    set_plot_style()
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return

    # Subsample for speed if requested
    plot_df = num_df.sample(n=min(sample_n, len(num_df)), random_state=42) if sample_n else num_df

    # Histograms
    ncols = min(4, plot_df.shape[1])
    nrows = int(np.ceil(plot_df.shape[1] / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, plot_df.columns):
        sns.histplot(plot_df[col].dropna(), kde=False, bins=30, ax=ax)
        ax.set_title(f"Histogram: {col}")
    # Hide any unused subplots
    for ax in axes[len(plot_df.columns):]:
        ax.axis("off")
    fig.suptitle("Numeric Distributions (Histograms)")
    fig.savefig(outdir / "numeric_histograms.png")
    plt.close(fig)

    # Boxplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, plot_df.columns):
        sns.boxplot(x=plot_df[col], ax=ax)
        ax.set_title(f"Boxplot: {col}")
    for ax in axes[len(plot_df.columns):]:
        ax.axis("off")
    fig.suptitle("Numeric Distributions (Boxplots)")
    fig.savefig(outdir / "numeric_boxplots.png")
    plt.close(fig)


def plot_categorical_counts(df: pd.DataFrame, outdir: Path, max_cats: int = 30) -> None:
    """Plot bar charts for categorical columns up to `max_cats` distinct values.
    """
    ensure_dir(outdir)
    set_plot_style()
    cat_df = df.select_dtypes(include=["object", "category", "bool"]).copy()
    if cat_df.shape[1] == 0:
        return

    for col in cat_df.columns:
        vc = cat_df[col].astype("category").value_counts(dropna=False)
        top = vc.head(max_cats)
        fig, ax = plt.subplots(figsize=(max(6, min(12, 0.2 * len(top))), 4))
        sns.barplot(x=top.values, y=top.index.astype(str), ax=ax, orient="h")
        ax.set_title(f"Counts: {col}")
        ax.set_xlabel("Count")
        fig.savefig(outdir / f"categorical_counts__{col}.png")
        plt.close(fig)


def plot_missingness_bars(df: pd.DataFrame, outdir: Path) -> None:
    """Plot percentage of missing values per column.
    """
    ensure_dir(outdir)
    set_plot_style()
    miss_pct = (df.isna().sum() / len(df) * 100.0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(miss_pct))))
    sns.barplot(x=miss_pct.values, y=miss_pct.index, ax=ax, orient="h")
    ax.set_title("Missingness (% of rows)")
    ax.set_xlabel("Percent missing")
    fig.savefig(outdir / "missingness_bars.png")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, outdir: Path, sample_n: Optional[int] = 10000) -> None:
    """Plot correlation heatmap for numeric columns.

    Subsamples to at most `sample_n` rows for speed if provided.
    """
    ensure_dir(outdir)
    set_plot_style()
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] <= 1:
        return
    if sample_n is not None and len(num_df) > sample_n:
        num_df = num_df.sample(sample_n, random_state=42)
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * corr.shape[1]), max(5, 0.6 * corr.shape[0])))
    sns.heatmap(corr, cmap="vlag", center=0.0, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap (numeric)")
    fig.savefig(outdir / "correlation_heatmap.png")
    plt.close(fig)


def plot_time_series_counts(df: pd.DataFrame, datetime_cols: List[str], outdir: Path, freq: str = "D") -> None:
    """If any datetime columns exist, plot counts over time at a given frequency.
    """
    ensure_dir(outdir)
    set_plot_style()
    if not datetime_cols:
        return

    # Use the first datetime column as the default temporal axis
    dt = datetime_cols[0]
    if df[dt].dtype.kind != "M":  # not datetime64[ns]
        # try to coerce if caller didn't already
        try:
            s = pd.to_datetime(df[dt], errors="coerce")
        except Exception:
            return
    else:
        s = df[dt]

    ts = s.dropna().dt.to_period(freq).dt.to_timestamp().value_counts().sort_index()
    if ts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax)
    ax.set_title(f"Record counts over time ({dt}, freq={freq})")
    ax.set_ylabel("Count")
    fig.savefig(outdir / f"time_counts_{dt}_{freq}.png")
    plt.close(fig)


# ------------------------------
# Preprocessing pipeline
# ------------------------------

class DatetimeFeaturizer(BaseEstimator, TransformerMixin):
    """Transform datetime-like columns into numeric calendar features.

    For each datetime column provided by the ColumnTransformer, this transformer
    extracts a set of components such as year, month, day, dayofweek, and hour.
    Output is a pandas DataFrame of numeric columns, suitable for downstream
    imputers/scalers and models.
    """

    def __init__(self, features: Tuple[str, ...] = ("year", "month", "day", "dayofweek", "hour")):
        self.features = tuple(features)
        self._cols_: List[str] = []

    def fit(self, X, y=None):
        # Remember incoming column names for consistent feature naming
        if isinstance(X, pd.DataFrame):
            self._cols_ = X.columns.tolist()
        else:
            # If ndarray is passed, synthesize generic names
            self._cols_ = [f"dt_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        # Normalize to DataFrame for convenient datetime access
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self._cols_)

        out_parts: List[pd.DataFrame] = []
        for col in self._cols_:
            # Coerce to datetime; failed parses become NaT
            s = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            feat_dict = {}
            if "year" in self.features:
                feat_dict[f"{col}__year"] = s.dt.year
            if "month" in self.features:
                feat_dict[f"{col}__month"] = s.dt.month
            if "day" in self.features:
                feat_dict[f"{col}__day"] = s.dt.day
            if "dayofweek" in self.features:
                feat_dict[f"{col}__dow"] = s.dt.dayofweek
            if "hour" in self.features:
                feat_dict[f"{col}__hour"] = s.dt.hour
            out_parts.append(pd.DataFrame(feat_dict))

        out = pd.concat(out_parts, axis=1)
        # Return as DataFrame; downstream steps can impute/scale as needed
        return out

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for datetime-derived columns.

        Parameters
        ----------
        input_features : array-like of str, default=None
            Not used; the transformer tracks its own column names via `self._cols_`.
        """

        output_names = []
        for col in self._cols_:
            if "year" in self.features:
                output_names.append(f"{col}__year")
            if "month" in self.features:
                output_names.append(f"{col}__month")
            if "day" in self.features:
                output_names.append(f"{col}__day")
            if "dayofweek" in self.features:
                output_names.append(f"{col}__dow")
            if "hour" in self.features:
                output_names.append(f"{col}__hour")
        return np.array(output_names)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with standard derived features added.

    Current engineered features:
    - age_bucket: categorical age groups
    - arrival_date_month: integer month extracted from arrival_date
    - arrival_date_weekday / arrival_date_weekend: boolean flags based on arrival weekday
    - arrival_season: category showing Nigerian weather patterns

    The function is intentionally conservative: if required source columns are
    missing, it simply skips those features.
    """

    df2 = df.copy()
    # Age bucket
    if "age" in df2.columns:
        try:
            age_bucket = pd.cut(
                df2["age"],
                bins=[0, 18, 35, 55, 75, np.inf],
                labels=["child", "young_adult", "adult", "middle_aged", "elderly"],
                include_lowest=True
            )
            # Force to plain string/object dtype so downstream imputers/encoders
            # see normal text values instead of a pandas Categorical.
            df2["age_bucket"] = age_bucket.astype(str)
        except Exception:
            pass

    # Arrival-date-based calendar features (using only information available at arrival)
    if "arrival_date" in df2.columns:
        try:
            arrival = pd.to_datetime(df2["arrival_date"], errors="coerce", infer_datetime_format=True)
            df2["arrival_date_month"] = arrival.dt.month
            dow = arrival.dt.weekday
            df2["arrival_date_weekday"] = dow <= 4  # Mon–Fri
            df2["arrival_date_weekend"] = dow >= 5  # Sat–Sun

            # Additional arrival-date-derived features
            # - arrival_dow: integer day-of-week (0=Mon..6=Sun) as categorical text
            # - arrival_part_of_month: 'early' (1–10), 'mid' (11–20), 'late' (21+)
            # - arrival_quarter: calendar quarter (1–4) as text
            df2["arrival_dow"] = dow.astype("Int64").astype(str)

            day = arrival.dt.day
            part = np.where(
                day <= 10,
                "early",
                np.where(day <= 20, "mid", "late"),
            )
            df2["arrival_part_of_month"] = part.astype("object")

            quarter = arrival.dt.quarter.astype("Int64").astype(str)
            df2["arrival_quarter"] = quarter

            # Nigerian season derived from month of arrival.
            # Simple two-season scheme:
            # - Dry: Nov–Mar (11, 12, 1, 2, 3)
            # - Rainy: Apr–Oct (4–10)
            month = df2["arrival_date_month"]
            season = np.where(
                month.isin([11, 12, 1, 2, 3]),
                "dry",
                np.where(month.isin([4, 5, 6, 7, 8, 9, 10]), "rainy", np.nan),
            )
            df2["arrival_season"] = season.astype("object")
        except Exception:
            pass
    return df2

def build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    datetime_cols: Optional[List[str]] = None,
    with_scaling: bool = True,
    ohe_min_frequency: Optional[int] = None,
) -> ColumnTransformer:
    """Create a ColumnTransformer that preprocesses numeric and categorical features.

    - Numeric: median imputation + optional StandardScaler
    - Categorical: most_frequent imputation + OneHotEncoder(handle_unknown='ignore')
    - Datetime: extract calendar features (year, month, day, dow, hour) + impute (most_frequent) + optional scale

    Parameters
    ----------
    numeric_cols : List[str]
        Numeric feature names.
    categorical_cols : List[str]
        Categorical feature names.
    with_scaling : bool
        Whether to scale numeric features. Disable if scale is meaningful.
    ohe_min_frequency : Optional[int]
        If provided, rare categories with count < min_frequency are grouped under '__rare__'.
        If ohe_min_frequency set to zero, OneHotEncoder creates a seperate column for every category
    Returns
    -------
    ColumnTransformer
        A transformer ready to be used as-is or wrapped into a Pipeline with a model.
    """
    # Numeric pipeline: impute then scale
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaling:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=num_steps)

    # Categorical pipeline: impute then one-hot
    ohe_kwargs = {"handle_unknown": "ignore", "sparse_output": False}
    if ohe_min_frequency is not None:
        # scikit-learn supports frequency-based rare category handling
        ohe_kwargs["min_frequency"] = ohe_min_frequency
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(**ohe_kwargs)),
        ]
    )

    # Datetime pipeline: featurize -> impute -> optional scale
    dt_steps: List[Tuple[str, object]] = [("featurize", DatetimeFeaturizer()),
                                          ("imputer", SimpleImputer(strategy="most_frequent"))]
    if with_scaling:
        dt_steps.append(("scaler", StandardScaler()))
    datetime_pipe = Pipeline(steps=dt_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols if len(numeric_cols) > 0 else []),
            ("categorical", cat_pipe, categorical_cols if len(categorical_cols) > 0 else []),
            ("datetime", datetime_pipe, datetime_cols if datetime_cols else []),
        ],
        remainder="drop",  # drop columns not explicitly listed
    )
    return preprocessor

def fit_and_persist_preprocessor(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    features: List[str],
    artifacts_dir: Path,
    fname: str = "preprocessing_pipeline.joblib",
) -> Path:
    """Fit the preprocessor on provided features and persist it.

    Returns the path to the saved artifact.
    """
    X = df[features].copy()
    preprocessor.fit(X)
    artifacts_dir = ensure_dir(artifacts_dir)
    out_path = artifacts_dir / fname
    joblib.dump(preprocessor, out_path)
    return out_path


def save_dataframe_parquet(df: pd.DataFrame, out_path: Path, index: bool = False) -> None:
    """Write a DataFrame to parquet using pyarrow engine.
    """
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=index)


# ------------------------------
# High-level orchestration helpers (optional)
# ------------------------------

def run_eda_suite(
    df: pd.DataFrame,
    reports_dir: Path,
    figures_dir: Path,
    datetime_cols: Optional[List[str]] = None,
    save_figs: bool = True,
    sample_n: Optional[int] = 5000,
    max_cats: int = 30,
) -> None:
    """Run a standard set of EDA tables and plots.

    This function is a convenience so the CLI can invoke one call.
    """
    export_basic_tables(df, reports_dir)
    if save_figs:
        plot_missingness_bars(df, figures_dir)
        plot_numeric_distributions(df, figures_dir, sample_n=sample_n)
        plot_categorical_counts(df, figures_dir, max_cats=max_cats)
        plot_correlation_heatmap(df, figures_dir, sample_n=sample_n)
        if datetime_cols:
            plot_time_series_counts(df, datetime_cols, figures_dir)


def summarize_to_json(data: Dict, out_path: Path) -> None:
    """Utility to persist dictionaries as pretty-printed JSON.
    """
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
