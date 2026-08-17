"""
preprocessing_utils.py

Generic, dataset-agnostic preprocessing & feature engineering utilities for the Preprocessing Studio.

IMPORTANT: All function signatures (inputs AND return types) are kept
identical to the original version, so this file is a drop-in replacement
and will not break any other page/module that already imports it.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #

def fill_missing_recommended(df):
    """Auto-impute every column: median for numeric, mode for categorical."""
    out = df.copy()
    for column in out.columns:
        if not out[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].fillna(out[column].median())
        else:
            mode = out[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            out[column] = out[column].fillna(fill_value)
    return out


def impute_column(df, column, strategy="median", fill_value=None):
    """
    Impute a single column. Returns the imputed DataFrame (same signature
    as before). Raises ValueError on invalid input instead of silently
    corrupting the column's dtype.
    """
    out = df.copy()
    if column not in out.columns:
        return out

    is_numeric = pd.api.types.is_numeric_dtype(out[column])

    if strategy == "median":
        if not is_numeric:
            raise ValueError(f"Median is only valid for numeric columns ('{column}' is not numeric).")
        out[column] = out[column].fillna(out[column].median())

    elif strategy == "mean":
        if not is_numeric:
            raise ValueError(f"Mean is only valid for numeric columns ('{column}' is not numeric).")
        out[column] = out[column].fillna(out[column].mean())

    elif strategy == "mode":
        mode = out[column].mode(dropna=True)
        val = mode.iloc[0] if not mode.empty else "Unknown"
        out[column] = out[column].fillna(val)

    elif strategy == "constant":
        val = fill_value if fill_value is not None else "Missing"
        if is_numeric:
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"'{val}' is not a valid number for numeric column '{column}'. "
                    "Enter a numeric fill value, or cast the column to text first."
                )
        out[column] = out[column].fillna(val)

    elif strategy == "drop_rows":
        out = out.dropna(subset=[column])

    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    return out


# --------------------------------------------------------------------------- #
# Row / column cleanup
# --------------------------------------------------------------------------- #

def remove_duplicates(df):
    return df.drop_duplicates().copy()


def drop_columns(df, cols_to_drop):
    cols = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols).copy()


def drop_constant_columns(df):
    columns = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=columns).copy(), columns


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #

def cap_outliers_iqr(df, column, factor=1.5):
    """Cap outliers in a single numeric column using the IQR rule."""
    out = df.copy()
    if column not in out.columns or not pd.api.types.is_numeric_dtype(out[column]):
        return out

    col_data = out[column].dropna()
    if len(col_data) < 4:
        return out

    q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    out[column] = out[column].clip(lower=lower, upper=upper)
    return out


def cap_outliers_zscore(df, column, threshold=3.0):
    """Cap outliers in a single numeric column using the Z-score rule."""
    out = df.copy()
    if column not in out.columns or not pd.api.types.is_numeric_dtype(out[column]):
        return out

    col_data = out[column].dropna()
    mean, std = col_data.mean(), col_data.std()
    if std == 0 or np.isnan(std):
        return out

    lower, upper = mean - threshold * std, mean + threshold * std
    out[column] = out[column].clip(lower=lower, upper=upper)
    return out


def apply_log1p(df, column):
    out = df.copy()
    if column in out.columns and pd.api.types.is_numeric_dtype(out[column]):
        out[column] = np.log1p(out[column].clip(lower=0))
    return out


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def encode_categorical(df, columns, method="onehot", max_onehot_categories=15):
    """Encode categorical columns."""
    out = df.copy()
    target_cols = [c for c in columns if c in out.columns]

    if method == "onehot":
        valid_onehot = [c for c in target_cols if out[c].nunique() <= max_onehot_categories]
        if valid_onehot:
            out = pd.get_dummies(out, columns=valid_onehot, drop_first=True, dtype=int)
    elif method == "label":
        for col in target_cols:
            out[col] = out[col].astype("category").cat.codes
    elif method == "frequency":
        for col in target_cols:
            freq = out[col].value_counts(normalize=True)
            out[col] = out[col].map(freq).fillna(0)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return out


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #

def scale_features(df, columns, method="standard"):
    """Scale numeric columns."""
    out = df.copy()
    target_cols = [c for c in columns if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    if not target_cols:
        return out

    cols_with_na = [c for c in target_cols if out[c].isna().any()]
    if cols_with_na:
        raise ValueError(
            f"Cannot scale columns with missing values: {cols_with_na}. "
            "Handle missing values first (Missing Values tab), then scale."
        )

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method '{method}'.")

    out[target_cols] = scaler.fit_transform(out[target_cols])
    return out


# --------------------------------------------------------------------------- #
# Type casting
# --------------------------------------------------------------------------- #

def cast_column_type(df, column, new_type):
    """Cast a column to a new dtype."""
    out = df.copy()
    if column not in out.columns:
        return out

    try:
        if new_type == "numeric":
            out[column] = pd.to_numeric(out[column], errors="coerce")
        elif new_type == "datetime":
            out[column] = pd.to_datetime(out[column], errors="coerce")
        elif new_type == "category":
            out[column] = out[column].astype("category")
        elif new_type == "string":
            out[column] = out[column].astype(str)
        else:
            raise ValueError(f"Unknown type '{new_type}'.")
    except Exception as e:
        raise ValueError(f"Could not cast '{column}' to {new_type}: {e}")

    return out


# --------------------------------------------------------------------------- #
# Feature Engineering & Recommendation Engine
# --------------------------------------------------------------------------- #

def recommend_features(df, target_col=None):
    """
    Analyzes any tabular dataset and generates intelligent, dataset-agnostic
    feature engineering recommendations based on dtypes, distributions, skewness,
    correlations, and cardinalities.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    target_col : str, optional
        The target column name (if known). Excluded from feature generation source.

    Returns
    -------
    list of dict
        List of recommendation objects with keys:
        'type', 'columns', 'title', 'description', 'priority', 'action', 'params'
    """
    recommendations = []
    if df is None or df.empty:
        return recommendations

    cols = [c for c in df.columns if c != target_col]

    # 1. Datetime feature extraction
    for col in cols:
        is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
        if not is_dt and (df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])):
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ['date', 'time', 'dt', 'timestamp', 'created', 'updated', 'dob']):
                try:
                    sample = pd.to_datetime(df[col].dropna().head(20), errors='coerce')
                    if sample.notna().mean() > 0.7:
                        is_dt = True
                except Exception:
                    pass
        if is_dt:
            recommendations.append({
                "type": "datetime",
                "columns": [col],
                "title": f"Extract Datetime Components from '{col}'",
                "description": f"Column '{col}' contains temporal data. Extracting Year, Month, Day, DayOfWeek, and Is_Weekend reveals seasonality and period patterns.",
                "priority": "High",
                "action": "create_datetime_features",
                "params": {"column": col}
            })

    # 2. Skewness / Log Transformations
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    for col in num_cols:
        valid = df[col].dropna()
        if len(valid) > 10:
            skew_val = valid.skew()
            if skew_val > 1.0:
                prio = "High" if skew_val > 2.0 else "Medium"
                recommendations.append({
                    "type": "log_transform",
                    "columns": [col],
                    "title": f"Log Transformation on Skewed Column '{col}'",
                    "description": f"'{col}' is heavily right-skewed (skewness = {skew_val:.2f}). Applying a logarithmic transform normalizes distribution for linear models and distance metrics.",
                    "priority": prio,
                    "action": "create_log_transform",
                    "params": {"column": col, "method": "log1p"}
                })

    # 3. Numeric Interaction / Ratios
    if len(num_cols) >= 2:
        try:
            corr_matrix = df[num_cols].corr().abs()
            pairs_added = 0
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    corr_val = corr_matrix.loc[c1, c2] if (c1 in corr_matrix.index and c2 in corr_matrix.columns) else 0
                    if 0.3 <= corr_val <= 0.85 and pairs_added < 3:
                        recommendations.append({
                            "type": "interaction",
                            "columns": [c1, c2],
                            "title": f"Interaction Feature: '{c1}' × '{c2}'",
                            "description": f"'{c1}' and '{c2}' show a moderate correlation ({corr_val:.2f}). Creating product interaction or ratio features can capture joint signals.",
                            "priority": "Medium",
                            "action": "create_interaction_features",
                            "params": {"col1": c1, "col2": c2, "operation": "multiply"}
                        })
                        pairs_added += 1
        except Exception:
            pass

    # 4. Group Aggregation (Categorical + Numerical)
    cat_cols = [c for c in cols if df[c].nunique() >= 2 and df[c].nunique() <= 30 and (not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 10)]
    if cat_cols and num_cols:
        added_grp = 0
        for cat in cat_cols:
            for num in num_cols:
                if cat != num and added_grp < 3:
                    recommendations.append({
                        "type": "group_aggregate",
                        "columns": [cat, num],
                        "title": f"Group Aggregates: Mean of '{num}' by '{cat}'",
                        "description": f"Aggregating numerical feature '{num}' across categorical groups in '{cat}' captures segment-level reference baselines.",
                        "priority": "Medium",
                        "action": "create_group_aggregate_features",
                        "params": {"group_col": cat, "target_col": num, "agg_funcs": ["mean", "std"]}
                    })
                    added_grp += 1

    # 5. Binning / Discretization
    for col in num_cols:
        if df[col].nunique() > 20:
            recommendations.append({
                "type": "binning",
                "columns": [col],
                "title": f"Quantile Binning on '{col}'",
                "description": f"'{col}' is continuous ({df[col].nunique()} distinct values). Converting into quantile bins helps decision trees and linear models handle non-linear thresholds.",
                "priority": "Low",
                "action": "create_binned_features",
                "params": {"column": col, "num_bins": 4, "strategy": "quantile"}
            })
            if len([r for r in recommendations if r["type"] == "binning"]) >= 2:
                break

    # 6. Text Statistics
    text_cols = [c for c in cols if (df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c])) and df[c].nunique() > 10]
    for col in text_cols:
        sample_str = df[col].dropna().astype(str)
        avg_len = sample_str.str.len().mean() if not sample_str.empty else 0
        if avg_len > 15:
            recommendations.append({
                "type": "text_stats",
                "columns": [col],
                "title": f"Text Statistics for '{col}'",
                "description": f"'{col}' contains textual data (avg length {avg_len:.1f} chars). Extracting character length and word count creates numeric indicators.",
                "priority": "Medium",
                "action": "create_text_stats_features",
                "params": {"column": col}
            })

    # 7. Frequency Encoding
    high_card_cols = [c for c in cols if df[c].nunique() > 15 and (df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c]))]
    for col in high_card_cols:
        recommendations.append({
            "type": "frequency_encode",
            "columns": [col],
            "title": f"Frequency Encoding for High-Cardinality '{col}'",
            "description": f"'{col}' has {df[col].nunique()} unique categories. Frequency encoding replaces categories with their percentage frequency.",
            "priority": "High",
            "action": "create_frequency_encoding",
            "params": {"column": col}
        })

    return recommendations


def create_datetime_features(df, column, features=None):
    """Extract datetime component features from a column."""
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    dt_series = pd.to_datetime(out[column], errors='coerce')
    if dt_series.isna().all():
        raise ValueError(f"Column '{column}' could not be parsed as valid dates/times.")

    if features is None:
        features = ["year", "month", "day", "dayofweek", "is_weekend"]

    if "year" in features:
        out[f"{column}_year"] = dt_series.dt.year
    if "month" in features:
        out[f"{column}_month"] = dt_series.dt.month
    if "day" in features:
        out[f"{column}_day"] = dt_series.dt.day
    if "dayofweek" in features:
        out[f"{column}_dayofweek"] = dt_series.dt.dayofweek
    if "is_weekend" in features:
        out[f"{column}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
    if "quarter" in features:
        out[f"{column}_quarter"] = dt_series.dt.quarter
    if "hour" in features:
        out[f"{column}_hour"] = dt_series.dt.hour

    return out


def create_interaction_features(df, col1, col2, operation="multiply"):
    """Create mathematical interaction feature between two numeric columns."""
    out = df.copy()
    for c in [col1, col2]:
        if c not in out.columns:
            raise ValueError(f"Column '{c}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(out[c]):
            raise ValueError(f"Column '{c}' must be numeric for interaction features.")

    if operation == "multiply":
        new_col = f"{col1}_x_{col2}"
        out[new_col] = out[col1] * out[col2]
    elif operation == "divide":
        new_col = f"{col1}_div_{col2}"
        denom = out[col2].replace(0, np.nan)
        out[new_col] = out[col1] / denom
    elif operation == "add":
        new_col = f"{col1}_plus_{col2}"
        out[new_col] = out[col1] + out[col2]
    elif operation == "subtract":
        new_col = f"{col1}_minus_{col2}"
        out[new_col] = out[col1] - out[col2]
    else:
        raise ValueError(f"Unknown interaction operation '{operation}'. Use multiply, divide, add, or subtract.")

    return out


def create_binned_features(df, column, num_bins=4, strategy="quantile", labels=None):
    """Discretize continuous column into bins."""
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"Column '{column}' not in dataset.")
    if not pd.api.types.is_numeric_dtype(out[column]):
        raise ValueError(f"Column '{column}' is not numeric.")

    new_col = f"{column}_binned"
    if strategy == "quantile":
        out[new_col] = pd.qcut(out[column], q=num_bins, labels=labels, duplicates='drop')
    elif strategy == "uniform":
        out[new_col] = pd.cut(out[column], bins=num_bins, labels=labels)
    else:
        raise ValueError(f"Unknown binning strategy '{strategy}'. Use quantile or uniform.")

    return out


def create_log_transform(df, column, method="log1p"):
    """Apply logarithmic or power transformation to reduce skewness."""
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
    if not pd.api.types.is_numeric_dtype(out[column]):
        raise ValueError(f"Column '{column}' is not numeric.")

    new_col = f"{column}_{method}"
    if method == "log1p":
        min_val = out[column].min()
        if min_val < 0:
            out[new_col] = np.log1p(out[column] - min_val)
        else:
            out[new_col] = np.log1p(out[column])
    elif method == "sqrt":
        min_val = out[column].min()
        if min_val < 0:
            out[new_col] = np.sqrt(out[column] - min_val)
        else:
            out[new_col] = np.sqrt(out[column])
    else:
        raise ValueError(f"Unknown transformation method '{method}'. Use log1p or sqrt.")

    return out


def create_group_aggregate_features(df, group_col, target_col, agg_funcs=None):
    """Group by categorical column and compute aggregate statistics for target numerical column."""
    out = df.copy()
    if group_col not in out.columns or target_col not in out.columns:
        raise ValueError(f"Columns '{group_col}' or '{target_col}' not found.")
    if not pd.api.types.is_numeric_dtype(out[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric.")

    if agg_funcs is None:
        agg_funcs = ["mean", "std"]

    for func in agg_funcs:
        agg_series = out.groupby(group_col)[target_col].transform(func)
        new_col = f"{target_col}_by_{group_col}_{func}"
        out[new_col] = agg_series

    return out


def create_text_stats_features(df, column):
    """Extract string length and word count statistics from text column."""
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    str_s = out[column].astype(str).fillna("")
    out[f"{column}_len"] = str_s.str.len()
    out[f"{column}_word_count"] = str_s.str.split().str.len()
    return out


def create_frequency_encoding(df, column):
    """Create frequency encoding for categorical feature."""
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    freq_map = out[column].value_counts(normalize=True).to_dict()
    out[f"{column}_freq"] = out[column].map(freq_map).fillna(0)
    return out


def auto_generate_recommended_features(df, target_col=None, max_features=5):
    """
    Automatically recommends and generates top N features on any dataset.
    Returns (updated_df, applied_summary_list)
    """
    recs = recommend_features(df, target_col=target_col)
    out = df.copy()
    applied = []

    for rec in recs[:max_features]:
        action = rec.get("action")
        params = rec.get("params", {})
        try:
            if action == "create_datetime_features":
                out = create_datetime_features(out, **params)
                applied.append(rec["title"])
            elif action == "create_log_transform":
                out = create_log_transform(out, **params)
                applied.append(rec["title"])
            elif action == "create_interaction_features":
                out = create_interaction_features(out, **params)
                applied.append(rec["title"])
            elif action == "create_group_aggregate_features":
                out = create_group_aggregate_features(out, **params)
                applied.append(rec["title"])
            elif action == "create_binned_features":
                out = create_binned_features(out, **params)
                applied.append(rec["title"])
            elif action == "create_text_stats_features":
                out = create_text_stats_features(out, **params)
                applied.append(rec["title"])
            elif action == "create_frequency_encoding":
                out = create_frequency_encoding(out, **params)
                applied.append(rec["title"])
        except Exception as e:
            pass

    return out, applied
