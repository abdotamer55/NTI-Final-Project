import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def fill_missing_recommended(df):
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
    out = df.copy()
    if column not in out.columns:
        return out
    
    if strategy == "median":
        val = out[column].median()
        out[column] = out[column].fillna(val)
    elif strategy == "mean":
        val = out[column].mean()
        out[column] = out[column].fillna(val)
    elif strategy == "mode":
        mode = out[column].mode(dropna=True)
        val = mode.iloc[0] if not mode.empty else "Unknown"
        out[column] = out[column].fillna(val)
    elif strategy == "constant":
        out[column] = out[column].fillna(fill_value if fill_value is not None else "Missing")
    elif strategy == "drop_rows":
        out = out.dropna(subset=[column])
    return out

def remove_duplicates(df):
    return df.drop_duplicates().copy()

def drop_columns(df, cols_to_drop):
    cols = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols).copy()

def drop_constant_columns(df):
    columns = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=columns).copy(), columns

def cap_outliers_iqr(df, column, factor=1.5):
    out = df.copy()
    if column not in out.columns or not pd.api.types.is_numeric_dtype(out[column]):
        return out
    
    col_data = out[column].dropna()
    if len(col_data) < 4:
        return out

    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    out[column] = out[column].clip(lower=lower_bound, upper=upper_bound)
    return out

def cap_outliers_zscore(df, column, threshold=3.0):
    out = df.copy()
    if column not in out.columns or not pd.api.types.is_numeric_dtype(out[column]):
        return out

    col_data = out[column].dropna()
    mean = col_data.mean()
    std = col_data.std()
    if std == 0 or np.isnan(std):
        return out

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    out[column] = out[column].clip(lower=lower_bound, upper=upper_bound)
    return out

def apply_log1p(df, column):
    out = df.copy()
    if column in out.columns and pd.api.types.is_numeric_dtype(out[column]):
        out[column] = np.log1p(out[column].clip(lower=0))
    return out

def encode_categorical(df, columns, method="onehot", max_onehot_categories=15):
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
            
    return out

def scale_features(df, columns, method="standard"):
    out = df.copy()
    target_cols = [c for c in columns if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    if not target_cols:
        return out

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        return out

    out[target_cols] = scaler.fit_transform(out[target_cols].fillna(out[target_cols].median()))
    return out

def cast_column_type(df, column, new_type):
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
    except Exception:
        pass
    return out
