import numpy as np
import pandas as pd

def analyze_dataset(df):
    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = [c for c in df.columns if c not in numeric]
    cardinality = df.nunique(dropna=True)
    missing_counts = df.isna().sum()
    missing = (df.isna().mean() * 100).sort_values(ascending=False)

    id_columns = [
        c for c in df.columns
        if c.lower().endswith(("id", "_id", "code", "index"))
        or (cardinality[c] >= max(len(df) * 0.95, 1) and len(df) > 10)
    ]

    constant_columns = [
        c for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]

    skewness = {}
    outliers = {}
    for c in numeric:
        col_data = df[c].dropna()
        if len(col_data) > 2:
            skewness[c] = float(col_data.skew())
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_cnt = int(((col_data < lower) | (col_data > upper)).sum())
                if outlier_cnt > 0:
                    outliers[c] = outlier_cnt

    high_cardinality = [
        c for c in categorical
        if cardinality[c] > max(20, len(df) * 0.05)
    ]

    # Highly correlated feature pairs
    high_correlation = []
    if len(numeric) >= 2:
        corr_matrix = df[numeric].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                val = corr_matrix.iloc[i, j]
                if val >= 0.85:
                    high_correlation.append((numeric[i], numeric[j], float(val)))

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_counts": missing_counts.to_dict(),
        "missing_percent": missing.to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "cardinality": cardinality.to_dict(),
        "high_cardinality": high_cardinality,
        "id_columns": id_columns,
        "constant_columns": constant_columns,
        "skewness": skewness,
        "outliers": outliers,
        "high_correlation": high_correlation,
    }
