import pandas as pd
import numpy as np

def validate_prediction_columns(df, required_features):
    return [feature for feature in required_features if feature not in df.columns]

def predict_batch(pipeline, df, features):
    return pipeline.predict(df[features])

def make_single_row(values, features, source_df=None):
    """
    Build a single-row DataFrame for prediction.

    Parameters
    ----------
    values : dict
        Feature name -> raw value collected from the UI.
    features : list of str
        Feature columns expected by the pipeline, in order.
    source_df : pd.DataFrame, optional
        The original training/reference dataset. When provided, each
        feature's dtype is taken from this DataFrame instead of being
        guessed from the string content of the value. This avoids
        corrupting categorical values that merely look numeric
        (e.g. zip codes like "00501", or coded IDs like "007"), which
        would otherwise lose meaning (or silently become "unknown"
        categories) if coerced to int/float.
    """
    formatted = {}
    for feature in features:
        val = values.get(feature)

        target_dtype = None
        if source_df is not None and feature in source_df.columns:
            target_dtype = source_df[feature].dtype

        if target_dtype is not None:
            if pd.api.types.is_numeric_dtype(target_dtype):
                try:
                    formatted[feature] = float(val) if val is not None else np.nan
                except (ValueError, TypeError):
                    formatted[feature] = np.nan
            else:
                # Categorical / object / string column: preserve as-is,
                # do NOT attempt numeric coercion even if it looks numeric.
                formatted[feature] = str(val)
        else:
            # No reference dtype available: fall back to best-effort
            # inference, but only coerce when the value is already a
            # real number, never guess from string shape.
            if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                formatted[feature] = val
            else:
                formatted[feature] = str(val)

    return pd.DataFrame([formatted])
