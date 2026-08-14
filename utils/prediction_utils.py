import pandas as pd
import numpy as np

def validate_prediction_columns(df, required_features):
    return [feature for feature in required_features if feature not in df.columns]

def predict_batch(pipeline, df, features):
    return pipeline.predict(df[features])

def make_single_row(values, features):
    formatted = {}
    for feature in features:
        val = values.get(feature)
        if isinstance(val, (int, float, np.number)):
            formatted[feature] = val
        else:
            try:
                if "." in str(val):
                    formatted[feature] = float(val)
                else:
                    formatted[feature] = int(val)
            except (ValueError, TypeError):
                formatted[feature] = str(val)
    return pd.DataFrame([formatted])
