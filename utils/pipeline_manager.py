import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def detect_problem_type(y):
    if not pd.api.types.is_numeric_dtype(y):
        return "Classification"

    unique = y.nunique(dropna=True)
    return "Classification" if unique <= 20 and unique / max(len(y), 1) < .05 else "Regression"

def build_preprocessor(X):
    numerical = X.select_dtypes(include="number").columns.tolist()
    all_categorical = [c for c in X.columns if c not in numerical]

    # Separate low-cardinality and high-cardinality categorical features to prevent MemoryError
    low_card_categorical = [c for c in all_categorical if X[c].nunique(dropna=True) <= 20]
    high_card_categorical = [c for c in all_categorical if c not in low_card_categorical]

    transformers = []

    if numerical:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numerical", numerical_pipeline, numerical))

    if low_card_categorical:
        low_card_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False))
        ])
        transformers.append(("low_card_cat", low_card_pipeline, low_card_categorical))

    if high_card_categorical:
        high_card_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
        transformers.append(("high_card_cat", high_card_pipeline, high_card_categorical))

    if not transformers:
        return ColumnTransformer([("passthrough", "passthrough", X.columns.tolist())])

    return ColumnTransformer(transformers=transformers)
