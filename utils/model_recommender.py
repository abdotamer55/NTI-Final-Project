import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error, silhouette_score
)
from utils.pipeline_manager import detect_problem_type, build_preprocessor

# Class balancing is optional and only kicks in if imbalanced-learn is
# installed. If it's missing, train_recommended_models raises a clear
# ValueError (instead of crashing on import) so the UI can show install
# instructions.
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


def get_available_models(problem_type):
    if problem_type == "Classification":
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Classifier": SVC(probability=True, random_state=42)
        }
    else:
        return {
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Regressor": SVR()
        }


def get_class_distribution(y):
    """Return a small summary DataFrame of class counts/percentages, plus the imbalance ratio (majority/minority)."""
    counts = y.value_counts()
    dist_df = pd.DataFrame({
        "Class": counts.index.astype(str),
        "Count": counts.values,
        "Percent": (counts.values / counts.sum() * 100).round(2)
    })
    ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
    return dist_df, ratio


def _build_sampler(balance_strategy, y_train):
    """
    Returns (sampler_or_None, warning_message_or_None).
    Falls back to a safer sampler (with a warning) if the requested one
    isn't statistically valid for the given training data.
    """
    if balance_strategy in (None, "none"):
        return None, None

    if not IMBLEARN_AVAILABLE:
        raise ValueError(
            "Class balancing requires the 'imbalanced-learn' package, which isn't installed. "
            "Run: pip install imbalanced-learn"
        )

    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min())

    if balance_strategy == "oversample":
        return RandomOverSampler(random_state=42), None

    if balance_strategy == "undersample":
        if min_class_count < 1:
            return None, "Undersampling skipped: smallest class has 0 samples in the training split."
        return RandomUnderSampler(random_state=42), None

    if balance_strategy == "smote":
        # SMOTE needs at least k_neighbors + 1 samples in the smallest class.
        if min_class_count < 6:
            return RandomOverSampler(random_state=42), (
                f"SMOTE needs at least 6 samples in the smallest class (found {min_class_count}). "
                "Used Random Oversampling instead."
            )
        k = min(5, min_class_count - 1)
        return SMOTE(random_state=42, k_neighbors=k), None

    raise ValueError(f"Unknown balance_strategy '{balance_strategy}'.")


def train_recommended_models(df, target, selected_models=None, test_size=0.20, cv_folds=5,
                              random_state=42, balance_strategy="none"):
    """
    Same return signature as before: (problem_type, results, bundles, feature_names, eval_data).
    eval_data now also carries an optional "balance_info" key when
    balance_strategy != "none":
        {"strategy": str, "applied": bool, "warning": str | None, "distribution_before": DataFrame}
    """
    X = df.drop(columns=[target])
    y = df[target]

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    problem_type = detect_problem_type(y)

    if balance_strategy not in (None, "none") and problem_type != "Classification":
        raise ValueError("Class balancing only applies to classification targets.")

    all_models = get_available_models(problem_type)

    if selected_models:
        candidate_models = {k: v for k, v in all_models.items() if k in selected_models}
    else:
        candidate_models = all_models

    stratify = None
    if problem_type == "Classification" and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    preprocessor = build_preprocessor(X)

    # Balancing is resolved once here (based on the training split only) and
    # reused for every candidate model below, so all models are compared on
    # a level footing.
    balance_info = None
    sampler_template = None
    if balance_strategy not in (None, "none"):
        sampler_template, warning = _build_sampler(balance_strategy, y_train)
        dist_before, ratio_before = get_class_distribution(y_train)
        balance_info = {
            "strategy": balance_strategy,
            "applied": sampler_template is not None,
            "warning": warning,
            "distribution_before": dist_before,
            "imbalance_ratio_before": ratio_before,
        }

    results = []
    bundles = {}
    eval_data = {
        "X_test": X_test,
        "y_test": y_test,
        "problem_type": problem_type
    }
    if balance_info is not None:
        eval_data["balance_info"] = balance_info

    for name, model in candidate_models.items():
        if sampler_template is not None:
            # sklearn's cross_val_score clones each pipeline step per fold,
            # so passing the same sampler instance is safe here.
            pipeline = ImbPipeline([
                ("preprocessor", preprocessor),
                ("sampler", sampler_template),
                ("model", model)
            ])
        else:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if problem_type == "Classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)

            # Cross validation (the sampler, if any, is refit within each
            # fold automatically since it's part of the pipeline).
            try:
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring="accuracy")
                cv_mean = float(cv_scores.mean())
            except Exception:
                cv_mean = acc

            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1 Score": f1,
                "Precision": prec,
                "Recall": rec,
                "CV Accuracy": cv_mean
            })
        else:
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = float(np.sqrt(mse))

            try:
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring="r2")
                cv_mean = float(cv_scores.mean())
            except Exception:
                cv_mean = r2

            results.append({
                "Model": name,
                "R2 Score": r2,
                "MAE": mae,
                "RMSE": rmse,
                "CV R2": cv_mean
            })

        bundles[name] = pipeline

    return problem_type, results, bundles, X.columns.tolist(), eval_data


def train_kmeans_clustering(df, n_clusters=3, features=None):
    if features is None:
        features = df.select_dtypes(include=np.number).columns.tolist()

    X = df[features].dropna()
    if len(X) < n_clusters:
        return None, None

    preprocessor = build_preprocessor(X)
    X_trans = preprocessor.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_trans)

    sil_score = float(silhouette_score(X_trans, labels)) if len(np.unique(labels)) > 1 else 0.0

    # 2D PCA for visual representation
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_trans)

    cluster_df = X.copy()
    cluster_df["Cluster"] = [f"Cluster {l}" for l in labels]
    cluster_df["PCA1"] = pca_coords[:, 0]
    cluster_df["PCA2"] = pca_coords[:, 1]

    return kmeans, {
        "silhouette_score": sil_score,
        "cluster_df": cluster_df,
        "features": features
    }
