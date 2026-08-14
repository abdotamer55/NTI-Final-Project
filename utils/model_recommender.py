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

def train_recommended_models(df, target, selected_models=None, test_size=0.20, cv_folds=5, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    problem_type = detect_problem_type(y)
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

    results = []
    bundles = {}
    eval_data = {
        "X_test": X_test,
        "y_test": y_test,
        "problem_type": problem_type
    }

    for name, model in candidate_models.items():
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
            
            # Cross validation
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
