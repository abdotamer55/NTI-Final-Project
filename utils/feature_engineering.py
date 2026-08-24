"""
utils/feature_engineering.py

Modular Feature Engineering Engine for DataPilot AI.

This module implements an automated, performance-guided feature engineering recommendation engine:
1. Baseline Model Evaluation (Cross-Validation score before new features)
2. Feature Importance Analysis (Ranks top numeric features via tree models)
3. Candidate Feature Generation (Log1p, Polynomial, Interaction, Ratio, Binning, Datetime)
4. Feature Impact Testing (Evaluates CV performance of each candidate against baseline)
5. Feature Validation & Ranking (Filters for positive improvement >= MIN_IMPROVEMENT)
6. Recommendation Construction & Application (Generates recommendations & applies selected features)

No external APIs, LLMs, or generative AI models are used. Uses Python, Pandas, NumPy, and Scikit-Learn only.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

from utils.pipeline_manager import detect_problem_type, build_preprocessor

# Configuration Defaults
MAX_TOP_FEATURES = 10
MAX_INTERACTION_FEATURES = 20
MAX_RATIO_FEATURES = 20
CV_FOLDS = 5
MIN_IMPROVEMENT = 0.005
MAX_EVALUATION_ROWS = 10000


def evaluate_baseline(df, target_col, problem_type=None, cv=CV_FOLDS, random_state=42):
    """
    Train and evaluate a baseline model before adding new engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and target.
    target_col : str
        Target variable column name.
    problem_type : str, optional
        'Classification' or 'Regression'. Auto-detected if None.
    cv : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Baseline scores containing primary score, metrics (Accuracy/F1 or R2/RMSE), and problem type.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df_clean = df.dropna(subset=[target_col]).copy()
    if len(df_clean) == 0:
        raise ValueError(f"Target column '{target_col}' contains only missing values.")

    # Downsample large datasets for speed during candidate evaluation
    if len(df_clean) > MAX_EVALUATION_ROWS:
        df_eval = df_clean.sample(n=MAX_EVALUATION_ROWS, random_state=random_state).copy()
    else:
        df_eval = df_clean

    X = df_eval.drop(columns=[target_col])
    y = df_eval[target_col]

    if problem_type is None:
        problem_type = detect_problem_type(y)

    if problem_type == "Classification" and y.nunique() < 2:
        raise ValueError(
            f"Target column '{target_col}' has only one distinct class after removing missing values. "
            "Cross-validation requires at least 2 classes."
        )

    n_samples = len(y)
    actual_cv = max(2, min(cv, n_samples))

    preprocessor = build_preprocessor(X)

    if problem_type == "Classification":
        model = ExtraTreesClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Determine if StratifiedKFold can be used
        min_class_cnt = y.value_counts().min() if y.nunique() > 0 else 0
        if min_class_cnt >= actual_cv:
            splitter = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
        else:
            splitter = KFold(n_splits=actual_cv, shuffle=True, random_state=random_state)

        scores = cross_validate(pipeline, X, y, cv=splitter, scoring=["accuracy", "f1_weighted"], error_score="raise")
        acc = float(np.mean(scores["test_accuracy"]))
        f1 = float(np.mean(scores["test_f1_weighted"]))

        return {
            "problem_type": "Classification",
            "primary_score": acc,
            "accuracy": acc,
            "f1": f1,
            "r2": None,
            "rmse": None,
            "cv_used": actual_cv,
            "eval_rows": len(X),
        }
    else:
        model = ExtraTreesRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        splitter = KFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
        scores = cross_validate(
            pipeline, X, y, cv=splitter,
            scoring=["r2", "neg_root_mean_squared_error"],
            error_score="raise"
        )
        r2_val = float(np.mean(scores["test_r2"]))
        rmse_val = float(-np.mean(scores["test_neg_root_mean_squared_error"]))

        return {
            "problem_type": "Regression",
            "primary_score": r2_val,
            "accuracy": None,
            "f1": None,
            "r2": r2_val,
            "rmse": rmse_val,
            "cv_used": actual_cv,
            "eval_rows": len(X),
        }


def get_feature_importance(df, target_col, problem_type=None, top_n=MAX_TOP_FEATURES, random_state=42):
    """
    Calculate feature importances of original features using a tree-based model.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Target variable.
    problem_type : str, optional
        'Classification' or 'Regression'.
    top_n : int, default=10
        Number of top important features to return.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'Importance'] sorted by importance descending.
    """
    if target_col not in df.columns:
        return pd.DataFrame(columns=["Feature", "Importance"])

    df_clean = df.dropna(subset=[target_col]).copy()
    if len(df_clean) > MAX_EVALUATION_ROWS:
        df_eval = df_clean.sample(n=MAX_EVALUATION_ROWS, random_state=random_state)
    else:
        df_eval = df_clean

    X = df_eval.drop(columns=[target_col])
    y = df_eval[target_col]

    if X.empty:
        return pd.DataFrame(columns=["Feature", "Importance"])

    if problem_type is None:
        problem_type = detect_problem_type(y)

    preprocessor = build_preprocessor(X)
    X_trans = preprocessor.fit_transform(X)

    if problem_type == "Classification":
        model = ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        model = ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

    model.fit(X_trans, y)

    # Get feature names from ColumnTransformer
    try:
        trans_feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback if get_feature_names_out fails
        trans_feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

    # Map transformed feature importances back to original features in X.
    #
    # sklearn's ColumnTransformer names transformed columns as either:
    #   "<transformer>__<column>"                 (e.g. numeric passthrough)
    #   "<transformer>__<column>_<category>"       (e.g. one-hot encoded)
    # Matching must respect this exact "__" / "_" boundary — a naive
    # substring check (e.g. `"age" in name`) would wrongly match unrelated
    # columns like "wage_scaled" or "electricity_bill", silently corrupting
    # every ranking that depends on this function.
    orig_features = X.columns.tolist()
    importance_map = {col: 0.0 for col in orig_features}
    # Longest-name-first so e.g. "income_level" is preferred over "income"
    # when both could technically match the same transformed name.
    sorted_features = sorted(orig_features, key=len, reverse=True)

    raw_importances = model.feature_importances_

    for name, imp in zip(trans_feature_names, raw_importances):
        remainder = name.split("__", 1)[1] if "__" in name else name
        matched_col = None
        for col in sorted_features:
            if remainder == col or remainder.startswith(f"{col}_"):
                matched_col = col
                break
        if matched_col is not None:
            importance_map[matched_col] += imp

    imp_df = pd.DataFrame([
        {"Feature": col, "Importance": importance_map[col]}
        for col in orig_features
    ]).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return imp_df


def generate_log_features(df, important_num_cols, target_col=None):
    """
    Generate log1p candidate features for non-negative, right-skewed numeric columns.

    Returns
    -------
    list of dict
    """
    candidates = []
    for col in important_num_cols:
        if col == target_col or col not in df.columns:
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        valid = series.dropna()
        if len(valid) == 0 or (valid < 0).any():
            continue

        skew_val = valid.skew()
        if abs(skew_val) > 0.75:
            new_series = np.log1p(series)
            if not np.isinf(new_series).any() and not new_series.isna().all():
                candidates.append({
                    "feature_name": f"{col}_log",
                    "feature_type": "Log Transformation",
                    "source_columns": [col],
                    "action_type": "apply_log_feature",
                    "action_label": f"Apply {col}_log Feature",
                    "series": new_series,
                })
    return candidates


def generate_polynomial_features(df, important_num_cols, target_col=None, degree=2):
    """
    Generate squared polynomial candidate features for important numeric columns.

    Returns
    -------
    list of dict
    """
    candidates = []
    for col in important_num_cols:
        if col == target_col or col not in df.columns:
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        new_series = series ** degree
        if not np.isinf(new_series).any() and not new_series.isna().all() and new_series.nunique() > 1:
            candidates.append({
                "feature_name": f"{col}_squared",
                "feature_type": "Polynomial Feature",
                "source_columns": [col],
                "action_type": "apply_polynomial_feature",
                "action_label": f"Apply {col}_squared Feature",
                "series": new_series,
            })
    return candidates


def generate_interaction_features(df, important_num_cols, target_col=None, max_features=MAX_INTERACTION_FEATURES):
    """
    Generate interaction (product) candidate features between pairs of important numeric columns.

    Returns
    -------
    list of dict
    """
    candidates = []
    num_cols = [c for c in important_num_cols if c != target_col and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    count = 0
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            if count >= max_features:
                break
            col1, col2 = num_cols[i], num_cols[j]
            new_series = df[col1] * df[col2]
            if not np.isinf(new_series).any() and not new_series.isna().all() and new_series.nunique() > 1:
                candidates.append({
                    "feature_name": f"{col1}_x_{col2}",
                    "feature_type": "Interaction Feature",
                    "source_columns": [col1, col2],
                    "action_type": "apply_interaction_feature",
                    "action_label": f"Apply {col1}_x_{col2} Feature",
                    "series": new_series,
                })
                count += 1
        if count >= max_features:
            break
    return candidates


def generate_ratio_features(df, important_num_cols, target_col=None, max_features=MAX_RATIO_FEATURES):
    """
    Generate ratio candidate features (A / B) between important numeric columns safely.

    Returns
    -------
    list of dict
    """
    candidates = []
    num_cols = [c for c in important_num_cols if c != target_col and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    count = 0
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            if i == j or count >= max_features:
                continue
            col1, col2 = num_cols[i], num_cols[j]

            denom = df[col2].replace(0, np.nan)
            ratio_series = df[col1] / denom
            ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan)

            if ratio_series.notna().sum() > 0.5 * len(df) and ratio_series.nunique() > 1:
                candidates.append({
                    "feature_name": f"{col1}_per_{col2}",
                    "feature_type": "Ratio Feature",
                    "source_columns": [col1, col2],
                    "action_type": "apply_ratio_feature",
                    "action_label": f"Apply {col1}_per_{col2} Feature",
                    "series": ratio_series,
                })
                count += 1
        if count >= max_features:
            break
    return candidates


def generate_binning_features(df, important_num_cols, target_col=None, q=4):
    """
    Generate quantile-based binning candidate features for numeric columns.

    Returns
    -------
    list of dict
    """
    candidates = []
    for col in important_num_cols:
        if col == target_col or col not in df.columns:
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= q:
            continue

        try:
            binned_series = pd.qcut(series, q=q, labels=False, duplicates="drop")
            if binned_series.nunique() > 1:
                candidates.append({
                    "feature_name": f"{col}_Binned",
                    "feature_type": "Binning Feature",
                    "source_columns": [col],
                    "action_type": "apply_binning_feature",
                    "action_label": f"Apply {col}_Binned Feature",
                    "series": binned_series,
                })
        except Exception:
            pass
    return candidates


def generate_datetime_features(df, target_col=None):
    """
    Automatically detect datetime columns and generate component candidate features.

    Returns
    -------
    list of dict
    """
    candidates = []
    cols = [c for c in df.columns if c != target_col]

    for col in cols:
        is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
        if not is_dt and (df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])):
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ["date", "time", "dt", "timestamp", "created", "updated", "dob"]):
                try:
                    sample = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                    if sample.notna().mean() > 0.7:
                        is_dt = True
                except Exception:
                    pass

        if is_dt:
            dt_series = pd.to_datetime(df[col], errors="coerce")

            components = [
                ("Year", dt_series.dt.year),
                ("Month", dt_series.dt.month),
                ("DayOfWeek", dt_series.dt.dayofweek),
                ("Quarter", dt_series.dt.quarter),
                ("IsWeekend", dt_series.dt.dayofweek.isin([5, 6]).astype(int)),
            ]

            for comp_name, comp_series in components:
                if comp_series.notna().sum() > 0 and comp_series.nunique() > 1:
                    candidates.append({
                        "feature_name": f"{col}_{comp_name}",
                        "feature_type": "Datetime Feature",
                        "source_columns": [col],
                        "action_type": "apply_datetime_feature",
                        "action_label": f"Apply {col}_{comp_name} Feature",
                        "series": comp_series,
                    })

    return candidates


def test_feature_impact(df, candidate, target_col, baseline_score, problem_type=None, cv=CV_FOLDS, random_state=42):
    """
    Test the CV impact of adding a single candidate feature to the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    candidate : dict
        Candidate feature definition dict containing 'feature_name' and 'series'.
    target_col : str
        Target column.
    baseline_score : float
        Primary score of the baseline model.
    problem_type : str, optional
        'Classification' or 'Regression'.
    cv : int, default=5
        CV folds.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Evaluation results with new score, improvement, and recommendation status.
    """
    feat_name = candidate["feature_name"]
    series = candidate["series"]

    X_test = df.drop(columns=[target_col]).copy()
    y_test = df[target_col]

    # Ensure no target leakage
    if feat_name in X_test.columns:
        X_test = X_test.drop(columns=[feat_name])

    X_test[feat_name] = series

    df_test = X_test.copy()
    df_test[target_col] = y_test

    try:
        eval_res = evaluate_baseline(df_test, target_col=target_col, problem_type=problem_type, cv=cv, random_state=random_state)
        new_score = eval_res["primary_score"]
        improvement = new_score - baseline_score

        is_recommended = improvement >= MIN_IMPROVEMENT

        metric_name = "Accuracy" if eval_res["problem_type"] == "Classification" else "R2 Score"
        imp_fmt = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"

        rec_text = (
            f"Applying {candidate['feature_type']} '{feat_name}' (derived from {candidate['source_columns']}) "
            f"improved cross-validation {metric_name} from {baseline_score:.4f} to {new_score:.4f} ({imp_fmt})."
            if is_recommended else
            f"Candidate feature '{feat_name}' yielded cross-validation {metric_name} of {new_score:.4f} ({imp_fmt}). "
            f"Improvement did not reach minimum threshold (+{MIN_IMPROVEMENT})."
        )

        return {
            "feature_name": feat_name,
            "feature_type": candidate["feature_type"],
            "source_columns": candidate["source_columns"],
            "baseline_score": baseline_score,
            "new_score": new_score,
            "improvement": improvement,
            "is_recommended": is_recommended,
            "recommendation": rec_text,
            "action_type": candidate["action_type"],
            "action_label": candidate["action_label"],
            "series": series,
            "metrics": eval_res
        }
    except Exception as err:
        return {
            "feature_name": feat_name,
            "feature_type": candidate["feature_type"],
            "source_columns": candidate["source_columns"],
            "baseline_score": baseline_score,
            "new_score": baseline_score,
            "improvement": 0.0,
            "is_recommended": False,
            "recommendation": f"Feature evaluation failed: {err}",
            "action_type": candidate["action_type"],
            "action_label": candidate["action_label"],
            "series": series,
            "metrics": {}
        }


def build_feature_engineering_recommendations(
    df,
    target_col,
    problem_type=None,
    top_n_features=MAX_TOP_FEATURES,
    cv_folds=CV_FOLDS,
    min_improvement=MIN_IMPROVEMENT,
    random_state=42
):
    """
    Run full automated Feature Engineering Analysis pipeline:
    1. Baseline evaluation
    2. Feature importance ranking
    3. Candidate generation
    4. Feature impact CV testing
    5. Ranking and Recommendation construction

    Returns
    -------
    dict
        Comprehensive results dictionary containing baseline, importances, tested candidates, and recommendations.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # 1. Baseline Model
    baseline = evaluate_baseline(df, target_col, problem_type=problem_type, cv=cv_folds, random_state=random_state)
    baseline_score = baseline["primary_score"]

    # 2. Feature Importance
    importance_df = get_feature_importance(df, target_col, problem_type=baseline["problem_type"], top_n=top_n_features, random_state=random_state)
    top_important_cols = importance_df.head(top_n_features)["Feature"].tolist()
    important_num_cols = [c for c in top_important_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # 3. Candidate Feature Generation
    candidates = []
    candidates.extend(generate_log_features(df, important_num_cols, target_col=target_col))
    candidates.extend(generate_polynomial_features(df, important_num_cols, target_col=target_col, degree=2))
    candidates.extend(generate_interaction_features(df, important_num_cols, target_col=target_col, max_features=MAX_INTERACTION_FEATURES))
    candidates.extend(generate_ratio_features(df, important_num_cols, target_col=target_col, max_features=MAX_RATIO_FEATURES))
    candidates.extend(generate_binning_features(df, important_num_cols, target_col=target_col, q=4))
    candidates.extend(generate_datetime_features(df, target_col=target_col))

    # 4. Feature Impact Testing
    tested_results = []
    for cand in candidates:
        res = test_feature_impact(
            df, cand, target_col=target_col,
            baseline_score=baseline_score,
            problem_type=baseline["problem_type"],
            cv=cv_folds,
            random_state=random_state
        )
        tested_results.append(res)

    # 5. Ranking & Filtering
    tested_df = pd.DataFrame(tested_results)
    if not tested_df.empty:
        tested_df = tested_df.sort_values(by="improvement", ascending=False).reset_index(drop=True)
        tested_results = tested_df.to_dict(orient="records")

    recommendations = []
    rec_id = 1
    for item in tested_results:
        if item["improvement"] >= min_improvement:
            imp_val = item["improvement"]
            severity = "high" if imp_val >= 0.03 else ("medium" if imp_val >= 0.01 else "info")
            rec_obj = {
                "id": f"feature_rec_{rec_id}",
                "feature_name": item["feature_name"],
                "feature_type": item["feature_type"],
                "source_columns": item["source_columns"],
                "severity": severity,
                "baseline_score": item["baseline_score"],
                "new_score": item["new_score"],
                "improvement": item["improvement"],
                "recommendation": item["recommendation"],
                "action_type": item["action_type"],
                "action_label": item["action_label"],
                "series": item["series"],
            }
            recommendations.append(rec_obj)
            rec_id += 1

    return {
        "baseline": baseline,
        "importance_df": importance_df,
        "total_original_features": len([c for c in df.columns if c != target_col]),
        "total_candidates": len(candidates),
        "total_tested": len(tested_results),
        "tested_results": tested_results,
        "recommendations": recommendations,
        "recommended_count": len(recommendations),
    }


def apply_feature_recommendations(df, selected_recommendations, target_col=None):
    """
    Apply selected feature engineering recommendations to the dataset safely.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    selected_recommendations : list of dict
        Selected feature recommendation dicts to apply.
    target_col : str, optional
        Target column to protect from accidental modification.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with engineered features.
    """
    out = df.copy()

    for rec in selected_recommendations:
        feat_name = rec.get("feature_name")
        if not feat_name or feat_name == target_col:
            continue

        series = rec.get("series")
        if series is None:
            # Fallback re-creation if series wasn't cached
            src_cols = rec.get("source_columns", [])
            f_type = rec.get("feature_type", "")

            if "Log" in f_type and src_cols:
                series = np.log1p(out[src_cols[0]].clip(lower=0))
            elif "Polynomial" in f_type and src_cols:
                series = out[src_cols[0]] ** 2
            elif "Interaction" in f_type and len(src_cols) >= 2:
                series = out[src_cols[0]] * out[src_cols[1]]
            elif "Ratio" in f_type and len(src_cols) >= 2:
                denom = out[src_cols[1]].replace(0, np.nan)
                series = (out[src_cols[0]] / denom).replace([np.inf, -np.inf], np.nan)
            elif "Binning" in f_type and src_cols:
                try:
                    series = pd.qcut(out[src_cols[0]], q=4, labels=False, duplicates="drop")
                except Exception:
                    series = None
            elif "Datetime" in f_type and src_cols:
                dt_series = pd.to_datetime(out[src_cols[0]], errors="coerce")
                if "Year" in feat_name:
                    series = dt_series.dt.year
                elif "Month" in feat_name:
                    series = dt_series.dt.month
                elif "DayOfWeek" in feat_name:
                    series = dt_series.dt.dayofweek
                elif "Quarter" in feat_name:
                    series = dt_series.dt.quarter
                elif "IsWeekend" in feat_name:
                    series = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

        if series is not None:
            # Clean infinite or invalid values safely
            if pd.api.types.is_numeric_dtype(series):
                series = series.replace([np.inf, -np.inf], np.nan)
                if series.isna().any():
                    med = series.median()
                    fill_val = med if not pd.isna(med) else 0
                    series = series.fillna(fill_val)

            out[feat_name] = series

    return out
