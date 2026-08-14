def build_recommendations(df, analysis):
    recommendations = []
    rec_id = 1

    # 1. Duplicate rows
    if analysis["duplicates"] > 0:
        recommendations.append({
            "id": f"rec_{rec_id}",
            "issue": "Duplicate Rows",
            "column": "All Rows",
            "severity": "high",
            "recommendation": f"Found {analysis['duplicates']} exact duplicate rows. Removing duplicates prevents data leakage and bias.",
            "action_type": "remove_duplicates",
            "action_label": f"🧹 Remove {analysis['duplicates']} Duplicate Rows"
        })
        rec_id += 1

    # 2. Missing Values
    for column, percent in analysis["missing_percent"].items():
        if percent <= 0:
            continue
        
        missing_cnt = analysis["missing_counts"].get(column, 0)
        
        if percent > 60:
            recommendations.append({
                "id": f"rec_{rec_id}",
                "issue": "High Missing Ratio",
                "column": column,
                "severity": "high",
                "recommendation": f"Column '{column}' has {percent:.1f}% ({missing_cnt} rows) missing values. Dropping this feature is recommended.",
                "action_type": "drop_column",
                "action_label": f"🗑️ Drop '{column}' Column"
            })
            rec_id += 1
        elif column in analysis["numeric_columns"]:
            skew = abs(analysis["skewness"].get(column, 0))
            if skew > 1:
                recommendations.append({
                    "id": f"rec_{rec_id}",
                    "issue": "Missing Values (Skewed)",
                    "column": column,
                    "severity": "medium",
                    "recommendation": f"Column '{column}' has {percent:.1f}% missing values and a skewed distribution (|skew|={skew:.2f}). Median imputation is recommended.",
                    "action_type": "impute_median",
                    "action_label": f"⚡ Impute '{column}' with Median"
                })
                rec_id += 1
            else:
                recommendations.append({
                    "id": f"rec_{rec_id}",
                    "issue": "Missing Values",
                    "column": column,
                    "severity": "medium",
                    "recommendation": f"Column '{column}' has {percent:.1f}% missing values. Mean imputation is recommended for symmetrical distributions.",
                    "action_type": "impute_mean",
                    "action_label": f"⚡ Impute '{column}' with Mean"
                })
                rec_id += 1
        else:
            recommendations.append({
                "id": f"rec_{rec_id}",
                "issue": "Missing Values (Categorical)",
                "column": column,
                "severity": "medium",
                "recommendation": f"Categorical column '{column}' has {percent:.1f}% missing values. Mode imputation or 'Unknown' category recommended.",
                "action_type": "impute_mode",
                "action_label": f"⚡ Impute '{column}' with Mode"
            })
            rec_id += 1

    # 3. Constant Columns
    for column in analysis["constant_columns"]:
        recommendations.append({
            "id": f"rec_{rec_id}",
            "issue": "Constant Column",
            "column": column,
            "severity": "medium",
            "recommendation": f"Column '{column}' carries zero variance (single unique value). Drop it to streamline modeling.",
            "action_type": "drop_column",
            "action_label": f"❌ Drop Zero-Variance '{column}'"
        })
        rec_id += 1

    # 4. ID / High Cardinality Identifiers
    for column in analysis["id_columns"]:
        recommendations.append({
            "id": f"rec_{rec_id}",
            "issue": "Identifier Detection",
            "column": column,
            "severity": "info",
            "recommendation": f"Column '{column}' appears to be an ID/key column with near-unique values. Exclude or drop from predictive modeling.",
            "action_type": "drop_column",
            "action_label": f"🗑️ Exclude Identifier '{column}'"
        })
        rec_id += 1

    # 5. Outliers
    for column, count in analysis["outliers"].items():
        percent_out = (count / analysis["rows"]) * 100
        if percent_out >= 1.0:
            recommendations.append({
                "id": f"rec_{rec_id}",
                "issue": "Outliers Detected",
                "column": column,
                "severity": "medium",
                "recommendation": f"Column '{column}' contains {count} outliers ({percent_out:.1f}% of rows) based on 1.5x IQR rule. Apply IQR capping (Winsorization) to avoid model distortion.",
                "action_type": "cap_outliers_iqr",
                "action_label": f"✂️ Cap Outliers in '{column}' (IQR)"
            })
            rec_id += 1

    # 6. Highly Skewed Features
    for column, skew_val in analysis["skewness"].items():
        if skew_val > 1.5 and column not in analysis["constant_columns"]:
            recommendations.append({
                "id": f"rec_{rec_id}",
                "issue": "High Right Skewness",
                "column": column,
                "severity": "info",
                "recommendation": f"Column '{column}' has strong positive skewness ({skew_val:.2f}). Applying a Log1p transformation will normalize its distribution.",
                "action_type": "apply_log1p",
                "action_label": f"📐 Apply Log1p to '{column}'"
            })
            rec_id += 1

    # 7. High Collinearity
    for col1, col2, corr_val in analysis["high_correlation"]:
        recommendations.append({
            "id": f"rec_{rec_id}",
            "issue": "High Collinearity",
            "column": f"{col1} & {col2}",
            "severity": "info",
            "recommendation": f"Columns '{col1}' and '{col2}' are highly correlated (r={corr_val:.2f}). Consider dropping '{col2}' to avoid multicollinearity.",
            "action_type": "drop_column_col2",
            "target_col": col2,
            "action_label": f"🗑️ Drop Correlated '{col2}'"
        })
        rec_id += 1

    return recommendations
