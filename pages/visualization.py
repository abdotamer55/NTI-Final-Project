import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.app_state import get_dataset
from src.ui import page_header, render_dataset_toolbar
from utils.pipeline_manager import detect_problem_type

CHARTS = [
    "Histogram",
    "Density / KDE Plot",
    "Box Plot",
    "Violin Plot",
    "Bar Chart",
    "Pie / Donut Chart",
    "Scatter Plot",
    "3D Scatter Plot",
    "Line Chart",
    "Area Chart",
    "Bubble Chart",
    "Correlation Heatmap",
    "Pivot Table Heatmap",
    "Sunburst Hierarchy",
    "Treemap",
    "Strip Plot",
    "ECDF (Cumulative Distribution)",
    "Parallel Coordinates",
    "Scatter Matrix (Pairplot)",
    "Missing Value Matrix"
]

COLOR_PALETTES = [
    "Viridis", "Plasma", "Turbo", "Inferno", "Magma",
    "RdBu", "Cividis", "Spectral", "Blues", "Purples", "Teal", "Sunset"
]

THEMES = ["plotly_dark", "plotly", "simple_white", "ggplot2", "seaborn"]

def _get_column_stats(df, col):
    """Compute quick statistical summary for a numeric or categorical column."""
    if col not in df.columns:
        return None
    series = df[col].dropna()
    if len(series) == 0:
        return None

    if pd.api.types.is_numeric_dtype(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_cnt = int(((series < lower) | (series > upper)).sum())
        skew_val = float(series.skew()) if len(series) > 2 else 0.0

        return {
            "type": "numeric",
            "count": len(series),
            "missing": int(df[col].isna().sum()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "iqr": float(iqr),
            "skew": skew_val,
            "outliers": outlier_cnt
        }
    else:
        vc = series.value_counts()
        top_val = vc.index[0] if len(vc) > 0 else "N/A"
        top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
        return {
            "type": "categorical",
            "count": len(series),
            "missing": int(df[col].isna().sum()),
            "unique": int(series.nunique()),
            "top": str(top_val),
            "top_freq": top_freq,
            "top_pct": (top_freq / max(len(series), 1)) * 100
        }

def _render_stats_card(stats, col_name):
    """Render a compact glassmorphic statistics card under the chart."""
    if not stats:
        return

    st.markdown(f"#### 💡 Instant Statistical Profile: `{col_name}`")
    if stats["type"] == "numeric":
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        k1.metric("Mean", f"{stats['mean']:.2f}")
        k2.metric("Median", f"{stats['median']:.2f}")
        k3.metric("Std Dev", f"{stats['std']:.2f}")
        k4.metric("Min", f"{stats['min']:.2f}")
        k5.metric("Max", f"{stats['max']:.2f}")
        k6.metric("Skewness", f"{stats['skew']:.2f}")
        k7.metric("Outliers (IQR)", f"{stats['outliers']:,}")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Valid Rows", f"{stats['count']:,}")
        k2.metric("Unique Classes", f"{stats['unique']:,}")
        k3.metric("Most Frequent", f"{stats['top']}")
        k4.metric("Top Class Share", f"{stats['top_pct']:.1f}%")

def render():
    page_header(
        "📊 Visualization Studio",
        "Interactive high-resolution data visualization suite with custom chart builders, automated EDA dashboards, and target relationship insights."
    )

    df = get_dataset()
    if df is None:
        st.warning("Please upload or load a dataset first to begin visual exploration.")
        return

    render_dataset_toolbar()

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    all_columns = df.columns.tolist()

    # 4-Tab High Value Visual Workspace
    tabs = st.tabs([
        "🎨 Custom Chart Studio",
        "⚡ Automated EDA Quick Visuals",
        "🎯 Feature vs Target Studio",
        "🩹 Data Quality & Missing Patterns"
    ])

    # =========================================================================
    # TAB 1: Custom Chart Studio
    # =========================================================================
    with tabs[0]:
        st.subheader("🎨 Interactive Custom Chart Builder")
        st.write("Construct tailored, publication-ready interactive visualizations with granular styling and aggregation options.")

        # Data Slicer Filter Expander
        with st.expander("🔍 Interactive Data Slicer & Filter (Optional)", expanded=False):
            fc1, fc2 = st.columns([1.5, 2.5])
            with fc1:
                filter_col = st.selectbox("Filter by Column", ["None"] + all_columns, key="vis_filter_col")
            
            working_df = df.copy()
            if filter_col != "None":
                with fc2:
                    if filter_col in numeric:
                        min_v = float(df[filter_col].min())
                        max_v = float(df[filter_col].max())
                        if min_v < max_v:
                            selected_range = st.slider(f"Filter Range for {filter_col}", min_v, max_v, (min_v, max_v), key="vis_filter_num")
                            working_df = working_df[(working_df[filter_col] >= selected_range[0]) & (working_df[filter_col] <= selected_range[1])]
                    else:
                        unique_vals = df[filter_col].dropna().unique().tolist()
                        selected_vals = st.multiselect(f"Select Categories for {filter_col}", unique_vals, default=unique_vals[:min(10, len(unique_vals))], key="vis_filter_cat")
                        if selected_vals:
                            working_df = working_df[working_df[filter_col].isin(selected_vals)]
                st.caption(f"Filtered subset: **{len(working_df):,}** of **{len(df):,}** rows")

        # Top Selection Controls
        c1, c2, c3, c4 = st.columns([1.6, 1.4, 1.2, 1.2])
        with c1:
            chart_type = st.selectbox("Chart Type", CHARTS, key="vis_chart_type")
        with c2:
            palette = st.selectbox("Color Palette", COLOR_PALETTES, key="vis_palette")
        with c3:
            theme = st.selectbox("Theme", THEMES, key="vis_theme")
        with c4:
            plot_height = st.slider("Plot Height (px)", 450, 900, 620, 50, key="vis_height")

        # Chart-Specific Parameter Controls
        st.markdown("---")
        x_col = None
        y_col = None
        z_col = None
        color_col = None
        size_col = None
        facet_col = None
        stat_target_col = None

        p1, p2, p3, p4 = st.columns(4)

        if chart_type in ["Histogram", "Density / KDE Plot", "ECDF (Cumulative Distribution)", "Pie / Donut Chart"]:
            with p1:
                x_col = st.selectbox("Target Feature (X)", all_columns, key="p_x")
                stat_target_col = x_col
            with p2:
                color_col = st.selectbox("Color / Grouping (Optional)", ["None"] + all_columns, key="p_color")
            with p3:
                facet_col = st.selectbox("Facet Subplots Column", ["None"] + all_columns, key="p_facet")
            with p4:
                nbins = st.slider("Bins Count", 10, 100, 35, key="p_nbins") if chart_type == "Histogram" else None

        elif chart_type in ["Box Plot", "Violin Plot"]:
            with p1:
                y_col = st.selectbox("Numeric Metric (Y)", numeric if numeric else all_columns, key="p_box_y")
                stat_target_col = y_col
            with p2:
                x_col = st.selectbox("Categorical Group (X)", ["None (Whole Dataset)"] + all_columns, key="p_box_x")
            with p3:
                color_col = st.selectbox("Subgroup (Color)", ["None"] + all_columns, key="p_box_color")
            with p4:
                facet_col = st.selectbox("Facet Column", ["None"] + all_columns, key="p_box_facet")

        elif chart_type == "Bar Chart":
            with p1:
                x_col = st.selectbox("Category Axis (X)", all_columns, key="p_bar_x")
                stat_target_col = x_col
            with p2:
                y_col = st.selectbox("Metric to Aggregate (Y)", ["None (Count Rows)"] + numeric, key="p_bar_y")
            with p3:
                color_col = st.selectbox("Color / Sub-category", ["None"] + all_columns, key="p_bar_color")
            with p4:
                agg_func = st.selectbox("Aggregation Method", ["mean", "sum", "count", "median", "max", "min"], key="p_bar_agg")

        elif chart_type in ["Scatter Plot", "Line Chart", "Area Chart", "Bubble Chart"]:
            with p1:
                x_col = st.selectbox("X Axis", numeric if numeric else all_columns, key="p_scat_x")
                stat_target_col = x_col
            with p2:
                y_col = st.selectbox("Y Axis", numeric if numeric else all_columns, index=min(1, len(numeric)-1) if numeric else 0, key="p_scat_y")
            with p3:
                color_col = st.selectbox("Color / Hue", ["None"] + all_columns, key="p_scat_color")
            with p4:
                if chart_type == "Bubble Chart":
                    size_col = st.selectbox("Bubble Size Metric", numeric, key="p_bubble_size")
                else:
                    facet_col = st.selectbox("Facet Subplots Column", ["None"] + all_columns, key="p_scat_facet")

        elif chart_type == "3D Scatter Plot":
            with p1:
                x_col = st.selectbox("X Column", numeric, key="p_3d_x")
            with p2:
                y_col = st.selectbox("Y Column", numeric, index=min(1, len(numeric)-1), key="p_3d_y")
            with p3:
                z_col = st.selectbox("Z Column", numeric, index=min(2, len(numeric)-1), key="p_3d_z")
            with p4:
                color_col = st.selectbox("Color Column", ["None"] + all_columns, key="p_3d_color")

        elif chart_type == "Correlation Heatmap":
            with p1:
                corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], key="p_corr_m")
            with p2:
                show_annot = st.checkbox("Show Correlation Values", value=True, key="p_corr_annot")
            with p3:
                corr_subset = st.multiselect("Select Numeric Features", numeric, default=numeric, key="p_corr_subset")

        elif chart_type == "Pivot Table Heatmap":
            with p1:
                pivot_row = st.selectbox("Row Category (Y)", categorical if categorical else all_columns, key="p_piv_row")
            with p2:
                pivot_col = st.selectbox("Column Category (X)", [c for c in all_columns if c != pivot_row], key="p_piv_col")
            with p3:
                pivot_val = st.selectbox("Value Metric (Z)", numeric if numeric else all_columns, key="p_piv_val")
            with p4:
                pivot_agg = st.selectbox("Aggregation", ["mean", "sum", "count", "median", "max", "min"], key="p_piv_agg")

        elif chart_type in ["Sunburst Hierarchy", "Treemap"]:
            with p1:
                hierarchy_cols = st.multiselect("Hierarchy Path (Levels)", all_columns, default=categorical[:min(3, len(categorical))] if categorical else all_columns[:2], key="p_hier_path")
            with p2:
                hier_val = st.selectbox("Weight / Size Metric", ["None (Count)"] + numeric, key="p_hier_val")

        elif chart_type == "Strip Plot":
            with p1:
                y_col = st.selectbox("Numeric Column (Y)", numeric if numeric else all_columns, key="p_strip_y")
            with p2:
                x_col = st.selectbox("Category (X)", ["None"] + all_columns, key="p_strip_x")
            with p3:
                color_col = st.selectbox("Color", ["None"] + all_columns, key="p_strip_color")

        elif chart_type == "Parallel Coordinates":
            with p1:
                par_dims = st.multiselect("Dimensions to Plot", numeric, default=numeric[:min(6, len(numeric))], key="p_par_dims")
            with p2:
                par_color = st.selectbox("Color Continuum Dimension", numeric, key="p_par_color") if numeric else None

        elif chart_type == "Scatter Matrix (Pairplot)":
            with p1:
                mat_dims = st.multiselect("Matrix Features (Select 2-5)", numeric, default=numeric[:min(4, len(numeric))], key="p_mat_dims")
            with p2:
                color_col = st.selectbox("Color Hue", ["None"] + all_columns, key="p_mat_color")

        # Advanced Plot Fine-Tuning Expander
        with st.expander("⚙️ Advanced Plot Options & Customization", expanded=False):
            adv1, adv2, adv3, adv4 = st.columns(4)
            custom_title = adv1.text_input("Custom Title", value=f"{chart_type}" + (f": {x_col}" if x_col else ""), key="vis_custom_title")
            log_y = adv2.checkbox("Logarithmic Y-Scale", key="vis_adv_log")
            show_trend = adv3.checkbox("Show OLS Trendline", value=False, key="vis_adv_trend") if chart_type == "Scatter Plot" else False
            top_n_limit = adv4.slider("Limit Top Categories", 5, 50, 25, key="vis_adv_topn")

        # Render Figure
        fig = None
        color_arg = None if (color_col == "None" or color_col is None) else color_col
        facet_arg = None if (facet_col == "None" or facet_col is None) else facet_col

        try:
            if chart_type == "Histogram":
                fig = px.histogram(
                    working_df, x=x_col, color=color_arg, nbins=nbins, marginal="box",
                    facet_col=facet_arg, color_discrete_sequence=px.colors.sequential.__dict__.get(palette, None)
                )

            elif chart_type == "Density / KDE Plot":
                fig = px.histogram(
                    working_df, x=x_col, color=color_arg, histnorm="probability density",
                    opacity=0.75, facet_col=facet_arg, marginal="violin"
                )

            elif chart_type == "Box Plot":
                actual_x = None if (x_col == "None (Whole Dataset)" or x_col is None) else x_col
                fig = px.box(
                    working_df, x=actual_x, y=y_col, color=color_arg if color_arg else actual_x,
                    points="outliers", facet_col=facet_arg
                )

            elif chart_type == "Violin Plot":
                actual_x = None if (x_col == "None (Whole Dataset)" or x_col is None) else x_col
                fig = px.violin(
                    working_df, x=actual_x, y=y_col, color=color_arg if color_arg else actual_x,
                    box=True, points="outliers", facet_col=facet_arg
                )

            elif chart_type == "Bar Chart":
                if y_col == "None (Count Rows)":
                    if color_arg:
                        plot_df = working_df.groupby([x_col, color_arg], dropna=False).size().reset_index(name="Count")
                        fig = px.bar(plot_df, x=x_col, y="Count", color=color_arg, barmode="group", title=custom_title)
                    else:
                        plot_df = working_df[x_col].astype(str).value_counts(dropna=False).head(top_n_limit).reset_index()
                        plot_df.columns = [x_col, "Count"]
                        fig = px.bar(plot_df, x=x_col, y="Count", color=x_col, title=custom_title)
                else:
                    if color_arg:
                        plot_df = working_df.groupby([x_col, color_arg], dropna=False)[y_col].agg(agg_func).reset_index()
                        fig = px.bar(plot_df, x=x_col, y=y_col, color=color_arg, barmode="group", title=custom_title)
                    else:
                        plot_df = working_df.groupby(x_col, dropna=False)[y_col].agg(agg_func).head(top_n_limit).reset_index()
                        fig = px.bar(plot_df, x=x_col, y=y_col, color=x_col, title=custom_title)

            elif chart_type == "Pie / Donut Chart":
                plot_df = working_df[x_col].astype(str).value_counts().head(top_n_limit).reset_index()
                plot_df.columns = [x_col, "Count"]
                fig = px.pie(plot_df, names=x_col, values="Count", hole=0.4, title=custom_title)

            elif chart_type == "Scatter Plot":
                clean_scat = working_df.dropna(subset=[x_col, y_col])
                trend_opt = "ols" if show_trend and len(clean_scat) < 5000 else None
                fig = px.scatter(
                    clean_scat, x=x_col, y=y_col, color=color_arg, facet_col=facet_arg,
                    trendline=trend_opt, opacity=0.8
                )

            elif chart_type == "3D Scatter Plot":
                if x_col and y_col and z_col:
                    fig = px.scatter_3d(working_df, x=x_col, y=y_col, z=z_col, color=color_arg, opacity=0.85)

            elif chart_type == "Line Chart":
                sorted_df = working_df.sort_values(x_col) if x_col in numeric else working_df
                fig = px.line(sorted_df, x=x_col, y=y_col, color=color_arg, markers=True, facet_col=facet_arg)

            elif chart_type == "Area Chart":
                sorted_df = working_df.sort_values(x_col) if x_col in numeric else working_df
                fig = px.area(sorted_df, x=x_col, y=y_col, color=color_arg, facet_col=facet_arg)

            elif chart_type == "Bubble Chart":
                clean_bub = working_df.dropna(subset=[x_col, y_col, size_col])
                fig = px.scatter(clean_bub, x=x_col, y=y_col, size=size_col, color=color_arg, facet_col=facet_arg)

            elif chart_type == "Correlation Heatmap":
                if len(corr_subset) < 2:
                    st.warning("Please select at least 2 numeric features for the correlation heatmap.")
                else:
                    corr = working_df[corr_subset].corr(method=corr_method)
                    fig = px.imshow(
                        corr, text_auto=".2f" if show_annot else False, aspect="auto",
                        color_continuous_scale=palette, zmin=-1, zmax=1
                    )

            elif chart_type == "Pivot Table Heatmap":
                piv_df = working_df.pivot_table(index=pivot_row, columns=pivot_col, values=pivot_val, aggfunc=pivot_agg, fill_value=0)
                fig = px.imshow(
                    piv_df, text_auto=".2f", aspect="auto",
                    color_continuous_scale=palette,
                    labels=dict(x=pivot_col, y=pivot_row, color=f"{pivot_agg}({pivot_val})")
                )

            elif chart_type in ["Sunburst Hierarchy", "Treemap"]:
                if not hierarchy_cols:
                    st.warning("Please select at least one hierarchy level column.")
                else:
                    val_arg = None if hier_val == "None (Count)" else hier_val
                    if chart_type == "Sunburst Hierarchy":
                        fig = px.sunburst(working_df, path=hierarchy_cols, values=val_arg)
                    else:
                        fig = px.treemap(working_df, path=hierarchy_cols, values=val_arg)

            elif chart_type == "Strip Plot":
                actual_x = None if x_col == "None" else x_col
                fig = px.strip(working_df, x=actual_x, y=y_col, color=color_arg)

            elif chart_type == "ECDF (Cumulative Distribution)":
                fig = px.ecdf(working_df, x=x_col, color=color_arg)

            elif chart_type == "Parallel Coordinates":
                if len(par_dims) < 2:
                    st.warning("Parallel Coordinates requires at least 2 numeric dimensions.")
                else:
                    clean_par = working_df.dropna(subset=par_dims)
                    if len(clean_par) > 2000:
                        clean_par = clean_par.sample(2000, random_state=42)
                        st.info("Sampled 2,000 records for optimal Parallel Coordinates rendering.")
                    fig = px.parallel_coordinates(clean_par, dimensions=par_dims, color=par_color if par_color else par_dims[0])

            elif chart_type == "Scatter Matrix (Pairplot)":
                if len(mat_dims) < 2:
                    st.warning("Please select at least 2 features for the Scatter Matrix.")
                else:
                    clean_mat = working_df.dropna(subset=mat_dims)
                    if len(clean_mat) > 1500:
                        clean_mat = clean_mat.sample(1500, random_state=42)
                        st.info("Sampled 1,500 records to prevent browser lag during pairwise scatter rendering.")
                    fig = px.scatter_matrix(clean_mat, dimensions=mat_dims, color=color_arg)

            elif chart_type == "Missing Value Matrix":
                sample_n = min(len(working_df), 1000)
                missing_mat = working_df.iloc[:sample_n].isna().astype(int)
                fig = px.imshow(
                    missing_mat.T, color_continuous_scale=["#141e36", "#ef4444"],
                    title=f"Missing Value Matrix (Top {sample_n} rows, Red = Missing)", aspect="auto"
                )

            # Apply Layout Settings
            if fig is not None:
                fig.update_layout(
                    title=custom_title,
                    template=theme,
                    height=plot_height,
                    margin=dict(l=30, r=30, t=60, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                if log_y:
                    try:
                        fig.update_yaxes(type="log")
                    except Exception:
                        pass

                st.plotly_chart(fig, use_container_width=True)

                # Export & Download Toolbar
                d1, d2, d3 = st.columns([1.5, 1.5, 3])
                with d1:
                    st.download_button(
                        "⬇ Download Interactive HTML",
                        fig.to_html(include_plotlyjs="cdn"),
                        file_name=f"{chart_type.lower().replace(' ', '_')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                with d2:
                    st.download_button(
                        "⬇ Export Chart Data Snapshot (CSV)",
                        working_df.to_csv(index=False).encode(),
                        file_name="chart_data_snapshot.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with d3:
                    st.caption("💡 You can interact directly with the plot (Zoom, Pan, Box Select, or hover for detailed values).")

                # Statistical Insights Card
                if stat_target_col:
                    stats = _get_column_stats(working_df, stat_target_col)
                    _render_stats_card(stats, stat_target_col)

        except Exception as err:
            st.error(f"⚠️ Unable to render chart with current parameters: {err}")

    # =========================================================================
    # TAB 2: Automated EDA Quick Visuals
    # =========================================================================
    with tabs[1]:
        st.subheader("⚡ Automated 1-Click EDA Dashboard")
        st.write("Generate batch distributions, categorical frequencies, and correlation maps instantly across your complete dataset.")

        eda_subtabs = st.tabs([
            "📈 Numerical Distributions",
            "🏷️ Categorical Frequencies",
            "📦 Outliers Boxplot Grid",
            "🔥 Instant Full Correlation Matrix"
        ])

        # Subtab 1: Numerical Distributions Grid
        with eda_subtabs[0]:
            st.markdown("#### Numerical Feature Distributions (Histograms + Boxplots)")
            if numeric:
                num_sel = st.multiselect("Features to Visualize", numeric, default=numeric[:min(8, len(numeric))], key="auto_num_sel")
                if num_sel:
                    cols_grid = st.columns(2)
                    for idx, num_col in enumerate(num_sel):
                        with cols_grid[idx % 2]:
                            stats = _get_column_stats(df, num_col)
                            badge_text = f"Mean: {stats['mean']:.2f} | Std: {stats['std']:.2f} | Skew: {stats['skew']:.2f}" if stats else ""
                            sub_fig = px.histogram(
                                df, x=num_col, marginal="box", nbins=30,
                                title=f"<b>{num_col}</b> <span style='font-size:0.8rem; color:#94a3b8;'>({badge_text})</span>",
                                color_discrete_sequence=["#6d5dfc"]
                            )
                            sub_fig.update_layout(
                                height=360, margin=dict(l=20, r=20, t=45, b=20),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#f8fafc"
                            )
                            st.plotly_chart(sub_fig, use_container_width=True)
                else:
                    st.info("Select one or more numeric features to generate distributions.")
            else:
                st.info("No numeric features detected in the dataset.")

        # Subtab 2: Categorical Frequencies Grid
        with eda_subtabs[1]:
            st.markdown("#### Categorical Feature Value Counts & Percentages")
            if categorical:
                cat_sel = st.multiselect("Categorical Features to Visualize", categorical, default=categorical[:min(6, len(categorical))], key="auto_cat_sel")
                if cat_sel:
                    cat_grid = st.columns(2)
                    for idx, cat_col in enumerate(cat_sel):
                        with cat_grid[idx % 2]:
                            counts_df = df[cat_col].astype(str).value_counts().head(15).reset_index()
                            counts_df.columns = [cat_col, "Count"]
                            counts_df["Percentage"] = (counts_df["Count"] / len(df)) * 100

                            cat_fig = px.bar(
                                counts_df, x="Count", y=cat_col, orientation="h",
                                text=counts_df["Percentage"].apply(lambda v: f"{v:.1f}%"),
                                title=f"<b>{cat_col}</b> (Top {len(counts_df)} Classes)",
                                color="Count", color_continuous_scale="Viridis"
                            )
                            cat_fig.update_layout(
                                height=360, margin=dict(l=20, r=20, t=45, b=20),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#f8fafc"
                            )
                            st.plotly_chart(cat_fig, use_container_width=True)
                else:
                    st.info("Select one or more categorical features.")
            else:
                st.info("No categorical features detected.")

        # Subtab 3: Outliers Boxplot Grid
        with eda_subtabs[2]:
            st.markdown("#### Outlier Profiling Across Numerical Features")
            if numeric:
                out_fig = px.box(
                    df[numeric], points="outliers",
                    title="Comparative Outliers Overview (All Numeric Columns)",
                    color_discrete_sequence=["#22d3ee"]
                )
                out_fig.update_layout(
                    height=500, margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc"
                )
                st.plotly_chart(out_fig, use_container_width=True)
            else:
                st.info("No numeric features available for outlier comparison.")

        # Subtab 4: Instant Correlation Matrix
        with eda_subtabs[3]:
            st.markdown("#### Full Pearson & Spearman Correlation Matrices")
            if len(numeric) >= 2:
                c_method = st.radio("Correlation Metric", ["Pearson (Linear)", "Spearman (Monotonic / Rank)"], horizontal=True, key="auto_corr_meth")
                m_str = "pearson" if "Pearson" in c_method else "spearman"
                corr_mat = df[numeric].corr(method=m_str)

                fig_full_corr = px.imshow(
                    corr_mat, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Complete {c_method} Matrix ({len(numeric)} Features)"
                )
                fig_full_corr.update_layout(
                    height=600, margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc"
                )
                st.plotly_chart(fig_full_corr, use_container_width=True)
            else:
                st.info("Requires at least 2 numeric features for correlation analysis.")

    # =========================================================================
    # TAB 3: Feature vs Target Studio
    # =========================================================================
    with tabs[2]:
        st.subheader("🎯 Feature vs Target Relationship Studio")
        st.write("Understand how input features drive and influence your target variable for Machine Learning.")

        tc1, tc2 = st.columns([2, 1.5])
        with tc1:
            target_var = st.selectbox("Select Target Variable (Dependent Variable)", all_columns, index=len(all_columns)-1, key="target_studio_var")
        
        target_series = df[target_var].dropna()
        prob_type = detect_problem_type(target_series)
        
        with tc2:
            st.markdown(
                f'''
                <div style="background:var(--surface); border:1px solid var(--border-bright); border-radius:12px; padding:0.75rem 1.1rem; margin-top:0.4rem;">
                    <b>Target:</b> <code style="color:var(--secondary);">{target_var}</code><br>
                    <b>Detected Task:</b> <span class="rec-badge info">{prob_type.upper()}</span>
                </div>
                ''',
                unsafe_allow_html=True
            )

        st.markdown("---")

        if prob_type == "Classification":
            st.markdown(f"#### 🎯 Target Class Distribution (`{target_var}`)")
            
            t_col1, t_col2 = st.columns([1.5, 2.5])
            with t_col1:
                class_counts = target_series.astype(str).value_counts().reset_index()
                class_counts.columns = ["Class", "Count"]
                class_counts["Percentage"] = (class_counts["Count"] / len(target_series)) * 100
                
                fig_t_pie = px.pie(
                    class_counts, names="Class", values="Count", hole=0.45,
                    title="Class Balance Ratio", color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_t_pie.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_t_pie, use_container_width=True)

                # Class Imbalance Alert
                min_class_pct = class_counts["Percentage"].min()
                if min_class_pct < 20.0 and len(class_counts) > 1:
                    st.warning(f"⚠️ Imbalanced classes detected: Minority class is **{min_class_pct:.1f}%** of data.")

            with t_col2:
                fig_t_bar = px.bar(
                    class_counts, x="Class", y="Count", color="Class",
                    text=class_counts["Percentage"].apply(lambda v: f"{v:.1f}%"),
                    title="Class Frequency Distribution"
                )
                fig_t_bar.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_t_bar, use_container_width=True)

            st.markdown("#### 🔬 Input Features vs Target Classes")
            feature_candidates = [c for c in all_columns if c != target_var]
            if feature_candidates:
                comp_feat = st.selectbox("Select Feature to Compare against Target", feature_candidates, key="comp_feat_clf")
                
                if comp_feat in numeric:
                    cf1, cf2 = st.columns(2)
                    with cf1:
                        fig_box_t = px.box(
                            df, x=target_var, y=comp_feat, color=target_var, points="outliers",
                            title=f"<b>{comp_feat}</b> Distribution across Target Classes"
                        )
                        fig_box_t.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                        st.plotly_chart(fig_box_t, use_container_width=True)

                    with cf2:
                        fig_hist_t = px.histogram(
                            df, x=comp_feat, color=target_var, barmode="overlay", marginal="rug",
                            opacity=0.7, title=f"KDE Density of <b>{comp_feat}</b> by Class"
                        )
                        fig_hist_t.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                        st.plotly_chart(fig_hist_t, use_container_width=True)
                else:
                    crosstab = pd.crosstab(df[comp_feat], df[target_var], normalize="index") * 100
                    fig_stack = px.bar(
                        crosstab, barmode="stack", title=f"100% Stacked Proportion of <b>{comp_feat}</b> vs Target",
                        labels={"value": "Proportion (%)", "index": comp_feat}
                    )
                    fig_stack.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_stack, use_container_width=True)

        else: # Regression Target
            st.markdown(f"#### 🎯 Continuous Target Distribution (`{target_var}`)")
            reg1, reg2 = st.columns([1.5, 2.5])
            with reg1:
                stats = _get_column_stats(df, target_var)
                _render_stats_card(stats, target_var)

            with reg2:
                fig_t_reg = px.histogram(
                    df, x=target_var, marginal="box", nbins=40,
                    title="Target Value Distribution & Spread", color_discrete_sequence=["#22d3ee"]
                )
                fig_t_reg.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_t_reg, use_container_width=True)

            st.markdown("#### 🔬 Input Features vs Target Variable (Scatter & Correlations)")
            num_feats = [c for c in numeric if c != target_var]
            if num_feats:
                # Feature correlation ranking
                corrs = df[num_feats].apply(lambda col: df[target_var].corr(col)).sort_values()
                corr_ranking_df = pd.DataFrame({"Feature": corrs.index, "Pearson Correlation (r)": corrs.values})

                rc1, rc2 = st.columns([1.8, 2.2])
                with rc1:
                    fig_rank = px.bar(
                        corr_ranking_df, x="Pearson Correlation (r)", y="Feature", orientation="h",
                        title=f"Feature Correlation Ranking with Target `{target_var}`",
                        color="Pearson Correlation (r)", color_continuous_scale="RdBu_r", range_color=[-1, 1]
                    )
                    fig_rank.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_rank, use_container_width=True)

                with rc2:
                    selected_scat_feat = st.selectbox("Select Feature for Scatter Correlation", num_feats, index=len(num_feats)-1, key="scat_reg_feat")
                    r_val = df[target_var].corr(df[selected_scat_feat])
                    clean_reg = df.dropna(subset=[selected_scat_feat, target_var])
                    fig_scat_reg = px.scatter(
                        clean_reg, x=selected_scat_feat, y=target_var, trendline="ols",
                        title=f"<b>{selected_scat_feat}</b> vs <b>{target_var}</b> (r = {r_val:.2f})",
                        opacity=0.75, color_discrete_sequence=["#6d5dfc"]
                    )
                    fig_scat_reg.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_scat_reg, use_container_width=True)

    # =========================================================================
    # TAB 4: Data Quality & Missing Patterns
    # =========================================================================
    with tabs[3]:
        st.subheader("🩹 Data Quality & Missingness Visualizer")
        st.write("Examine data completeness, missingness co-occurrence patterns, and distinct cardinality distributions.")

        missing_counts = df.isna().sum()
        missing_pct = (missing_counts / len(df)) * 100
        miss_df = pd.DataFrame({"Feature": df.columns, "Missing Count": missing_counts.values, "Missing %": missing_pct.values})
        miss_df = miss_df[miss_df["Missing Count"] > 0].sort_values("Missing %", ascending=True)

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            if not miss_df.empty:
                fig_miss_bar = px.bar(
                    miss_df, x="Missing %", y="Feature", orientation="h",
                    text=miss_df["Missing %"].apply(lambda v: f"{v:.1f}% ({int(miss_df.loc[miss_df['Missing %'] == v, 'Missing Count'].values[0]):,})"),
                    title="Missing Values Ratio (%) by Column",
                    color="Missing %", color_continuous_scale="Reds"
                )
                fig_miss_bar.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_miss_bar, use_container_width=True)
            else:
                st.success("🎉 Perfect Data Health: 0 missing values across all features!")

        with q_col2:
            card_df = pd.DataFrame({"Feature": df.columns, "Unique Values": [df[c].nunique() for c in df.columns]}).sort_values("Unique Values", ascending=True)
            fig_card = px.bar(
                card_df, x="Unique Values", y="Feature", orientation="h",
                title="Distinct Value Cardinality per Feature",
                color="Unique Values", color_continuous_scale="Teal"
            )
            fig_card.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_card, use_container_width=True)

        if not miss_df.empty and len(miss_df) >= 2:
            st.markdown("#### 🔗 Missing Value Co-occurrence Correlation")
            st.write("Identify whether missing values in one feature predict or co-occur with missing values in another feature.")
            null_corr = df[miss_df["Feature"].tolist()].isna().corr()
            fig_null_corr = px.imshow(
                null_corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="Blues", zmin=-1, zmax=1,
                title="Missingness Null Correlation Matrix"
            )
            fig_null_corr.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_null_corr, use_container_width=True)
