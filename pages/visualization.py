import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.app_state import get_dataset
from src.ui import page_header, render_dataset_toolbar

CHARTS = [
    "Histogram", "Box Plot", "Violin Plot", "Bar Chart", "Pie / Donut Chart",
    "Scatter Plot", "3D Scatter Plot", "Line Chart", "Area Chart", "Density Histogram",
    "Correlation Heatmap", "Bubble Chart", "Sunburst", "Treemap",
    "Strip Plot", "ECDF", "Parallel Coordinates", "Scatter Matrix", "Missing Value Matrix"
]

COLOR_SCALES = ["viridis", "plasma", "turbo", "inferno", "magma", "RdBu", "cividis", "spectral"]

def _aggregate(df, x, y, agg_func="mean", top_n=30):
    if y is None or agg_func == "count":
        res = df[x].astype(str).value_counts(dropna=False).head(top_n).rename_axis(x).reset_index(name="Value")
    else:
        res = df.groupby(x, dropna=False)[y].agg(agg_func).reset_index().head(top_n).rename(columns={y: "Value"})
    return res

def render():
    page_header("📊 Visualization Studio", "Interactive high-resolution data visualization studio with custom themes and export options.")
    df = get_dataset()
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    render_dataset_toolbar()

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    all_columns = df.columns.tolist()

    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.2, 1.2])
    with c1:
        chart_type = st.selectbox("Chart Type", CHARTS, key="vis_chart_type")
    with c2:
        x = st.selectbox("Primary Axis (X / Main)", all_columns, key="vis_x")
    with c3:
        color = st.selectbox("Color / Group", ["None"] + all_columns, key="vis_color")
    with c4:
        palette = st.selectbox("Color Palette", COLOR_SCALES, key="vis_palette")

    color_arg = None if color == "None" else color

    y = None
    z = None
    if chart_type in ["Scatter Plot", "3D Scatter Plot", "Line Chart", "Area Chart", "Bubble Chart", "Bar Chart"]:
        candidates = numeric if numeric else all_columns
        y = st.selectbox("Y Column", candidates, key="vis_y")

    if chart_type == "3D Scatter Plot":
        z = st.selectbox("Z Column (3D)", numeric, key="vis_z")

    with st.expander("⚙️ Advanced Plot Options & Aesthetics", expanded=False):
        o1, o2, o3, o4 = st.columns(4)
        top_n = o1.slider("Top Categories Limit", 5, 100, 30, 5, key="vis_topn")
        orientation = o2.selectbox("Orientation", ["Vertical", "Horizontal"], key="vis_orient")
        title = o3.text_input("Custom Title", value=f"{chart_type}: {x}" + (f" vs {y}" if y else ""), key="vis_title")
        template = o4.selectbox("Theme Template", ["plotly_dark", "plotly", "simple_white", "ggplot2", "seaborn"], key="vis_theme")

        opt1, opt2, opt3, opt4 = st.columns(4)
        log_scale = opt1.checkbox("Logarithmic Y-Scale", key="vis_log")
        show_points = opt2.checkbox("Show Points / Markers", value=True, key="vis_pts")
        facet_col_sel = opt3.selectbox("Facet Subplots Column", ["None"] + all_columns, key="vis_facet")
        agg_func = opt4.selectbox("Bar Aggregation", ["mean", "sum", "count", "median", "max", "min"], key="vis_agg")

    facet_arg = None if facet_col_sel == "None" else facet_col_sel
    fig = None

    try:
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x, color=color_arg, nbins=40, marginal="box", facet_col=facet_arg, color_discrete_sequence=px.colors.sequential.__dict__.get(palette, None))
        elif chart_type == "Density Histogram":
            fig = px.histogram(df, x=x, color=color_arg, histnorm="probability density", opacity=0.75, facet_col=facet_arg)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=color_arg if color_arg else None, y=x if x in numeric else None, points="all" if show_points else False, facet_col=facet_arg)
            if x not in numeric:
                st.info("For a useful Box Plot, select a numeric feature.")
        elif chart_type == "Violin Plot":
            if x not in numeric:
                st.warning("Violin Plot requires a numeric main column.")
                return
            fig = px.violin(df, x=color_arg if color_arg else None, y=x, box=True, points="all" if show_points else False, facet_col=facet_arg)
        elif chart_type == "Bar Chart":
            plot_df = _aggregate(df, x, y, agg_func=agg_func, top_n=top_n)
            if orientation == "Vertical":
                fig = px.bar(plot_df, x=x, y="Value", color=x if color_arg is None else color_arg, title=title)
            else:
                fig = px.bar(plot_df, y=x, x="Value", orientation="h", title=title)
        elif chart_type == "Pie / Donut Chart":
            plot_df = _aggregate(df, x, y=None, agg_func="count", top_n=top_n)
            fig = px.pie(plot_df, names=x, values="Value", hole=0.4)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x, y=y, color=color_arg, facet_col=facet_arg, trendline="ols" if show_points and len(df) < 2000 else None)
        elif chart_type == "3D Scatter Plot":
            if not (x in numeric and y in numeric and z in numeric):
                st.warning("3D Scatter Plot requires 3 numeric columns (X, Y, Z).")
                return
            fig = px.scatter_3d(df, x=x, y=y, z=z, color=color_arg)
        elif chart_type == "Line Chart":
            sorted_df = df.sort_values(x) if x in numeric else df
            fig = px.line(sorted_df, x=x, y=y, color=color_arg, markers=show_points, facet_col=facet_arg)
        elif chart_type == "Area Chart":
            sorted_df = df.sort_values(x) if x in numeric else df
            fig = px.area(sorted_df, x=x, y=y, color=color_arg, facet_col=facet_arg)
        elif chart_type == "Bubble Chart":
            size_col = st.selectbox("Bubble Size Column", numeric, key="vis_size")
            fig = px.scatter(df, x=x, y=y, size=size_col, color=color_arg, facet_col=facet_arg)
        elif chart_type == "Correlation Heatmap":
            if len(numeric) < 2:
                st.warning("Correlation requires at least two numeric columns.")
                return
            corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], key="vis_corr_m")
            corr = df[numeric].corr(method=corr_method)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=palette)
        elif chart_type == "Sunburst":
            path_cols = st.multiselect("Hierarchy Columns", all_columns, default=[x] + ([color_arg] if color_arg else []), key="sunburst_path")
            if not path_cols:
                st.warning("Select at least one hierarchy column.")
                return
            fig = px.sunburst(df, path=path_cols)
        elif chart_type == "Treemap":
            path_cols = st.multiselect("Hierarchy Columns", all_columns, default=[x] + ([color_arg] if color_arg else []), key="treemap_path")
            if not path_cols:
                st.warning("Select at least one hierarchy column.")
                return
            fig = px.treemap(df, path=path_cols)
        elif chart_type == "Strip Plot":
            fig = px.strip(df, y=x if x in numeric else y, x=color_arg, color=color_arg)
        elif chart_type == "ECDF":
            fig = px.ecdf(df, x=x if x in numeric else numeric[0], color=color_arg)
        elif chart_type == "Parallel Coordinates":
            if len(numeric) < 2:
                st.warning("Requires at least 2 numeric columns.")
                return
            dims = st.multiselect("Dimensions", numeric, default=numeric[:min(5, len(numeric))], key="par_dims")
            fig = px.parallel_coordinates(df.dropna(subset=dims), dimensions=dims, color=dims[0])
        elif chart_type == "Scatter Matrix":
            matrix_dims = st.multiselect("Matrix Columns", numeric, default=numeric[:min(4, len(numeric))], key="mat_dims")
            if len(matrix_dims) < 2:
                st.warning("Select at least 2 dimensions.")
                return
            fig = px.scatter_matrix(df, dimensions=matrix_dims, color=color_arg)
        elif chart_type == "Missing Value Matrix":
            missing_mat = df.isna().astype(int)
            fig = px.imshow(missing_mat.T, color_continuous_scale="Reds", title="Missing Value Matrix (Red = Missing)", aspect="auto")

        if fig is not None:
            fig.update_layout(
                title=title,
                template=template,
                height=650,
                margin=dict(l=30, r=30, t=70, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            if log_scale and chart_type in ["Histogram", "Bar Chart", "Scatter Plot", "Line Chart", "Area Chart", "Bubble Chart"]:
                try:
                    fig.update_yaxes(type="log")
                except Exception:
                    pass

            st.plotly_chart(fig, use_container_width=True)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇ Download Interactive HTML Plot",
                    fig.to_html(include_plotlyjs="cdn"),
                    file_name=f"{chart_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            with d2:
                st.info("💡 You can interact with Plotly charts directly (zoom, pan, hover, save PNG).")

    except Exception as error:
        st.error(f"Unable to render chart with selected parameters: {error}")

    st.markdown("### Quick Data Overview")
    q1, q2, q3 = st.columns(3)
    q1.metric("Numeric Features", len(numeric))
    q2.metric("Categorical Features", len(categorical))
    q3.metric("Chart Types Available", len(CHARTS))
