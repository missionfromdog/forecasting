import streamlit as st
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, Naive, SeasonalNaive, RandomWalkWithDrift, Theta
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go
from pandas import concat
import traceback
import io

# Load custom fonts, logo, and styles
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;700&display=swap" rel="stylesheet">
    <style>
        .adv-header {
            background: linear-gradient(90deg, #fd6b2f 0%, #8624f5 100%);
            padding: 2rem 1rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .adv-header img {
            height: 64px;
            border-radius: 8px;
        }
        .adv-header-text h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            color: white;
            margin: 0;
        }
        .adv-header-text p {
            font-family: 'Poppins', sans-serif;
            font-weight: 300;
            color: #f0f0f0;
            margin: 0.2rem 0 0;
            font-size: 1.1rem;
        }
        .nav-bar {
            display: flex;
            gap: 2rem;
            padding: 1rem 0 2rem;
            border-bottom: 1px solid #333;
            margin-bottom: 2rem;
        }
        .nav-bar a {
            font-family: 'Poppins', sans-serif;
            color: #fd6b2f;
            text-decoration: none;
            font-weight: 600;
            transition: 0.2s ease;
        }
        .nav-bar a:hover {
            color: #ff924f;
        }
    </style>
    <div class="adv-header">
        <img src="https://www.youradv.com/wp-content/uploads/2024/11/advantage-meta.png" alt="ADV Logo" />
        <div class="adv-header-text">
            <h1>ADV Forecast Studio</h1>
            <p>Upload. Analyze. Predict. Powered by StatsForecast.</p>
        </div>
    </div>
    <div class="nav-bar">
        <a href="#upload">Upload</a>
        <a href="#settings">Settings</a>
        <a href="#results">Results</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
### <span id="upload">Upload Requirements</span>
Upload your time series CSV file with **two required columns**:

- A **date column** named `ds` that contains timestamps (daily, weekly, or monthly cadence).
- A **numeric value column** (e.g., `sales`, `demand`) to forecast.

The app ensembles several StatsForecast models, picks the best via backtesting, and shows the forecast (with an optional 95 % confidence band).
""",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

st.markdown("### <span id=\"settings\">Settings</span>", unsafe_allow_html=True)

freq = st.selectbox(
    "Select data frequency:",
    options=["D", "W", "M"],
    index=1,
    format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x],
)

horizon = st.number_input(
    "Forecast horizon (periods):", min_value=1, max_value=100, value=12, step=1
)

season_length_map = {"D": 7, "W": 52, "M": 12}
season_length = season_length_map.get(freq, 12)

st.markdown("### <span id=\"results\"></span>", unsafe_allow_html=True)

# Load and process uploaded CSV
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Auto-detect columns
    datetime_col = "ds"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) > 1:
        st.warning("Multiple numeric columns detected. Using the first one: " + numeric_cols[0])

    target_col = numeric_cols[0] if numeric_cols else None

    if datetime_col not in df.columns or target_col is None:
        st.error("CSV must have a 'ds' datetime column and at least one numeric value column.")
        st.stop()

    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["ds", target_col]].rename(columns={target_col: "y"})
    df = df.sort_values("ds")

    # Drop duplicates and fill gaps
    df = df.groupby("ds", as_index=False).sum()
    df = df.set_index("ds").asfreq(freq).reset_index()
    df = df.dropna()
    df["unique_id"] = "main_series"

    models = [
        AutoARIMA(),
        Naive(),
        SeasonalNaive(season_length=season_length),
        RandomWalkWithDrift(),
        Theta(),
    ]

    valid_models = []
    backtest_dfs = []

    for model in models:
        try:
            sf = StatsForecast(models=[model], freq=freq)
            bt_df = sf.cross_validation(df=df, h=horizon, level=[95])
            bt_df["model"] = model.__class__.__name__
            bt_df = bt_df.rename(columns={model.__class__.__name__: "y_hat"})
            backtest_dfs.append(bt_df)
            valid_models.append(model)
        except Exception as e:
            st.warning(f"{model.__class__.__name__} failed during backtesting.")

    if not backtest_dfs:
        st.error("All models failed during backtesting.")
        st.stop()

    from sklearn.metrics import mean_absolute_error
    all_bt = pd.concat(backtest_dfs, ignore_index=True)
  
    model_metrics = all_bt.dropna(subset=["y", "y_hat"]).copy()
    model_metrics = model_metrics.groupby("model").apply(lambda x: pd.Series({
        "MAE": mean_absolute_error(x["y"].astype(float), x["y_hat"].astype(float)),
        "RMSE": np.sqrt(((x["y"].astype(float) - x["y_hat"].astype(float)) ** 2).mean()),
        "MAPE": (abs((x["y"] - x["y_hat"]) / x["y"]).replace([np.inf, -np.inf], np.nan).dropna().mean()) * 100
    }))

    best_model_name = model_metrics["MAE"].idxmin()
    best_model = next((m for m in valid_models if m.__class__.__name__ == best_model_name), None)

    st.success(f"Best model selected based on MAE: {best_model_name} (MAE = {model_metrics.loc[best_model_name, 'MAE']:.2f})")

    sf_final = StatsForecast(models=[best_model], freq=freq)
    forecast_df = sf_final.forecast(df=df, h=horizon, level=[95])
    forecast_df = forecast_df.rename(columns={best_model_name: "y_hat"})
    forecast_df = forecast_df.merge(df, on="ds", how="outer")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y"], name="Actual", mode="lines", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y_hat"], name=f"Forecast ({best_model_name})", mode="lines", line=dict(color="#ff7f0e")))

    if "lo-95" in forecast_df.columns and "hi-95" in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["hi-95"], name="Upper 95%", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["lo-95"], name="Lower 95%", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=True))

    fig.update_layout(title="Forecast Results (Best Model)", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    # Compare all models
    show_ci = st.checkbox("Show confidence intervals for all models", value=True)
    st.markdown("#### Forecast Comparison Across Models")
    compare_fig = go.Figure()
    ci_columns = [col for col in all_bt.columns if col.startswith("lo-95") or col.startswith("hi-95")]
    for model_name in all_bt["model"].unique():
        compare_df = all_bt[all_bt["model"] == model_name]
        compare_fig.add_trace(go.Scatter(x=compare_df["ds"], y=compare_df["y_hat"], name=f"{model_name} Forecast", mode="lines"))
        if show_ci and "hi-95" in compare_df.columns and "lo-95" in compare_df.columns:
            compare_fig.add_trace(go.Scatter(x=compare_df["ds"], y=compare_df["hi-95"], name=f"{model_name} Upper 95%", line=dict(width=0), showlegend=False))
            compare_fig.add_trace(go.Scatter(x=compare_df["ds"], y=compare_df["lo-95"], name=f"{model_name} Lower 95%", fill='tonexty', line=dict(width=0), fillcolor='rgba(255,127,14,0.15)', showlegend=False))

    compare_fig.add_trace(go.Scatter(x=all_bt["ds"], y=all_bt["y"], name="Actual", mode="lines+markers", line=dict(color="#1f77b4", dash="dot")))
    compare_fig.update_layout(title="Backtest Forecasts by Model", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(compare_fig, use_container_width=True)

    st.markdown("#### Model Performance")
    st.dataframe(model_metrics.reset_index().round(2))
