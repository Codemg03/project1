import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import timedelta

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Care Forecast Dashboard", layout="wide")

# =========================
# SIDEBAR STYLING (BLACK THEME)
# =========================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #000000;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        color: white !important;
    }

    .stSlider label, .stCheckbox label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    with open("care_forecast_models.pkl", "rb") as f:
        return pickle.load(f)

models = load_models()
rf_model = models.get("rf_model")
gbr_model = models.get("gbr_model")
sarima_model = models.get("sarima_model")
ets_model = models.get("ets_model")
features = models.get("features", [])
capacity_limit = models.get("capacity_limit", 2500)
surge_threshold = models.get("surge_threshold", 2450)
model_metrics = models.get("model_metrics", None)

# =========================
# LOAD PREDICTIONS
# =========================
@st.cache_resource
def load_all_predictions():
    with open("all_model_predictions.pkl", "rb") as f:
        return pickle.load(f)

all_predictions = load_all_predictions()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("historical_data.csv")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "net_pressure" not in df.columns:
        df["net_pressure"] = (
            df["Children transferred out of CBP custody"]
            - df["Children discharged from HHS Care"]
        )

    return df

df = load_data()

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("Forecast Controls")

model_choice = st.sidebar.selectbox(
    "Select Forecast Model",
    ["Naïve Persistence", "Random Forest", "Gradient Boosting", "SARIMA", "ETS"]
)

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

show_test_predictions = st.sidebar.checkbox(
    "Show Test Predictions",
    value=True
)

# =========================
# RECURSIVE FORECAST FUNCTION
# =========================
def recursive_forecast(model, df_hist, horizon, feature_list):
    df_temp = df_hist.copy()
    forecasts, net_pressure_forecast = [], []

    for _ in range(horizon):
        last_row = df_temp.iloc[-1]
        next_date = last_row["Date"] + timedelta(days=1)

        lag_1 = last_row["Children in HHS Care"]
        lag_7 = df_temp["Children in HHS Care"].iloc[-7] if len(df_temp) >= 7 else lag_1
        lag_14 = df_temp["Children in HHS Care"].iloc[-14] if len(df_temp) >= 14 else lag_1

        new_features = {
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "roll_mean_7": df_temp["Children in HHS Care"].tail(7).mean(),
            "roll_mean_14": df_temp["Children in HHS Care"].tail(14).mean(),
            "roll_var_7": df_temp["Children in HHS Care"].tail(7).var(),
            "roll_var_14": df_temp["Children in HHS Care"].tail(14).var(),
            "net_pressure": last_row["net_pressure"],
            "day_of_week": next_date.dayofweek,
            "is_weekend": 1 if next_date.dayofweek >= 5 else 0,
            "month": next_date.month
        }

        X_new = pd.DataFrame([new_features])[feature_list]
        y_pred = model.predict(X_new)[0]

        transfers = df_temp["Children transferred out of CBP custody"].tail(7).mean()
        discharges = df_temp["Children discharged from HHS Care"].tail(7).mean()
        net_pressure_value = transfers - discharges

        forecasts.append(y_pred)
        net_pressure_forecast.append(net_pressure_value)

        df_temp = pd.concat([df_temp, pd.DataFrame({
            "Date": [next_date],
            "Children in HHS Care": [y_pred],
            "Children transferred out of CBP custody": [transfers],
            "Children discharged from HHS Care": [discharges],
            "net_pressure": [net_pressure_value]
        })], ignore_index=True)

    return np.array(forecasts), np.array(net_pressure_forecast)

# =========================
# FORECAST LOGIC
# =========================
last_value = df["Children in HHS Care"].iloc[-1]
last_date = df["Date"].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]

if model_choice == "Naïve Persistence":
    forecast = np.repeat(last_value, forecast_horizon)
    net_pressure_forecast = np.repeat(
        df["Children transferred out of CBP custody"].tail(7).mean()
        - df["Children discharged from HHS Care"].tail(7).mean(),
        forecast_horizon
    )

elif model_choice == "Random Forest":
    forecast, net_pressure_forecast = recursive_forecast(rf_model, df, forecast_horizon, features)

elif model_choice == "Gradient Boosting":
    forecast, net_pressure_forecast = recursive_forecast(gbr_model, df, forecast_horizon, features)

elif model_choice == "SARIMA":
    forecast = sarima_model.forecast(steps=forecast_horizon)
    net_pressure_forecast = np.repeat(
        df["Children transferred out of CBP custody"].tail(7).mean()
        - df["Children discharged from HHS Care"].tail(7).mean(),
        forecast_horizon
    )

elif model_choice == "ETS":
    forecast = ets_model.forecast(forecast_horizon)
    net_pressure_forecast = np.repeat(
        df["Children transferred out of CBP custody"].tail(7).mean()
        - df["Children discharged from HHS Care"].tail(7).mean(),
        forecast_horizon
    )

forecast = np.array(forecast)

# =========================
# CONFIDENCE INTERVAL
# =========================
volatility = df["Children in HHS Care"].tail(60).std()
upper = forecast + 1.96 * volatility
lower = forecast - 1.96 * volatility

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📈 Forecast Dashboard", "📊 Model Performance"])

# =========================
# TAB 1
# =========================
with tab1:
    st.title("Predictive Forecasting of Care Load & Net Pressure")

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Latest Forecast", f"{forecast[-1]:.0f}")
    k2.metric("Average Forecast", f"{forecast.mean():.0f}")
    k3.metric("Surge Days", f"{np.sum(forecast > surge_threshold)}")
    k4.metric("Capacity Breach Days", f"{np.sum(forecast > capacity_limit)}")
    k5.metric("Latest Net Pressure", f"{net_pressure_forecast[-1]:.0f}")

    # CARE FORECAST
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Children in HHS Care"], name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast"))
    fig.add_trace(go.Scatter(x=future_dates, y=upper, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=future_dates, y=lower, fill="tonexty", name="Confidence"))

    fig.add_hline(y=surge_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=capacity_limit, line_dash="dash", line_color="black")

    st.plotly_chart(fig, use_container_width=True)

    # NET PRESSURE
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Date"], y=df["net_pressure"], name="Historical"))
    fig2.add_trace(go.Scatter(x=future_dates, y=net_pressure_forecast, name="Forecast"))

    st.plotly_chart(fig2, use_container_width=True)

    # TEST PREDICTIONS
    if show_test_predictions:
        st.subheader("Test Predictions Across Models")

        if all_predictions:
            selected_model = st.selectbox("Select Model", list(all_predictions.keys()))

            max_len = min(30, len(all_predictions[selected_model]))
            dates = df["Date"].iloc[-max_len:]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=dates,
                y=df["Children in HHS Care"].iloc[-max_len:],
                name="Actual"
            ))
            fig3.add_trace(go.Scatter(
                x=dates,
                y=all_predictions[selected_model][-max_len:],
                name="Predicted"
            ))

            st.plotly_chart(fig3, use_container_width=True)

# =========================
# TAB 2
# =========================
with tab2:
    st.title("Model Performance")

    if model_metrics:
        perf_df = pd.DataFrame(model_metrics)
        st.dataframe(perf_df)

        best = perf_df.sort_values("RMSE").iloc[0]["Model"]
        st.success(f" 🏆 Best Model: {best}")



        fig = go.Figure()
        fig.add_trace(go.Bar(x=perf_df["Model"], y=perf_df["RMSE"]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model metrics found.")