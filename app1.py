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
# SIDEBAR STYLE
# =========================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# SAFE STATE INIT (PREVENT RE-RUN ISSUES)
# =========================
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

# =========================
# LOAD DATA (FAST + CACHED)
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
# LOAD MODELS (LAZY + CACHED)
# =========================
@st.cache_resource
def load_models():
    with open("care_forecast_models.pkl", "rb") as f:
        return pickle.load(f)

models = None

def get_models():
    global models
    if models is None:
        models = load_models()
    return models

# =========================
# LOAD PREDICTIONS (SAFE)
# =========================
@st.cache_resource
def load_all_predictions():
    with open("all_model_predictions.pkl", "rb") as f:
        return pickle.load(f)

all_predictions = load_all_predictions()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Forecast Controls")

model_choice = st.sidebar.selectbox(
    "Select Forecast Model",
    ["Naïve Persistence", "Random Forest", "Gradient Boosting", "SARIMA", "ETS"]
)

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

show_test_predictions = st.sidebar.checkbox("Show Test Predictions", True)

# =========================
# FAST RECURSIVE FORECAST (OPTIMIZED)
# =========================
def recursive_forecast(model, df_hist, horizon, feature_list):
    df_temp = df_hist.copy()

    forecasts = []
    net_pressure_forecast = []

    for _ in range(horizon):

        last_row = df_temp.iloc[-1]
        next_date = last_row["Date"] + timedelta(days=1)

        lag_1 = last_row["Children in HHS Care"]
        lag_7 = df_temp["Children in HHS Care"].iloc[-7] if len(df_temp) >= 7 else lag_1
        lag_14 = df_temp["Children in HHS Care"].iloc[-14] if len(df_temp) >= 14 else lag_1

        features_dict = {
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "roll_mean_7": df_temp["Children in HHS Care"].tail(7).mean(),
            "roll_mean_14": df_temp["Children in HHS Care"].tail(14).mean(),
            "roll_var_7": df_temp["Children in HHS Care"].tail(7).var(),
            "roll_var_14": df_temp["Children in HHS Care"].tail(14).var(),
            "net_pressure": last_row["net_pressure"],
            "day_of_week": next_date.dayofweek,
            "is_weekend": int(next_date.dayofweek >= 5),
            "month": next_date.month
        }

        X_new = pd.DataFrame([features_dict])[feature_list]
        y_pred = model.predict(X_new)[0]

        forecasts.append(y_pred)

        net_pressure_forecast.append(
            df_temp["Children transferred out of CBP custody"].tail(7).mean()
            - df_temp["Children discharged from HHS Care"].tail(7).mean()
        )

        # FAST APPEND (NO CONCAT)
        df_temp.loc[len(df_temp)] = {
            "Date": next_date,
            "Children in HHS Care": y_pred,
            "Children transferred out of CBP custody": df_temp["Children transferred out of CBP custody"].mean(),
            "Children discharged from HHS Care": df_temp["Children discharged from HHS Care"].mean(),
            "net_pressure": net_pressure_forecast[-1]
        }

    return np.array(forecasts), np.array(net_pressure_forecast)

# =========================
# FORECAST LOGIC (SAFE MODEL ACCESS)
# =========================
last_value = df["Children in HHS Care"].iloc[-1]
last_date = df["Date"].iloc[-1]

future_dates = [
    last_date + timedelta(days=i)
    for i in range(1, forecast_horizon + 1)
]

models = get_models()

if model_choice == "Naïve Persistence":
    forecast = np.repeat(last_value, forecast_horizon)

elif model_choice == "Random Forest":
    forecast, _ = recursive_forecast(
        models["rf_model"], df, forecast_horizon, models["features"]
    )

elif model_choice == "Gradient Boosting":
    forecast, _ = recursive_forecast(
        models["gbr_model"], df, forecast_horizon, models["features"]
    )

elif model_choice == "SARIMA":
    forecast = models["sarima_model"].forecast(steps=forecast_horizon)

elif model_choice == "ETS":
    forecast = models["ets_model"].forecast(forecast_horizon)

forecast = np.array(forecast)

net_pressure_forecast = np.repeat(df["net_pressure"].mean(), forecast_horizon)

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
    st.title("Care Load Forecast Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Latest Forecast", f"{forecast[-1]:.0f}")
    c2.metric("Average Forecast", f"{forecast.mean():.0f}")
    c3.metric("Surge Days", f"{np.sum(forecast > 2450)}")
    c4.metric("Capacity Breach", f"{np.sum(forecast > 2500)}")
    c5.metric("Net Pressure", f"{net_pressure_forecast[-1]:.0f}")

    # =========================
    # MAIN FORECAST PLOT
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Children in HHS Care"],
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        name="Forecast"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        fill="tonexty",
        name="Confidence"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # NET PRESSURE
    # =========================
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df["Date"],
        y=df["net_pressure"],
        name="Historical"
    ))

    fig2.add_trace(go.Scatter(
        x=future_dates,
        y=net_pressure_forecast,
        name="Forecast"
    ))

    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # TEST PREDICTIONS
    # =========================
    if show_test_predictions and all_predictions:

        selected_model = st.selectbox(
            "Select Model",
            list(all_predictions.keys())
        )

        preds = np.array(all_predictions[selected_model])

        max_len = min(30, len(preds))

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=df["Date"].iloc[-max_len:],
            y=df["Children in HHS Care"].iloc[-max_len:],
            name="Actual"
        ))

        fig3.add_trace(go.Scatter(
            x=df["Date"].iloc[-max_len:],
            y=preds[-max_len:],
            name="Predicted"
        ))

        st.plotly_chart(fig3, use_container_width=True)

# =========================
# TAB 2
# =========================
with tab2:
    st.title("Model Performance")

    if "model_metrics" in models:
        perf_df = pd.DataFrame(models["model_metrics"])

        st.dataframe(perf_df)

        best = perf_df.sort_values("RMSE").iloc[0]["Model"]
        st.success(f"🏆 Best Model: {best}")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=perf_df["Model"], y=perf_df["RMSE"]))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No model metrics found.")