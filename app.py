import json
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
from curl_cffi import requests as curl_requests


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="DOGE CNN-LSTM Predictor", layout="wide")
st.title("🐶 Dogecoin Next-Day Close Prediction (CNN-LSTM)")
st.caption(
    "Live data → feature engineering → scaling → CNN-LSTM inference → visualization"
)


# -----------------------------
# Load artifacts (model + scaler + config)
# -----------------------------
@st.cache_resource
def load_artifacts(artifacts_dir: str = "doge_deploy_artifacts"):
    model_path = os.path.join(artifacts_dir, "best_cnn_lstm.keras")
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    config_path = os.path.join(artifacts_dir, "config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    return model, scaler, config


# -----------------------------
# Live fetch DOGE-USD from Yahoo Finance
# -----------------------------
@st.cache_data(ttl=60 * 60)
def fetch_doge_data(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    # Force a supported impersonation target.
    # "chrome" maps to the latest supported target in your curl_cffi build.
    session = curl_requests.Session(impersonate="chrome")

    df = yf.download(
        "DOGE-USD",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        session=session,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [str(c).strip() for c in df.columns]
    return df



# -----------------------------
# Feature engineering to match training
# -----------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Date exists and is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Create MA7 and MA30 on Close
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()

    # Drop rows where MAs are NaN (early rows)
    df = df.dropna().reset_index(drop=True)
    return df


def predict_next_close(df_feat: pd.DataFrame, model, scaler, config) -> float:
    features = config["features"]  # ['Open','High','Low','Close','Volume','MA7','MA30']
    seq_len = int(config["sequence_length"])  # 60
    close_idx = int(config["close_index"])  # 3

    if len(df_feat) < seq_len:
        raise ValueError(
            f"Not enough rows after feature engineering. Need at least {seq_len}, got {len(df_feat)}."
        )

    # Select feature matrix in correct order
    X_mat = df_feat[features].values
    X_scaled = scaler.transform(X_mat)

    # Take last 60 timesteps
    x_last = X_scaled[-seq_len:]  # (60, 7)
    x_last = np.expand_dims(x_last, axis=0)  # (1, 60, 7)

    pred_scaled = model.predict(x_last, verbose=0)[0, 0]

    # Inverse scale only the Close dimension using scaler min/max for that feature
    close_min = scaler.data_min_[close_idx]
    close_max = scaler.data_max_[close_idx]
    pred_close = pred_scaled * (close_max - close_min) + close_min
    return float(pred_close)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")
period = st.sidebar.selectbox(
    "History window to fetch", ["1y", "2y", "5y", "max"], index=2
)
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)
show_rows = st.sidebar.slider("Rows to display", 5, 200, 50)

artifacts_dir = st.sidebar.text_input("Artifacts folder", "doge_deploy_artifacts")


# -----------------------------
# Check Artifact Status
# -----------------------------
try:
    model, scaler, config = load_artifacts(artifacts_dir)
except Exception as e:
    st.error(f"Failed to load deployment artifacts: {e}")
    st.stop()

raw = fetch_doge_data(period=period, interval=interval)

# Normalize yfinance column names if they include the ticker
rename_map = {}
for col in raw.columns:
    if col.startswith(("Open_", "High_", "Low_", "Close_", "Adj Close_", "Volume_")):
        base = col.split("_", 1)[0]  # "Open_DOGE-USD" -> "Open"
        rename_map[col] = base

raw = raw.rename(columns=rename_map)

if raw.empty:
    st.error(
        "No data returned from the data source (yfinance). Try again later or change period."
    )
    st.stop()

feat = prepare_features(raw)

# Show dataframe
st.subheader("Live Dogecoin Data (engineered features included)")
st.dataframe(feat.tail(show_rows), use_container_width=True)

# Candlestick chart
st.subheader("Candlestick Chart (DOGE-USD)")
cand = go.Figure(
    data=[
        go.Candlestick(
            x=feat["Date"],
            open=feat["Open"],
            high=feat["High"],
            low=feat["Low"],
            close=feat["Close"],
            name="OHLC",
        )
    ]
)
cand.update_layout(height=450, xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(cand, use_container_width=True)

# Close + MAs chart
st.subheader("Close Price + Moving Averages")
close_fig = go.Figure()
close_fig.add_trace(
    go.Scatter(x=feat["Date"], y=feat["Close"], mode="lines", name="Close")
)
close_fig.add_trace(go.Scatter(x=feat["Date"], y=feat["MA7"], mode="lines", name="MA7"))
close_fig.add_trace(
    go.Scatter(x=feat["Date"], y=feat["MA30"], mode="lines", name="MA30")
)
close_fig.update_layout(height=350, xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(close_fig, use_container_width=True)

# Prediction
st.subheader("Next-Day Close Prediction")
try:
    pred_close = predict_next_close(feat, model, scaler, config)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

last_date = feat["Date"].iloc[-1]
pred_date = last_date + timedelta(days=1)

col1, col2, col3 = st.columns(3)
col1.metric("Last Available Date", last_date.date().isoformat())
col2.metric("Last Close (USD)", f"{feat['Close'].iloc[-1]:.6f}")
col3.metric("Predicted Next Close (USD)", f"{pred_close:.6f}")

# Plot last 60 days + predicted next day
st.subheader("Last 60 Days + Predicted Next Day")
seq_len = int(config["sequence_length"])
plot_df = feat.tail(seq_len).copy()

pred_point = pd.DataFrame({"Date": [pred_date], "Close": [pred_close]})
line_fig = go.Figure()
line_fig.add_trace(
    go.Scatter(
        x=plot_df["Date"],
        y=plot_df["Close"],
        mode="lines+markers",
        name="Last 60 Close",
    )
)
line_fig.add_trace(
    go.Scatter(
        x=pred_point["Date"],
        y=pred_point["Close"],
        mode="markers",
        name="Predicted Next Close",
    )
)
line_fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(line_fig, use_container_width=True)

# Optional: allow download of last 60 + prediction as CSV
out = pd.concat([plot_df[["Date", "Close"]], pred_point], ignore_index=True)
csv = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download last 60 days + prediction (CSV)",
    data=csv,
    file_name="doge_last60_plus_prediction.csv",
    mime="text/csv",
)
