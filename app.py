import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go


# Set page config
st.set_page_config(
    page_title="Nigerian Stock Market Predictor 📈", page_icon="📊", layout="wide"
)

# App title and description
st.title("📈 Nigerian Stock Market Price Prediction")
st.markdown("""
This app predicts future stock prices for Nigerian companies based on historical data.
Select a ticker and specify how many days you'd like to forecast ahead.
""")


# Function to get available tickers
@st.cache_data
def get_available_tickers():
    # Method 1: Get from model files
    model_files = []
    if os.path.exists("models"):
        model_files = [
            f.replace("_model.pkl", "")
            for f in os.listdir("models")
            if f.endswith("_model.pkl")
        ]

    # Method 2: Get from CSV if no models found
    if not model_files:
        try:
            df = pd.read_csv(os.path.join("data", "Nigerian_stock_market.csv"))
            if "Ticker" in df.columns:
                model_files = df["Ticker"].unique().tolist()
        except:
            st.error("Could not find models or data file.")
            return []

    return sorted(model_files)


@st.cache_data
def load_data(ticker):
    try:
        file_path = os.path.join("data", "Nigerian_stock_market.csv")
        df = pd.read_csv(file_path)

        # Filter data for the ticker
        df = df[df["Ticker"] == ticker]

        # Drop unnecessary columns
        if "Unnamed: 7" in df.columns:
            df.drop(columns=["Unnamed: 7"], inplace=True)
        if "Unnamed: 8" in df.columns:
            df.drop(columns=["Unnamed: 8"], inplace=True)

        # Rename columns
        if "Vol." in df.columns:
            df.rename(columns={"Vol.": "Volume(Million)"}, inplace=True)
        if "Change %" in df.columns:
            df.rename(columns={"Change %": "Change(%)"}, inplace=True)

        # Convert the 'Date' column to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Convert Price, Open, High, Low to float
        for col in ["Price", "Open", "High", "Low"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", "").astype(float)

        # Convert "Change(%)" to float
        if "Change(%)" in df.columns:
            df["Change(%)"] = (
                df["Change(%)"].astype(str).str.replace("%", "").astype(float)
            )

        # Convert "Volume(Million)" to float
        if "Volume(Million)" in df.columns:

            def convert_volume(value):
                if isinstance(value, str):
                    value = value.strip()
                    if value.endswith("M"):
                        return float(value[:-1]) * 1e6
                    elif value.endswith("K"):
                        return float(value[:-1]) * 1e3
                    else:
                        try:
                            return float(value)
                        except:
                            return np.nan
                return value

            df["Volume(Million)"] = df["Volume(Million)"].apply(convert_volume)

        # Sort by date
        df = df.sort_values("Date")

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def predict_future_prices(ticker, days=5):
    try:
        # Load the model
        model_path = os.path.join("models", f"{ticker}_model.pkl")

        if not os.path.exists(model_path):
            st.error(f"No trained model found for {ticker}.")
            return pd.DataFrame()

        # Load model
        model = joblib.load(model_path)

        # IMPORTANT: Use exactly the same features as during training
        features = ["Open", "High", "Low", "Volume(Million)", "Change(%)"]

        # Get latest data for this ticker
        df = load_data(ticker)

        if df.empty:
            st.error(f"No data available for {ticker}")
            return pd.DataFrame()

        # Check for missing features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.warning(
                f"Missing features: {', '.join(missing_features)}. Adding them with default values."
            )
            for feature in missing_features:
                df[feature] = 0.0

        # Use current date as starting point
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Generate future dates starting from today
        future_dates = [today + timedelta(days=i) for i in range(days)]

        # Create future data with small variations starting from the last row
        future_data = []

        # Get the last NON-NaN values for each feature
        last_values = []
        for feature in features:
            # Get last non-NaN value or use 0.0 as fallback
            non_nan_values = df[feature].dropna()
            if len(non_nan_values) > 0:
                last_values.append(float(non_nan_values.iloc[-1]))
            else:
                last_values.append(0.0)

        last_values = np.array(last_values)

        # Generate future feature values with small random variations
        for _ in range(days):
            # Create variations between -2% and +2%
            variations = np.random.uniform(-0.02, 0.02, size=len(features))
            # Apply variations to last values
            new_values = last_values * (1 + variations)
            # Add to future data
            future_data.append(new_values)
            # Update last values for next iteration
            last_values = new_values

        # Convert to proper numpy array
        X_future = np.array(future_data)

        # Verify no NaNs exist in the prediction data
        if np.isnan(X_future).any():
            # Replace any remaining NaNs with zeros
            X_future = np.nan_to_num(X_future, nan=0.0)
            st.warning("Some NaN values were replaced with zeros for prediction.")

        # Predict prices
        future_prices = model.predict(X_future)

        # Create results dataframe
        results = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_prices})

        return results

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        return pd.DataFrame()


# Sidebar for inputs
st.sidebar.header("📊 Forecast Settings")

# Get the list of available tickers
tickers = get_available_tickers()

if not tickers:
    st.error("No tickers available. Please check your data or models.")
    st.stop()

# Dropdown for ticker selection
selected_ticker = st.sidebar.selectbox("Select Stock Ticker 🏢", tickers)

# Slider for forecast days
forecast_days = st.sidebar.slider(
    "Forecast Days 📅", min_value=1, max_value=30, value=7
)

# Add a predict button
predict_button = st.sidebar.button("Predict Prices 🔮")

# Display historical data
st.subheader(f"📜 Historical Data for {selected_ticker}")

# Load historical data for the selected ticker
hist_data = load_data(selected_ticker)

if not hist_data.empty:
    # Show recent price history as a chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_data["Date"].tail(30),
            y=hist_data["Price"].tail(30),
            mode="lines+markers",
            name="Historical Price",
            line=dict(color="royalblue", width=2),
        )
    )
    fig.update_layout(
        title=f"Recent Price History for {selected_ticker}",
        xaxis_title="Date",
        yaxis_title="Price (₦)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show the last 5 days of data
    st.subheader(f"Latest Trading Data for {selected_ticker}")
    st.dataframe(
        hist_data.tail().sort_values("Date", ascending=False).reset_index(drop=True)
    )
else:
    st.warning(f"No historical data available for {selected_ticker}")


# Make and display predictions if the button is clicked
if predict_button:
    st.subheader(
        f"🔮 Price Predictions for {selected_ticker} (Next {forecast_days} days)"
    )

    # Show a spinner while making predictions
    with st.spinner("Generating predictions..."):
        predictions = predict_future_prices(selected_ticker, forecast_days)

    if not predictions.empty:
        # Display predictions as a chart
        fig = go.Figure()

        # Add historical data (last 30 days)
        if not hist_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist_data["Date"].tail(30),
                    y=hist_data["Price"].tail(30),
                    mode="lines",
                    name="Historical",
                    line=dict(color="royalblue", width=2),
                )
            )

        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=predictions["Date"],
                y=predictions["Predicted_Price"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="green", width=3, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            )
        )

        fig.update_layout(
            title=f"Price Forecast for {selected_ticker} (Starting {datetime.now().strftime('%Y-%m-%d')})",
            xaxis_title="Date",
            yaxis_title="Price (₦)",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display predictions as a table with formatted dates
        st.subheader("📋 Detailed Forecast")

        # Format the prediction data for display
        display_df = predictions[["Date", "Predicted_Price"]].copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df.columns = ["Date 📅", "Predicted Price 💰"]

        # Show the table
        st.dataframe(display_df, use_container_width=True)

    else:
        st.error("Failed to generate predictions. Please try again.")


# Add footer
st.markdown("---")
st.markdown("### 📈 Nigerian Stock Market Price Prediction App")
st.markdown(
    "Built with Streamlit and ML models trained on historical Nigerian stock data."
)
