import os


import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    def __init__(self, ticker):
        """
        Initialize a LinearRegressionModel for stock price prediction.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock whose price will be predicted
        """
        self.ticker = ticker
        self.model = None
        self.file_path = os.path.join("data", "Nigerian_stock_market.csv")
        self.model_path = os.path.join("models", f"{ticker}_model.pkl")
        self.features = ["Open", "High", "Low", "Volume(Million)"]
        self.target = "Price"
        self.metadata = {}

    def wrangle(self, filename=None):
        """
        Process and clean stock market data for the specified ticker.

        Args:
            filename (str, optional): Path to the CSV file containing stock data.
                                    If None, uses the default file_path.

        Returns:
            pandas.DataFrame: Cleaned and processed DataFrame for the ticker
        """
        # Use provided filename or default to self.file_path
        if filename is None:
            filename = self.file_path

        # Load the data from the file into a DataFrame
        df = pd.read_csv(filename)

        # Filter data for the specified ticker
        if "Ticker" in df.columns:
            df = df[df["Ticker"] == self.ticker]
            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")

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

        # Convert "[Price, Open, High, Low]" to float
        convert = ["Price", "Open", "High", "Low"]
        for c in convert:
            if c in df.columns:
                df[c] = (
                    df[c]
                    .str.replace(",", "", regex=False)
                    .astype(float, errors="ignore")
                )

        # Convert "Change(%)" to float
        if "Change(%)" in df.columns:
            df["Change(%)"] = (
                df["Change(%)"]
                .str.replace("%", "", regex=False)
                .astype(float, errors="ignore")
            )

        # Convert "Volume(Million)" to float with unit handling
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
                        except ValueError:
                            return None
                return value

            df["Volume(Million)"] = (
                df["Volume(Million)"]
                .apply(convert_volume)
                .astype(float, errors="ignore")
            )

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Set 'Date' as the index
        df.set_index("Date", inplace=True)

        return df

    def fit(self, X=None, y=None, train_size=0.8):
        """
        Train the linear regression model with the provided data.
        If X and y are not provided, loads and processes data for the ticker.

        Args:
            X: Features for training, if None loads from file
            y: Target variable for training, if None loads from file
            train_size: Fraction of data to use for training (0-1)

        Returns:
            self: The trained model instance
        """
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # If X and y are not provided, load and prepare data
        if X is None or y is None:
            df = self.wrangle()

            # Use default features and target if available in dataframe
            available_features = [f for f in self.features if f in df.columns]
            if not available_features:
                raise ValueError(
                    f"None of the default features {self.features} found in data"
                )

            if self.target not in df.columns:
                raise ValueError(f"Target column {self.target} not found in data")

            # Update features to only use available ones
            self.features = available_features

            # Extract features and target
            X = df[self.features].values
            y = df[self.target].values

        # Create and train the model
        self.model = LinearRegression()
        self.model.fit(X, y)

        # Save metadata
        self.metadata = {
            "ticker": self.ticker,
            "features": self.features,
            "target": self.target,
            "train_size": train_size,
            "data_shape": X.shape,
        }

        # Save the trained model
        self.save_model(self.model)

        return self

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        return self.model.predict(X)

    def save_model(self, model):
        joblib.dump(model, self.model_path)

    def save_metadata(self, metadata):
        """Save model metadata alongside the model"""
        self.metadata.update(metadata)
        metadata_path = self.model_path.replace(".pkl", "_metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def __clean_price_prediction(self, prediction, dates):
        """Reformat model price prediction to JSON.

        Parameters
        ----------
        prediction : np.ndarray
            Array of price predictions
        dates : list or pd.DatetimeIndex
            Dates corresponding to the predictions

        Returns
        -------
        dict
            Price forecast. Each key is date in ISO 8601 format.
            Each value is the predicted price.
        """
        # Ensure dates are in datetime format if they aren't already
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        # Create prediction index labels, ISO 8601 format
        prediction_index = [d.isoformat() for d in dates]

        # Extract predictions and flatten if needed
        data = prediction if prediction.ndim == 1 else prediction.flatten()

        # Combine data and prediction_index into Series
        prediction_formatted = pd.Series(data, index=prediction_index)

        # Return Series as dictionary
        return prediction_formatted.to_dict()

    def predict_price(self, X, forecast_dates):
        """Predict prices using the trained model

        Parameters
        ----------
        X : array-like
            Features for prediction
        forecast_dates : list or pd.DatetimeIndex
            Dates corresponding to the predictions

        Returns
        -------
        dict
            Price forecast. Each key is date in ISO 8601 format.
            Each value is the predicted price.
        """
        # Use the existing predict method to get raw predictions
        predictions = self.predict(X)

        # Format prediction with __clean_price_prediction
        prediction_formatted = self.__clean_price_prediction(
            predictions, forecast_dates
        )

        # Return formatted predictions
        return prediction_formatted
