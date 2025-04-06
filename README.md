# Nigerian Stock Market Price Prediction

This project predicts stock prices for companies on the Nigerian Stock Exchange using machine learning. Users can select a stock ticker and see price predictions for future days through an easy-to-use web app.

use

## Features

- 📈 Predict future stock prices based on historical data
- 🏢 Support for multiple Nigerian companies
- 📊 Interactive charts showing historical prices and predictions
- 📅 Forecasts starting from today for any number of days ahead
- 💹 Detailed prediction tables with dates and prices

## How It Works

### 1. Data Collection and Cleaning

We collect Nigerian stock market data with these fields:
- Date
- Open price
- High price
- Low price  
- Close price
- Volume
- Percentage change

The data is cleaned by:
- Removing unnecessary columns
- Converting text to proper numbers
- Handling missing values
- Converting volume notations (M for million, K for thousand)

### 2. Training Models

For each company (ticker):
1. We split the data into training (80%) and testing (20%) sets
2. We train a Linear Regression model using these features:
   - Open price
   - High price
   - Low price
   - Trading volume
   - Percentage change
3. The model learns to predict the closing price
4. We save each trained model as a file

### 3. Web Application

Our Streamlit app allows users to:
1. Select a company from a dropdown menu
2. Choose how many days to predict (1-30)
3. Click "Predict" to see future price estimates
4. View results as both a chart and a table

## How to Run the Project

### Requirements

- Python 3.7 or higher
- Required packages listed in `requirements.txt`

### Setup

1. Clone this repository
```
git clone <repository-url>
```

2. Install required packages
```
pip install -r requirements.txt
```

3. Run the Streamlit app
```
streamlit run app.py
```

## Project Structure

```
Nigerian Stock Market Price Prediction/
├── data/
│   └── Nigerian_stock_market.csv    # Stock market data
├── models/                          # Trained models
│   ├── AIICO_model.pkl
│   ├── DANGCEM_model.pkl
│   └── ...
├── app.py                           # Streamlit web application
|___ Notebooks                       #Notebook Folder
└── requirements.txt                 # Project dependencies
```

## Usage Example

1. Open the app in your web browser
2. Select a stock ticker (e.g., "DANGCEM")  
3. Use the slider to select how many days to forecast (e.g., 7 days)
4. Click "Predict Prices" button
5. View the historical data and predictions chart
6. Check the detailed forecast table below the chart

## Future Improvements

- Add more advanced prediction models
- Include more Nigerian stocks
- Add confidence intervals to predictions
- Provide trading signals (buy/sell/hold recommendations)
- Automatic data updates

## Credits

- Stock data sourced from Nigerian Stock Exchange
- Built with Python, Streamlit, and scikit-learn