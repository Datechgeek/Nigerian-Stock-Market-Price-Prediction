# Nigerian Stock Market Price Prediction

📈 Nigerian Stock Market Predictor
An end-to-end Machine Learning pipeline that predicts stock prices for 10 major blue-chip companies listed on the Nigerian Stock Exchange (NSE). This project handles multi-sector financial data (Banking, FMCG, Telco, etc.) and serves the model via an interactive web interface.

🔴 Live Demo: Click here to test the application

🏗️ Engineering Architecture
This project implements a full inference pipeline, decoupling the training logic from the user-facing application. (This diagram will render automatically on GitHub)

Code snippet

graph LR
    A[User Input] -->|Select NSE Ticker & Date| B(Streamlit Interface)
    B -->|Request Prediction| C{Inference Engine}
    D[Historical Market Data] -->|Training| E[Scikit-Learn Model]
    E -->|Pickle Serialization| F[Model Registry .pkl]
    F -->|Load Model| C
    C -->|Return Price| B
🛠️ Tech Stack & Tools
Machine Learning & Data:

Deployment & App:

🚀 Key Features (Engineering Focus)
End-to-End Pipeline: Successfully integrated data preprocessing, model inference, and frontend visualization into a single deployable unit.

Multi-Sector Capability: The model is trained to handle volatility across different industries on the NSE, not just banking sectors.

Model Persistence: Utilized joblib/pickle for efficient model serialization, allowing for low-latency inference without retraining on every request.

Scalable Architecture: Designed the codebase with modular functions, making it easy to add new tickers (e.g., MTN, Dangote Cement) without breaking the UI logic.

💻 Local Installation
To run this inference pipeline on your local machine:

Clone the repository

Bash

git clone https://github.com/Datechgeek/Nigerian-Stock-Market-Price-Prediction.git
cd Nigerian-Stock-Market-Price-Prediction
Install dependencies

Bash

pip install -r requirements.txt
Run the application

Bash

streamlit run app.py
📊 Model Performance
 Best Performing Algorithm: [Linear Regression]

Metrics: Achieved an RMSE of [Insert Value] during validation testing.

Note: Financial market prediction is stochastic; this tool serves as a technical demonstration of ML engineering capabilities rather than financial advice.

Author: [Micah Okpara] Connecting with me: LinkedIn-https://www.linkedin.com/in/micah-okpara/ | Twitter-https://x.com/Micah_AI
