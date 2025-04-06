from fastapi import FastAPI
from pydantic import BaseModel
from model import LinearRegressionModel


class FitIn(BaseModel):
    ticker: str


class FitOut(FitIn):
    success: bool
    message: str


class PredictIn(BaseModel):
    ticker: str
    n_days: int


class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


def build_model(ticker: str):
    model = LinearRegressionModel(ticker)
    return model


app = FastAPI()


@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request: FitIn):
    """Fit linear regression model, return confirmation message.

    Parameters
    ----------
    request : FitIn
        Must contain ticker and other parameters

    Returns
    ------
    dict
        Must conform to `FitOut` class
    """
    # Create `response` dictionary from `request`
    response = request.model_dump()

    # Create try block to handle exceptions
    try:
        # Initialize the model with the ticker
        model = LinearRegressionModel(ticker=request.ticker)
        model.fit()

        # Add `"success"` key to `response`
        response["success"] = True

        # Add `"message"` key to `response` with model path and used features
        response["message"] = (
            f"Model for {request.ticker} trained successfully with "
            f"features {model.features} to predict {model.target}. "
            f"Model saved at {model.model_path}"
        )

    # Create except block
    except Exception as e:
        # Add `"success"` key to `response`
        response["success"] = False

        # Add `"message"` key to `response` with error message
        response["message"] = f"Error fitting model: {str(e)}"

    # Return response
    return response
