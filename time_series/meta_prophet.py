# src/time_series/meta_prophet.py
from typing import Tuple
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import logging
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.base_component import BaseComponent

class MetaProphet(BaseComponent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.logger.info("MetaProphet initialized")

    def get_prophet_metrics(
        self, forecast: pd.DataFrame, filtered: pd.DataFrame
    ) -> Tuple[float, float]:
        self.logger.info("Calculating Prophet metrics...")
        mse = mean_squared_error(forecast["y"], filtered["yhat"])
        r2 = r2_score(forecast["y"], filtered["yhat"])
        return mse, r2

    def fit_model_lin(
        self,
        df: pd.DataFrame,
        period: int,
        frq: str | None = None,
        ci: float = 1.0,
        not_neg: bool = False,
    ) -> Tuple[Prophet, pd.DataFrame]:

        self.logger.info("Fitting Prophet model (linear growth)...")
        model = Prophet(interval_width=ci)
        model.fit(df)

        future = model.make_future_dataframe(periods=period, freq=frq) \
            if frq else model.make_future_dataframe(periods=period)

        forecast = model.predict(future)

        if not_neg:
            forecast["yhat"] = forecast["yhat"].clip(lower=0)
            forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
            forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

        return model, forecast
