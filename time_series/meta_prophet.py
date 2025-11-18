# src/time_series/meta_prophet.py
from typing import Tuple, Optional
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.base_component import BaseComponent
from pathlib import Path


class MetaProphet(BaseComponent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.logger.info("MetaProphet initialized")
        self.output_dir: Path = config.plots_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_prophet_metrics(self, forecast: pd.DataFrame, df: pd.DataFrame) -> tuple[float, float]:
        self.logger.info("get_prophet_metrics started...")
        truth = df["y"]
        predicts = forecast.loc[df.index, "yhat"]  # align indices
        mse = mean_squared_error(truth, predicts)
        r2 = r2_score(truth, predicts)
        return mse, r2

    def fit_model_lin(
        self,
        df: pd.DataFrame,
        period: int,
        frq: str | None = None,
        ci: float = 1.0,
        not_neg: bool = False
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

    # ------------------------------------------------------------------
    # FIT MODEL (LOGISTIC)
    # ------------------------------------------------------------------
    def fit_model_log(
            self,
            df: pd.DataFrame,
            period: float,
            frq: str,
            ci: float = 1.0,
            cap: Optional[pd.Series] = None,
            floor: Optional[pd.Series] = None
    ) -> Tuple[Prophet, pd.DataFrame]:

        self.logger.info("fit_model_log started...")

        # Add logistic bounds
        if cap is not None:
            df["cap"] = cap
        if floor is not None:
            df["floor"] = floor

        if cap is None and floor is None:
            st_dev = df["y"].std()
            cap_val = df["y"].max() + (2 * st_dev)
            floor_val = df["y"].min() - (2 * st_dev)
            df["cap"] = cap_val
            df["floor"] = floor_val

        model = Prophet(growth="logistic", interval_width=ci)
        model.fit(df)

        future = model.make_future_dataframe(periods=period, freq=frq)
        future["cap"] = df["cap"].iloc[-1]
        future["floor"] = df["floor"].iloc[-1]

        forecast = model.predict(future)
        return model, forecast

    # ------------------------------------------------------------------
    # SAVE ALL PLOTS
    # ------------------------------------------------------------------
    def show_plots(
            self,
            model: Prophet,
            forecast: pd.DataFrame,
            df: pd.DataFrame
    ) -> None:

        self.logger.info("show_plots started...")

        # ------------------------------------------------------------------
        # 1. Prophet Forecast Plot
        # ------------------------------------------------------------------
        fig1 = model.plot(forecast)
        fig1.savefig(self.output_dir / "prophet_forecast.png", dpi=200)
        plt.close(fig1)

        # ------------------------------------------------------------------
        # 2. Prophet Components Plot
        # ------------------------------------------------------------------
        fig2 = model.plot_components(forecast)
        fig2.savefig(self.output_dir / "prophet_components.png", dpi=200)
        plt.close(fig2)

        # ------------------------------------------------------------------
        # 3. Altair Chart (saved to HTML)
        # ------------------------------------------------------------------
        df_melt = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].melt(
            "ds", var_name="Metric", value_name="Sales"
        )

        chart = (
            alt.Chart(df_melt)
            .mark_line(point=True)
            .encode(
                x=alt.X("ds:T", title="Date"),
                y=alt.Y("Sales", title="Sales (billion USD)"),
                color="Metric",
                tooltip=["ds", "Sales", "Metric"],
            )
            .properties(title="Sales Forecast")
        )
        chart.save(str(self.output_dir / "altair_forecast.html"))

        # ------------------------------------------------------------------
        # 4. Forecast Future Only
        # ------------------------------------------------------------------
        forecast_only = forecast[forecast["ds"] >= df["ds"].max()]

        plt.figure(figsize=(20, 15))
        plt.plot(forecast_only["ds"], forecast_only["yhat"], label="Forecast")
        plt.fill_between(
            forecast_only["ds"],
            forecast_only["yhat_lower"],
            forecast_only["yhat_upper"],
            alpha=0.2,
            label="Confidence Interval",
        )
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Forecast (Future Only)")
        plt.legend()
        plt.grid(False)
        plt.savefig(self.output_dir / "forecast_only.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------------
        # 5. Histogram + KDE
        # ------------------------------------------------------------------
        returns = df["y"].pct_change().dropna() * 100

        plt.figure(figsize=(10, 5))
        sns.histplot(returns, kde=True, bins=50)
        plt.title("Distribution of Percentage Changes")
        plt.xlabel("Changes (%)")
        plt.ylabel("Frequency")
        plt.savefig(self.output_dir / "histogram_pct_changes.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------------
        # 6. Boxplot
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=returns.squeeze())
        plt.title("Boxplot of Percentage Changes")
        plt.xlabel("Changes (%)")
        plt.savefig(self.output_dir / "boxplot_pct_changes.png", dpi=200)
        plt.close()

        self.logger.info(f"All plots saved to: {self.output_dir}")
