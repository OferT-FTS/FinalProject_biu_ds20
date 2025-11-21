from typing import Tuple
import pandas as pd
from src.common.base_component import BaseComponent
from src.data_load import DataLoad
from time_series.meta_prophet import MetaProphet
from time_series.sarimax import SArimaX


class MLPipeline(BaseComponent):
    def __init__(self, config) -> None:
        super().__init__(config)

    def run(self) -> None:
        self.logger.info("Running machine learning pipeline...")

        data_loader = DataLoad(self.config.ts_data_file, self.config)
        df: pd.DataFrame = data_loader.import_data()
        # Use only 'Date' and 'Sales (billion USD)' columns
        df = df[['Date', 'Sales (billion USD)']]

        # Rename the columns
        df = df.rename(columns={'Date': 'ds', 'Sales (billion USD)': 'y'})
        print(df.info())
        # Convert the column `ds` to datetime, specifying the date format
        df['ds'] = pd.to_datetime(df['ds'], format='%d-%m-%Y')
        meta_p = MetaProphet(self.config)
        model, forecast = meta_p.fit_model_lin(
            df=df,
            period=12,
            frq='M',
            ci=0.95,
            not_neg=False
        )
        print(forecast.info())
        forecast_filtered = forecast[forecast['ds'].isin(df['ds'])]
        mse, r2 = meta_p.get_prophet_metrics(
            forecast,  # the full forecast DataFrame
            df  # the original df with ds + y
        )
        self.logger.info(f"mse: {mse}, r2: {r2}")
        self.logger.info("Pipeline completed successfully.")

        model, forecast = meta_p.fit_model_log(
                df=df,
                period=12,
                frq='M',
                ci=0.95,
                cap=None,
                floor=None
        )

        meta_p.show_plots(
                model=model,
                forecast=forecast,
                df=df
        )

        # SArimaX Modelling
        #arima model
        sarimax = SArimaX(self.config)
        sarimax.run_model()
        # plots, metrics etc.
        sarimax.model_results()