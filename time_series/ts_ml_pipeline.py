import pandas as pd
from pmdarima.arima import ndiffs
from src.common.base_component import BaseComponent
from src.common.data_un_load import DataUnLoad
from time_series.src.meta_prophet import MetaProphet
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class TsMLPipeline(BaseComponent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.logger.info("Initializing ML pipeline...")

    def run_prophet(self,frq: str='d') -> None:
        self.logger.info("Time Series Running Prophet machine learning pipeline...")

        data_loader = DataUnLoad(self.config)
        df: pd.DataFrame = data_loader.import_data(self.config.ts_data_file, ';')

        df = df[['timeClose', 'close']]

        df = df.rename(columns={'timeClose': 'ds', 'close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%dT%H:%M:%S.%fZ')
        df = df.sort_values(by='ds').reset_index(drop=True)

        meta_p = MetaProphet(self.config)
        model, forecast = meta_p.fit_model_lin(
            df=df,
            period=12,
            frq='d',
            ci=0.95,
            not_neg=False
        )
        self.logger.info("Time Series Prophet forecast data frame info():")
        self.logger.info(forecast.info())
        forecast_filtered = forecast[forecast['ds'].isin(df['ds'])]
        mse, r2 = meta_p.get_prophet_metrics(
            forecast,  # the full forecast DataFrame
            df  # the original df with ds + y
        )
        self.logger.info(f"mse: {mse}, r2: {r2}")
        self.logger.info("ML Pipeline Prophet Linear Model Completed Successfully.")

        #save plots to plots folder
        meta_p.show_plots(model=model, forecast=forecast, df=df)

        #fit logistic model
        model, forecast = meta_p.fit_model_log(
                df=df,
                period=5,
                frq='D',
                ci=0.95,
                cap=None,
                floor=None,
        )

        self.logger.info(f"mse: {mse}, r2: {r2}")
        self.logger.info("ML Pipeline Prophet Logistic Model Completed Successfully.")

        #save plots to plots folder
        meta_p.show_plots(model=model, forecast=forecast, df=df)


    def run_s_arima_x(self):
        self.logger.info("Time Series Running SARIMAX ML pipeline...")
        data_loader = DataUnLoad(self.config)
        data=data_loader.download_ticker("IBM", start=datetime(2020, 1, 1), end=datetime(2020, 12, 31))

        path = self.config.ts_data_raw_dir + '/IBM.csv'
        data_loader.write_df_to_csv(data, path)

        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(data, label='Original')
        plt.legend(loc='upper left')

        # פירוק למרכיבים
        result = seasonal_decompose(data, model='additive', period=252)  # 252 ימי מסחר בשנה plt.plot(result.trend, label='Trend')

        plt.figure(figsize=(12, 8))
        plt.subplot(412)
        plt.plot(result.trend, label='Trend')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        plt.legend(loc='upper left')
        plt.figure(figsize=(12, 8))
        plt.subplot(413)
        plt.plot(result.trend, label='Trend')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        result_adf = adfuller(data)
        print(f'ADF Statistic: {result_adf[0]}')
        print(f'p-value: {result_adf[1]}')
        print('Critical Values:')
        if result_adf[1] <= 0.05:
            print("The time series is stationary.")
        else:
            print(
                "The time series is not stationary. Consider transforming the data (e.g., differencing) before building a model.")

        plot_acf(data, lags=40)
        plt.title("ACF - Orginal data (No diff)")
        plt.show()

        plt.figure(figsize=(10, 4))

        diffs = data.pct_change().dropna()  # מסיר את ה-NaN הראשון
        # אם diff הצליח להפוך את הסדרה לסטציונרית → ה-ACF ייחתך מהר
        plot_acf(diffs, lags=40)
        plt.title("ACF - D=1")
        plt.show()



        # מבחן ADF על סדרת התשואות
        result_adf = adfuller(diffs)

        print(f'ADF Statistic: {result_adf[0]}')
        print(f'p-value: {result_adf[1]}')
        print('Critical Values:')
        for key, value in result_adf[4].items():
            print(f"   {key}: {value}")

        # מסקנה
        if result_adf[1] <= 0.05:
            print("\n  מתאימה להריץ ARIMA הסדרה סטציונרית.")
        else:
            print("\n הסדרה לא סטציונרית – כדאי לבצע טרנספורמציה נוספת.")

        plot_pacf(diffs.dropna(), lags=20)
        plt.title('PACF Plot')
        plt.show()

        plot_acf(diffs.dropna(), lags=20)
        plt.title('ACF Plot')
        plt.show()

        d = ndiffs(diffs, test='adf')  # Determine the value of d using the ADF

        print(f'd = {d}')




    def run_xgboost(self):
        self.logger.info("Time Series Running XGBoost machine learning pipeline...")