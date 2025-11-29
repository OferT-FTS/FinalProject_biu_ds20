""" SARIMAX = Seasonal ARIMA with eXogenous regressors.
It models a time series as AR (autoregressive) + I (integrated = differencing) + MA (moving average),
with optional seasonal ARIMA terms and optional exogenous (external) predictors.
"""
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pmdarima as pm
import matplotlib.colors as mcolors
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from src.common.base_component import BaseComponent
from datetime import datetime
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore") # נטרול אזהרות
from time_series_forecasting_with_machine_learning_yt import X_train


class SArimaXTs(BaseComponent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_dir: Path = config.fr_plots_dir
        self.logger.info("Object SArimaX created successfully...")

    def get_metrics(self):
        self.logger.info("SArimaX get_metrics started...")


    def split_data(self, df: pd.DataFrame=None, split_ratio: float = 0.9)->tuple[pd.DataFrame,pd.DataFrame]:
        self.logger.info("SArimaX split_data started...")
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        if df is not None and not df.empty:
            split_index = int(len(df) * split_ratio)
            train_data = df.iloc[:split_index]
            test_data = df.iloc[split_index:]
            self.logger.info(f"(train)len: {len(train_data)}")
            self.logger.info(f"len(test) : {len(test_data )}")
        else:
            self.logger.info("SArimaX data to split is empty")
        return train_data, test_data


    def show_plots(self, y_test, predictions, forecast):
        self.logger.info("SArimaX show_plots started...")
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label='Actual', color='blue')
        plt.plot(y_test.index, predictions, label='Forecast', color='red')

        confidence_intervals = forecast.conf_int()

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        # רווח סמך – טווח אי-ודאות
        plt.fill_between(y_test.index,
                         confidence_intervals.iloc[:, 0],
                         confidence_intervals.iloc[:, 1],
                         color='red', alpha=0.2, label='Confidence Interval')

        plt.title('Forecast vs. Actual with Confidence Intervals')
        plt.xlabel('Date')
        plt.ylabel('Madad')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def model_results(self):
        self.logger.info("SArimaX model_results started...")
        self.show_plots()
        self.get_metrics()

    def save_model(self):
        self.logger.info("SArimaX save model started...")

    def run_model(self):
        self.logger.info("SArimaX run_model started...")
        #prepare data
        #auto_arima to get the best parameters
        #sarima with p,q and i from auto_arima, decide on seasonal parameters P,Q,I and S
            # to optimize on AIC and also print RMSE
        #now forecast per day and every new Y_hat insert back into model and fit again

    def fit_model(self, train_data, test_data, s, exog=None, parameters_method: str='auto_arima', sp: int=12):
        self.logger.info("SArimaX fit_model started...")
        history = train_data.values.tolist()  # ← כאן השינוי!
        predictions = []

        if parameters_method == 'auto_arima':
            aa_model = auto_arima(train_data,sp)
            ordr=aa_model.order
            ordr_s = aa_model.seasonal_order
        else:
            aa_model = self.get_loop_opt_params(train_data, sp, exog)
            ordr = aa_model.order
            ordr_s = aa_model.seasonal_order

        for t in range(len(test_data)):
            try:
                model = SARIMAX(train_data,
                           order=ordr,
                           seasonal_order=ordr_s, # seasonal period in data True\False
                           m=sp,  # seasonal period, integer
                           stepwise=True,  # fast and smart run ? True\False
                           disp=False,
                           exog=exog,# exogenous parameters data frame
                           suppress_warnings = True,  # supress warnings True\False
                           error_action = True  # continue when non-valid models exist
                         )

                model_fit = model.fit(disp=False)
                yhat = model_fit.forecast()[0]
            except Exception as e:
                print(f"Error SArimaX model train {t}: {e}")
                yhat = train_data[-1]  # fallback

            predictions.append(yhat)

            actual_obs = float(test_data.iloc[t])
            train_data.append(actual_obs)

            if (t + 1) % 10 == 0 or t == len(test_data) - 1:
                print(f"step: {t + 1}/{len(test_data)}: predict={yhat:.2f} real value: {actual_obs:.2f}")

        results_df = pd.DataFrame({
            'Date': test_data.index,
            'Actual': test_data.values,
            'Predicted': predictions
        })
        results_df.to_csv('TA35_ARIMA_forecast.csv', index=False, encoding='utf-8-sig')

        self.logger,int("SArimaX prediction saved to csv")
        self.logger.info("prediction process completed successfully.")

        # --- הערכת ביצועים ---
        predictions_series = pd.Series(predictions, index=test_data.index)
        rmse = np.sqrt(mean_squared_error(test_data, predictions_series))
        mape = mean_absolute_percentage_error(test_data, predictions_series) * 100

        print("\n--- הערכת ביצועי המודל ---")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

    def get_auto_arima_model(self,y: pd.DataFrame, ex:pd.DataFrame=None, s: bool=True, sp: int=12):
        self.logger.info("SArimaX get_auto_arima_params started...")
        auto_arima_model = auto_arima(y,
                                      exogenous=ex, #exeogenous parameters data frame,
                                      seasonal=s,  # seasonal period in data True\False
                                      m=sp,  # seasonal period, integer
                                      stepwise=True,  # fast and smart run ? True\False
                                      suppress_warnings=True, #supress warnings True\False
                                      error_action='ignore',
                                      trace=True)  # continue when non-valid models exist

        self.logger.info(auto_arima_model.summary())
        return auto_arima_model

    def get_loop_opt_params(self,train,test, ex_train: pd.DataFrame=None, ex_test: pd.DataFrame=None, sp: int=12, exog: pd.DataFrame =None):
        self.logger.info("SArimaX get_loop_opt_params started...")
        p = q = range(0, 5)
        d= ndiffs(train,test='adf')
        self.logger.info(f"Time Series SARIMAX got I(d) from adf: {d}")
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], sp) for x in list(itertools.product(p, d, q))]

        best_rmse = float('inf')
        best_aic = float("inf")
        best_bic = float("inf")
        best_order = None
        best_seasonal_order = None
        best_fit= None

        # Iterate through all parameter combinations
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = SARIMAX(
                        train,
                        order=param,
                        seasonal_order=param_seasonal,
                        disp=False,
                        exog=exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    model_fit = model.fit(disp=False)

                    aic = model_fit.aic
                    bic = model_fit.bic
                    # 2. יצירת חיזוי על תקופת ה-Test
                    forecast_test_temp = model_fit.get_forecast(steps=len(test),
                                                                       exog=ex_test)
                    predicted_means_temp = forecast_test_temp.predicted_mean

                    current_mse = mean_squared_error(ex_test, predicted_means_temp)
                    current_rmse = np.sqrt(current_mse)

                    # 4. בדיקה אם המודל הנוכחי טוב יותר
                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_order = param
                        best_seasonal_order = param_seasonal
                        best_aic = aic
                        best_bic = bic
                        best_fit = model_fit  # שמירת המודל

# if aic < best_aic:
#     best_aic = aic # add rmse option
#     best_order = param
#     best_seasonal_order = param_seasonal
# if bic < best_bic:
#     best_bic = bic
                except Exception as e:
                    continue

        self.logger.info('Best AIC: {}, Best Order: {}, Best Seasonal Order: {}'.format(best_aic, best_order, best_seasonal_order))
        self.logger.info(best_fit.summary())
        self.logger.info(best_order)
        # Fit the model with the best parameters
        # best_model = SARIMAX(
        #                 train,
        #                 order=best_order,
        #                 seasonal_order=best_seasonal_order,
        #                 disp = False,
        #                 exog = ex_train,
        #                 enforce_stationarity = False,
        #                 enforce_invertibility = False
        #               )
        # 5. הדפסת התוצאות
        print("\n---   חיפוש הסתיים! ---")
        if best_fit is None:
            print("לא נמצא מודל תקין. נסה לשנות את טווח הפרמטרים או הנתונים.")
        else:
            print(f"המודל הטוב ביותר שנמצא: SARIMAX{best_order}{best_seasonal_order}")
            print(f"ערך RMSE הנמוך ביותר (על Test): {best_rmse:.2f}")

            sarimax_fit = best_fit
            best_model_title = f"SARIMAX{best_order}{best_seasonal_order}"

            # ----------------------------
            # 4) הצגת גרף ביצועים על סט ה-Test
            # ----------------------------
            print("שלב 4: מציג ביצועים על סט המבחן (Test)...")
            forecast_test_res = sarimax_fit.get_forecast(steps=len(test),
                                                         exog=ex_test)
            forecast_test = forecast_test_res.predicted_mean
            conf_int_test = forecast_test_res.conf_int()
            forecast_test.index = test.index
            conf_int_test.index = test.index

            plt.figure(figsize=(18, 12))
            plt.plot(train, label='Train')
            plt.plot(test, label='Test (Actual)')
            plt.plot(forecast_test, label='SARIMAX Forecast (Test)', color='green')
            plt.fill_between(forecast_test.index, conf_int_test.iloc[:, 0], conf_int_test.iloc[:, 1],
                             color='lightgreen', alpha=0.3, label='95% CI Test')

            plt.title(f"Best Model (by RMSE): {best_model_title} - Forecast Test (with Exog)")
            plt.legend()
            plt.show()

            # ================================================================
            # כאן מתחיל החלק של חיזוי 30 יום קדימה
            # ================================================================

            # ----------------------------
            # 5) אימון מחדש של המודל הטוב ביותר על כל הנתונים
            # ----------------------------
            print("\nשלב 5: מאמן מחדש את המודל הטוב ביותר על כל הנתונים (100%)...")

            final_order = best_order
            final_seasonal_order = best_seasonal_order

            endog_full = data_final['SP500']
            exog_full = data_final[['LQD', 'SP500_MA5', 'VIX']]

            warnings.filterwarnings("ignore")

            final_model = SARIMAX(endog_full,
                                  exog=exog_full,
                                  order=final_order,
                                  seasonal_order=final_seasonal_order,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)

            final_model_fit = final_model.fit(disp=False)

            warnings.filterwarnings("default")
            print(f"אימון סופי על כל הנתונים הסתיים (מודל: SARIMAX{final_order}{final_seasonal_order}).")

            # ----------------------------
            # 6) חיזוי 30 ימים קדימה (עם חיזוי המשתנים החיצוניים)
            # ----------------------------
            print("\nשלב 6: מתחיל בתהליך חיזוי 30 ימים קדימה...")
            future_steps = 30
        return best_model