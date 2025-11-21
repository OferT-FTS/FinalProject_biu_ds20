""" SARIMAX = Seasonal ARIMA with eXogenous regressors.
It models a time series as AR (autoregressive) + I (integrated = differencing) + MA (moving average),
with optional seasonal ARIMA terms and optional exogenous (external) predictors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
import matplotlib.colors as mcolors
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from src.common.base_component import BaseComponent
from datetime import datetime
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


class SArimaX(BaseComponent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logger.info("Object Arima created successfully...")

    def get_metrics(self):
        self.logger.info("get_metrics started...")

    def split_data(self, df)->tuple[pd.DataFrame,pd.DataFrame]:
        self.logger.info("split_data started...")
        # X_train, X_test =
        # return X_train, X_test

    def show_plots(self):
        self.logger.info("show_plots started...")

    def model_results(self):
        self.logger.info("model_results started...")
        self.show_plots()
        self.get_metrics()

    def save_model(self):
        self.logger.info("save model started...")

    def run_model(self):
        self.logger.info("run_model started...")
        #prepare data
        #auto_arima to get the best parameters
        #sarima with p,q and i from auto_arima, decide on seasonal parameters P,Q,I and S
            # to optimize on AIC and also print RMSE
        #now forecast per day and every new Y_hat insert back into model and fit again
    def arima_with_xgboost(self):
        self.logger.info("arima_with_xgboost started...")
        # next lesson ?!?
