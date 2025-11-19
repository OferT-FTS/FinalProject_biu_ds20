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

class SArimaX(BaseComponent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logger.info("Object Arima created successfully...")

    def get_metrics(self):
        self.logger.info("get_metrics started...")

    def split_data(self, df)->tuple[pd.DataFrame,pd.DataFrame]:
        self.logger.info("split_data started...")
        X_train, X_test =
        return X_train, X_test

    def show_plots(self):
        self.logger.info("show_plots started...")

    def model_results(self):
        self.logger.info("model_results started...")
        self.show_plots()
        self.get_metrics()

    def arima_model(self):
        self.logger.info("arima_model started...")

    def sarima_model(self):
        self.logger.info("sarima_model started...")

    def sarimax_model(self):
        self.logger.info("sarimax_model started...")

    def save_model(self):
        self.logger.info("save model started...")

    def run_model(self, model_type):
        self.logger.info("run_model started...")
        if model_type=="arima":
            return self.arima_model(  )

        elif model_type=="sarima":
            return self.sarima_model()

        elif model_type=="sarimax":
           return self.sarimax_model()
        else:
            self.logger.infile(f"No Valid Model Type Given: {model_type}")
            return None, None
