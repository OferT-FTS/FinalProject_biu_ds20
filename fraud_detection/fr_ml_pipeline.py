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
from pathlib import Path
from fraud_detection.src.fraud_detection_phases  import ModelFraud
import fraud_detection.src.pr_0_defs

class FrMLPipeline(BaseComponent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.logger.info("Initializing Fraud ML pipeline...")

    def run_fraud_pipeline(self) -> None:
        loader_obj = DataUnLoad(self.config)
        fraud_model=ModelFraud(self.config, loader_obj)
        with self.timer("handle_data_load"):
            fraud_model.handle_data_load()
        with self.timer("data_preparation"):
            fraud_model.data_preparation()
        with self.timer("tests_data_preparation"):
            fraud_model.tests_data_preparation()
        # with self.timer("eda_tests"):
        #     fraud_model.eda_tests()
        with self.timer("feature_engineering"):
            fraud_model.feature_engineering()
        with self.timer("roll_stats_selection_models_fit"):
            fraud_model.roll_stats_selection_models_fit()

