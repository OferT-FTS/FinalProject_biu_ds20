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

import src.pr_0_defs

class FrMLPipeline(BaseComponent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.logger.info("Initializing Fraud ML pipeline...")

    def run_fraud_pipeline(self) -> None:
        handle_data_load
