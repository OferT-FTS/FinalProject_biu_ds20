from time_series.ts_ml_pipeline import TsMLPipeline
from fraud_detection.fr_ml_pipeline import FrMLPipeline
from config.config import AppConfig
from config.logging_config import setup_logging

def main():
    config = AppConfig()  # pydantic config

    #time series logging
    # setup_logging(config.ts_log_file, config.ts_log_level)

    #fraud detection logging
    setup_logging(config.fr_log_file, config.fr_log_level)

    #ts_pipeline = TsMLPipeline(config)
    fr_pipeline = FrMLPipeline(config)
    #ts_pipeline.run_prophet()
    #fr_pipeline.run_fraud_pipeline(config)
    # ts_pipeline.run_s_arima_x()
    # ts_pipeline.run_xgboost()

if __name__ == "__main__":
    main()
