The goal of the python project is to capture as much as possible of the study material.

The project consists of python programs relevant for credit card fraud detection, time series, LSTM and more models.

Two configs for all models:
- config/config.py settings for all models
- logging_config.py for logging

Credit Card Fraud Model:
- main.py
  - config = AppConfig()  # pydantic config
    setup_logging(config.fr_log_file, config.fr_log_level) # fraud log file and log level definitions 
    fr_pipeline.run_fraud_pipeline() # run fraud pipeline
  - The intermediate run and final results can be viewed:
    - in the log file fraud_detection/logs/app.log
    - reports in fraud_detection/reports/
    - plots in fraud_detection/plots
    - data pickle and csv intermediate results fraud_detection/data/processed/ 
    - final results saved in fraud_detection/reports/
  

Prophet Time series model: 
- main.py
  config = AppConfig()  # pydantic config
    setup_logging(config.ts_log_file, config.ts_log_level)
    ts_pipeline = TsMLPipeline(config)
    ts_pipeline.run_prophet()
  
- All models have the same folders structure

