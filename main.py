from src.ml_pipeline import MLPipeline
from time_series.config.config import AppConfig
from time_series.config.logging_config import setup_logging

def main():
    config = AppConfig()  # pydantic config

    setup_logging(config.log_file, config.log_level)
    pipeline = MLPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
