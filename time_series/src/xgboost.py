from src.common.base_component import BaseComponent


class XGBoostTs(BaseComponent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("XGBoost object initialized.")

    def get_xgboost_metrics(self):
        self.logger.info("XGBoost metrics started...")

    def get_xgboost_model(self):
        self.logger.info("XGBoost get_xgboost_model started...")

    def optuna_optimizer(self):
        self.logger.info("XGBoost optimize started...")

    def get_results(self):
        self.logger.info("XGBoost get_results started...")
