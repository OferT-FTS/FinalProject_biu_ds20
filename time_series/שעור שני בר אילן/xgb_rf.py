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
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

class TsXgbRf(BaseComponent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("TsXgbRf initialized...")

    def print_results(self):
        self.logger.info("TsXgbRf results printed...")

    def fit(self):
        self.logger.info("fit started...")


    def optuna(self, X_train, X_test, y_train, y_test):
        # ============================================================
        # XGBOOST OPTUNA OPTIMIZATION
        # ============================================================

        def objective(trial):
            """
            Objective function for XGBoost hyperparameter optimization
            """
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'verbosity': 0
            }

            clf = XGBClassifier(**params)

            # Use F1-score for imbalanced fraud detection
            scores = cross_val_score(
                clf, X_train, y_train,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )

            return scores.mean()


        # ============================================================
        # RUN OPTIMIZATION
        # ============================================================

        print("=" * 100)
        print("XGBOOST HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("=" * 100)

        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)

        # ============================================================
        # RESULTS ANALYSIS
        # ============================================================

        print("\n" + "=" * 100)
        print("OPTIMIZATION RESULTS")
        print("=" * 100)
        print(f"\nBest F1-Score: {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # ============================================================
        # TRAIN FINAL MODEL WITH BEST PARAMETERS
        # ============================================================

        best_params = study.best_params
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        final_xgb = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        print("\n" + "=" * 100)
        print("TRAINING FINAL XGBOOST MODEL")
        print("=" * 100)

        # Train with early stopping on dev set
        final_xgb.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=50
        )

        print(f"\n✓ Model trained. Best iteration: {final_xgb.best_iteration}")

        # # ============================================================
        # # EVALUATION ON DEV SET
        # # ============================================================
        #
        #
        #
        # y_dev_pred = final_xgb.predict(X_dev)
        # y_dev_proba = final_xgb.predict_proba(X_dev)[:, 1]
        #
        # print("\n" + "=" * 100)
        # print("DEV SET PERFORMANCE")
        # print("=" * 100)
        #
        # tn, fp, fn, tp = confusion_matrix(y_dev, y_dev_pred).ravel()
        # print(f"\nConfusion Matrix:")
        # print(f"  TP: {tp}, FP: {fp}")
        # print(f"  FN: {fn}, TN: {tn}")
        #
        # print(f"\nClassification Report:")
        # print(classification_report(y_dev, y_dev_pred))
        #
        # print(f"\nMetrics:")
        # print(f"  Precision: {precision_score(y_dev, y_dev_pred):.4f}")
        # print(f"  Recall: {recall_score(y_dev, y_dev_pred):.4f}")
        # print(f"  F1-Score: {f1_score(y_dev, y_dev_pred):.4f}")
        # print(f"  AUC-ROC: {roc_auc_score(y_dev, y_dev_proba):.4f}")

        # ============================================================
        # EVALUATION ON TEST SET
        # ============================================================

        y_test_pred = final_xgb.predict(X_test)
        y_test_proba = final_xgb.predict_proba(X_test)[:, 1]

        print("\n" + "=" * 100)
        print("TEST SET PERFORMANCE")
        print("=" * 100)

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, FP: {fp}")
        print(f"  FN: {fn}, TN: {tn}")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred))

        print(f"\nMetrics:")
        print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
        print(f"  Recall: {recall_score(y_test, y_test_pred):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_test_pred):.4f}")
        print(f"  AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}")