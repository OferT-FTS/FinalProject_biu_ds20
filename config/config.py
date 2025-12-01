from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
from pyprojroot import here
import os

class AppConfig(BaseModel):
    # project_root: Path = Field(default_factory=lambda: here())
    # ts_data_dir: Path = Field(default_factory=lambda: here("time_series/"))
    # ts_data_raw_dir: Path = Field(default_factory=lambda: here("time_series/data/raw"))
    # ts_data_process_dir: Path = Field(default_factory=lambda: here("time_series/processed"))
    # ts_data_file: Path = Field(default_factory=lambda: here("time_series/data/raw/Bitcoin.csv"))
    #
    # plots_dir: Path = Field(default_factory=lambda: here() / "time_series/plots")
    #
    # log_dir: Path = Field(default_factory=lambda: here("time_series/logs"))
    # log_file: Path = Field(default_factory=lambda: here("time_series/logs/app.log"))
    # log_level: str = "DEBUG"

    project_root: Path = Field(default_factory=lambda: here())

    # Time series paths
    ts_data_dir: Path = Field(default_factory=lambda: here("time_series"))
    ts_data_raw_dir: Path = Field(default_factory=lambda: here("time_series/raw"))
    ts_data_process_dir: Path = Field(default_factory=lambda: here("time_series/processed"))
    ts_data_file: Path = Field(default_factory=lambda: here("time_series/raw/Bitcoin.csv"))

    # time series Plots
    ts_plots_dir: Path = Field(default_factory=lambda: here("time_series/plots"))

    # time series Logs
    ts_log_dir: Path = Field(default_factory=lambda: here("time_series/logs"))
    ts_log_file: Path = Field(default_factory=lambda: here("time_series/logs/app.log"))
    ts_log_level: str = "DEBUG"

    # fraud detection paths
    fr_data_dir: Path = Field(default_factory=lambda: here("fraud_detection"))
    fr_data_raw_dir: Path = Field(default_factory=lambda: here("fraud_detection/data/raw"))
    fr_data_process_dir: Path = Field(default_factory=lambda: here("fraud_detection/data/processed"))

    fr_reports_dir: Path = Field(default_factory=lambda: here("fraud_detection/reports"))
    fr_plots_dir: Path = Field(default_factory=lambda: here("fraud_detection/plots"))

    # fraud detection Logs
    fr_log_dir: Path = Field(default_factory=lambda: here("fraud_detection/logs"))
    fr_log_file: Path = Field(default_factory=lambda: here("fraud_detection/logs/app.log"))
    fr_log_level: str = "DEBUG"
    fr_data_file: Path = Field(default_factory=lambda: here("fraud_detection/data/raw/credit_card_fraud.csv"))


    # ML
    target: str = "y"
    train_size: float = 0.8
    dev_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42

    model_name: str = "fraud_detector_v1"
    model_dir: Path = Path("data/models")

    model_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 8,
            "class_weight": "balanced",
        }
    )

    class Config:
        frozen = True
        validate_assignment = True
        extra = "forbid"

'''This is a Pydantic model configuration class. Here's what each setting does:
frozen = True prevents the model instance from being modified after creation. Once instantiated, 
you can't change any field values—attempting to do so raises a validation error. This makes instances immutable.
validate_assignment = True enables validation when you assign values to fields, even after the model is created. 
This means if you try to assign an invalid value to a field, Pydantic will validate it immediately and raise an 
error if it doesn't conform to the field's type or constraints. 
extra = "forbid" causes Pydantic to reject any extra fields that aren't defined in the model. If you try to create 
an instance with undeclared fields, Pydantic raises a validation error instead of silently ignoring them.
Together, these settings create a strict, immutable model where fields are validated on assignment and no unknown 
fields are permitted. However, there's a practical note: 
frozen = True and validate_assignment = True work together—if the model is frozen, assignment validation doesn't 
matter much since you can't assign anything anyway. The frozen setting takes precedence.
'''