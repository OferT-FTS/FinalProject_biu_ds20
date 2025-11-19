from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
from pyprojroot import here

class AppConfig(BaseModel):
    project_root: Path = Field(default_factory=lambda: here())
    ts_data_dir: Path = Field(default_factory=lambda: here("time_series/"))
    ts_data_file: Path = Field(default_factory=lambda: here("time_series/data/raw/sale-ecomm.xls"))

    plots_dir: Path = Field(default_factory=lambda: here() / "time_series/plots")

    log_dir: Path = Field(default_factory=lambda: here("time_series/logs"))
    log_file: Path = Field(default_factory=lambda: here("time_series/logs/app.log"))
    log_level: str = "DEBUG"

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
