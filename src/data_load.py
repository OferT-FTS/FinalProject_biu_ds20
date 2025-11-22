# src/data/data_load.py
import pandas as pd
from pathlib import Path
import pickle
from typing import Optional
from src.common.base_component import BaseComponent

class DataLoad(BaseComponent):
    def __init__(self, infile: Path, config) -> None:
        super().__init__(config)
        self.infile: Path = infile
        self.logger.info(f"DataLoad initialized with file: {self.infile}")

    def valid_file(self) -> bool:
        """Check whether file exists and is a file."""
        exists = self.infile.exists() and self.infile.is_file()
        if exists:
            self.logger.info(f"File exists: {self.infile}")
        else:
            self.logger.error(f"File not found: {self.infile}")
        return exists

    def import_data(self, seperator: str=None) -> pd.DataFrame:
        """Load CSV, Excel, TXT, or pickle safely."""
        if not self.valid_file():
            raise FileNotFoundError(f"File not found: {self.infile}")

        ext = self.infile.suffix.lower()
        self.logger.info(f"Detected file extension: {ext}")

        try:
            if ext == ".csv":
                if seperator is None:
                    return pd.read_csv(self.infile)
                else:
                    return pd.read_csv(self.infile, sep=seperator)
            if ext in (".xls", ".xlsx"):
                return pd.read_excel(self.infile)
            if ext == ".txt":
                return pd.read_csv(self.infile, sep=None, engine="python")
            if ext == ".pkl":
                return pd.read_pickle(self.infile)

        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            raise

        raise ValueError(f"Unsupported extension: {ext}")