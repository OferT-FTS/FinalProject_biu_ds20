import pandas as pd
from pathlib import Path
import pickle
from typing import Optional
from src.common.base_component import BaseComponent
import os
from datetime import datetime
import yfinance as yf

class DataUnLoad(BaseComponent):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.infile: Path = infile
        self.logger.info(f"DataLoad initialized...")

    def valid_file(self, infile) -> bool:
        """Check whether file exists and is a file."""
        exists = infile.exists() and infile.is_file()
        if exists:
            self.logger.info(f"File exists: {infile}")
        else:
            self.logger.error(f"File not found: {infile}")
        return exists

    def import_data(self, infile, seperator: str=None) -> pd.DataFrame:
        """Load CSV, Excel, TXT, or pickle safely."""
        if not self.valid_file(infile):
            raise FileNotFoundError(f"File not found: {infile}")

        ext = infile.suffix.lower()
        self.logger.info(f"Detected file extension: {ext}")

        try:
            if ext == ".csv":
                if seperator is None:
                    return pd.read_csv(infile)
                else:
                    return pd.read_csv(infile, sep=seperator)
            if ext in (".xls", ".xlsx"):
                return pd.read_excel(infile)
            if ext == ".txt":
                return pd.read_csv(infile, sep=None, engine="python")
            if ext == ".pkl":
                return pd.read_pickle(infile)

        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            raise

        raise ValueError(f"Unsupported extension: {ext}")

    def write_df_to_csv(self, df: pd.DataFrame, infile: str) -> None:
        self.logger.info(f"Writing DataFrame to {infile} started...")
        path = Path(infile)
        if os.path.exists(infile) and os.path.isfile(infile):
            os.remove(infile)
            self.logger.info(f"The file '{infile}' has been deleted.")
        else:
            self.logger.info(f"The file '{infile}' does not exist, so not deleted.")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            self.logger.info(f"DataFrame saved to {path}")
        except Exception as e:
            self.logger.info(f"Error saving DataFrame to CSV: {e}")


    def write_df_to_pickle(self, df :pd.DataFrame, infile :str) -> None:
        self.logger.info(f"Writing DataFrame to {infile} started...")
        path = Path(infile)
        if os.path.exists(infile):
            try:
                os.remove(infile)
                self.logger.info(f"File '{infile}' deleted successfully.")
            except OSError as e:
                self.logger.info(f"Error deleting file '{infile}': {e}")
        else:
            self.logger.info(f"File '{infile}' does not exist, so not deleted.")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_pickle(path)
            self.logger.info(f"DataFrame saved to {path}")

        except Exception as e:
            self.logger.info(f"Error saving DataFrame to pickle: {e}")


    def download_ticker(self, ticker: str, start: datetime, end: datetime)->pd.DataFrame:
        self.logger.info(f"Downloading ticker {ticker} from {start} to {end}...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)

            # 1. Check if data is empty
            if df.empty:
                raise ValueError(f"No data returned for ticker '{ticker}' "
                                 f"between {start.date()} and {end.date()}")

            # 2. Check if Close column exists
            if "Close" not in df.columns:
                raise KeyError(f"'Close' column missing for ticker '{ticker}'. "
                               f"Available columns: {list(df.columns)}")

            return df

        except Exception as e:
            raise RuntimeError(f"Failed downloading data for {ticker}: {e}") from e