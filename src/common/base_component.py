# src/common/base_component.py
import logging
from typing import Any

class BaseComponent:
    def __init__(self, config: Any) -> None:
        """
        Base class providing:
        - shared config
        - shared project logger
        """
        self.config = config
        self.logger = logging.getLogger("ml_project")
