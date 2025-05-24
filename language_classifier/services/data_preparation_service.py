"""This module contains the DataPreparationService to prepare data for predictions."""

import logging


logger = logging.getLogger("data_preparation_service")


class DataPreparationService:
    """Class used to prepare the data for the predictions."""

    def __init__(self) -> None:
        """Initializes the class."""
        self.logger = logging.getLogger("data_preparation_service")
