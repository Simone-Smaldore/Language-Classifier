"""This module contains the PredictionService class used to make the predictions."""

import logging


logger = logging.getLogger("prediction_service")


class PredictionService:
    """Class used to make the prediction."""

    def __init__(self) -> None:
        """Initializes the class."""
        self.logger = logging.getLogger("prediction_service")
