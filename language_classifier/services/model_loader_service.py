"""This module contains the ModelLoaderService class used to load the model."""

import logging
import pickle

from language_classifier.const import PATH_MODEL

logger = logging.getLogger("model_loader_service")


class ModelLoaderService:
    """Class used to load the model."""

    def __init__(self) -> None:
        """Initializes the class."""
        self.logger = logging.getLogger("model_loader_service")

    def load_model(self) -> None:
        global model
        with PATH_MODEL.open("rb") as f:
            model = pickle.load(f)
        self.logger.info("Model loaded from the disk")
