"""This module contains the ModelLoaderService class used to load the model."""

import logging
import pickle

from language_classifier.const import PATH_MODEL, PATH_VECTORIZER

logger = logging.getLogger("model_loader_service")


class ModelLoaderService:
    """Class used to load the model."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoaderService, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializes the class."""
        self.logger = logging.getLogger("model_loader_service")
        if not hasattr(self, "model"):
            self.model = self._load_model()
        if not hasattr(self, "vectorizer"):
            self.vectorizer = self._load_vectorizer()

    def _load_model(self):
        with PATH_MODEL.open("rb") as f:
            model = pickle.load(f)
        self.logger.info("Model loaded from the disk")
        return model

    def _load_vectorizer(self):
        with PATH_VECTORIZER.open("rb") as f:
            vectorizer = pickle.load(f)
        self.logger.info("Vectorizer loaded from the disk")
        return vectorizer
