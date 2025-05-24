"""Contain the ModelLoaderService class used to load the model."""

import logging
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from language_classifier.const import PATH_MODEL, PATH_VECTORIZER

logger = logging.getLogger("model_loader_service")


class ModelLoaderService:
    """Class used to load the model."""

    _instance = None

    def __new__(cls: type["ModelLoaderService"]) -> "ModelLoaderService":
        """
        Create or return the singleton instance of ModelLoaderService.

        This method ensures that only one instance of the class exists by
        overriding the default object creation behavior. If an instance
        already exists, it returns that instance; otherwise, it creates a new one.

        Args:
            cls (type): The class itself.

        Returns:
            ModelLoaderService: The singleton instance of the class.

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the class."""
        self.logger = logging.getLogger("model_loader_service")
        if not hasattr(self, "model"):
            self.model = self._load_model()
        if not hasattr(self, "vectorizer"):
            self.vectorizer = self._load_vectorizer()

    def _load_model(self) -> MultinomialNB:
        with PATH_MODEL.open("rb") as f:
            model = pickle.load(f)
        self.logger.info("Model loaded from the disk")
        return model

    def _load_vectorizer(self) -> CountVectorizer:
        with PATH_VECTORIZER.open("rb") as f:
            vectorizer = pickle.load(f)
        self.logger.info("Vectorizer loaded from the disk")
        return vectorizer
