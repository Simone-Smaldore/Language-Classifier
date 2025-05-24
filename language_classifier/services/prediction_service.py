"""This module contains the PredictionService class used to make the predictions."""

import logging

from language_classifier.const import ITALIAN_PRED
from language_classifier.services.data_preparation_service import DataPreparationService
from language_classifier.services.model_loader_service import ModelLoaderService


logger = logging.getLogger("prediction_service")


class PredictionService:
    """Class used to make the prediction."""

    def __init__(self) -> None:
        """Initializes the class."""
        self.logger = logging.getLogger("prediction_service")

    def predict_phrase_language(
        self, phrase: str, data_preparation_service: DataPreparationService
    ) -> tuple[int, str]:
        prepared_data = data_preparation_service.prepare_data(phrase)
        num_words_embedded = prepared_data.indptr[1] - prepared_data.indptr[0]
        self.logger.info(
            f"{num_words_embedded} unique words are embedded to predict the language"
        )
        pred = ModelLoaderService().model.predict(prepared_data)[0]
        return int(pred), "Italian" if pred == ITALIAN_PRED else "English"
