"""Contain the PredictionService class used to make the predictions."""

import logging

from language_classifier.const import ITALIAN_PRED
from language_classifier.services.data_preparation_service import DataPreparationService
from language_classifier.services.model_loader_service import ModelLoaderService

logger = logging.getLogger("prediction_service")


class PredictionService:
    """Class used to make the prediction."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.logger = logging.getLogger("prediction_service")

    def predict_phrase_language(
        self,
        phrase: str,
        data_preparation_service: DataPreparationService,
    ) -> tuple[int, str]:
        """
        Predict the language of a given phrase.

        This method prepares the input phrase using the provided data preparation service,
        performs the language prediction using a preloaded model, and returns both the
        numerical prediction and its corresponding language label.

        Args:
            phrase (str): The input text phrase to classify.
            data_preparation_service (DataPreparationService): Service to preprocess the phrase.

        Returns:
            tuple[int, str]: A tuple containing the predicted language label as an integer
            and the corresponding language name ("Italian" or "English").

        """
        prepared_data = data_preparation_service.prepare_data(phrase)
        num_words_embedded = prepared_data.indptr[1] - prepared_data.indptr[0]
        self.logger.info(
            "%d unique words are embedded to predict the language",
            num_words_embedded,
        )
        pred = ModelLoaderService().model.predict(prepared_data)[0]
        return int(pred), "Italian" if pred == ITALIAN_PRED else "English"
