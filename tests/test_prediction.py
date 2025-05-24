"""Test module for the PredictionService."""

import pytest

from language_classifier.services.data_preparation_service import DataPreparationService
from language_classifier.services.prediction_service import PredictionService


@pytest.mark.parametrize(
    ("phrase", "expected_language"),
    [
        ("Ciao, come va?", "Italian"),
        ("Hello, how are you?", "English"),
    ],
)
def test_predict_phrase_language_real_model(
    phrase: str,
    expected_language: str,
) -> None:
    """
    Test that PredictionService correctly classifies the language of a phrase using the real DataPreparationService and model.

    Args:
        phrase (str): The input phrase to classify.
        expected_language (str): The expected language label ("Italian" or "English").

    """
    prediction_service = PredictionService()
    data_preparation_service = DataPreparationService()

    prediction, language = prediction_service.predict_phrase_language(
        phrase,
        data_preparation_service,
    )

    assert language == expected_language
    assert isinstance(prediction, int)
