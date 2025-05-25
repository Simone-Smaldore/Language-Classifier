"""Test module for the PredictionService."""

import pytest

from language_classifier.services.data_preparation_service import DataPreparationService
from language_classifier.services.prediction_service import PredictionService


@pytest.mark.parametrize(
    ("phrase", "expected_prediction"),
    [
        ("Ciao, come va?", 1),
        ("Hello, how are you?", 0),
    ],
)
def test_predict_phrase_language_real_model(
    phrase: str,
    expected_prediction: int,
) -> None:
    """
    Test that PredictionService correctly classifies the language of a phrase using the real DataPreparationService and model.

    Args:
        phrase (str): The input phrase to classify.
        expected_prediction (int): The expected prediction.

    """
    prediction_service = PredictionService()
    data_preparation_service = DataPreparationService()

    prediction = prediction_service.predict_phrase_language(
        phrase,
        data_preparation_service,
    )

    assert prediction == expected_prediction
    assert isinstance(prediction, int)
