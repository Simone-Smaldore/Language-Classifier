"""
Define the main prediction route.

This route is used to predict whether a given phrase is in Italian or English.
"""

from flask import Blueprint, Response, jsonify, request
from injector import inject

from language_classifier.services.data_preparation_service import DataPreparationService
from language_classifier.services.prediction_service import PredictionService

prediction_blueprint = Blueprint("prediction_blueprint", __name__)


@prediction_blueprint.route("/predict", methods=["POST"])
@inject
def predict(
    prediction_service: PredictionService,
    data_preparation_service: DataPreparationService,
) -> Response | tuple[Response, int]:
    """
    Handle a POST request to predict the language of a given text.

    This endpoint expects a JSON payload containing a "text" field.
    It uses the injected prediction and data preparation services to process
    the input and return the predicted language.

    Args:
        prediction_service (PredictionService): Injected Service responsible for language prediction.
        data_preparation_service (DataPreparationService): Injected Service that preprocesses input text.

    Returns:
        Response | tuple[Response, int]: A JSON response with the prediction result,
        or an error message with a 400 status code if no text is provided.

    """
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction, result_language = prediction_service.predict_phrase_language(
        text,
        data_preparation_service,
    )
    return jsonify({"prediction": prediction, "language": result_language})
