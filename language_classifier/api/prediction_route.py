"""This module contains the prediction route.

It is used to predict whether a phrase is in Italian or English.
"""

from flask import Blueprint, jsonify, request
from injector import inject
from language_classifier.services.data_preparation_service import DataPreparationService
from language_classifier.services.prediction_service import PredictionService

prediction_blueprint = Blueprint("prediction_blueprint", __name__)


@prediction_blueprint.route("/predict", methods=["POST"])
@inject
def predict(
    prediction_service: PredictionService,
    data_preparation_service: DataPreparationService,
):
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction, result_language = prediction_service.predict_phrase_language(
        text, data_preparation_service
    )
    return jsonify({"prediction": prediction, "language": result_language})
