"""This module contains the prediction route.

It is used to predict whether a phrase is in Italian or English.
"""

from flask import Blueprint, jsonify
from injector import inject

from language_classifier.services.prediction_service import PredictionService

prediction_blueprint = Blueprint("prediction_blueprint", __name__)


@prediction_blueprint.route("/predict", methods=["POST"])
@inject
def predict(prediction_service: PredictionService):
    print("test")
    print(prediction_service)
    return jsonify({"language": "prediction"})
