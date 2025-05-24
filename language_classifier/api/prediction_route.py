"""This module contains the prediction route.

It is used to predict whether a phrase is in Italian or English.
"""

from flask import Blueprint, jsonify

prediction_blueprint = Blueprint("prediction_blueprint", __name__)


@prediction_blueprint.route("/predict", methods=["POST"])
def predict():
    print("test")
    return jsonify({"language": "prediction"})
