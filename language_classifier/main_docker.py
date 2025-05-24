"""
Main function of the application.

Contains the startup of the application and the configuration of the injection.
"""

import logging

from flask import Flask
from flask_cors import CORS
from flask_injector import FlaskInjector
from injector import Binder, singleton

from language_classifier.api.prediction_route import prediction_blueprint
from language_classifier.const import CONFIGURATION_FOLDER
from language_classifier.logging_module.logger_initalizer import setup_logging
from language_classifier.services.data_preparation_service import DataPreparationService
from language_classifier.services.model_loader_service import ModelLoaderService
from language_classifier.services.prediction_service import PredictionService

logger = logging.getLogger("main_docker")


def configure_injection(binder: Binder) -> None:
    """
    Configure injection.

    This function contains instructions for configuring injection.
    """
    binder.bind(PredictionService, to=PredictionService, scope=singleton)
    binder.bind(DataPreparationService, to=DataPreparationService, scope=singleton)


setup_logging(f"{CONFIGURATION_FOLDER}/config-logger.json")
app = Flask(__name__)
CORS(app)
app.register_blueprint(prediction_blueprint, url_prefix="/")
ModelLoaderService()
logger.info("Starting application")
FlaskInjector(app=app, modules=[configure_injection])
