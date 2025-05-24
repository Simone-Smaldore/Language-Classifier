"""
Collection of utility functions for logging.

There is a function for logging setup.
"""

import json
import logging.config
from pathlib import Path


def setup_logging(file_path: str) -> None:
    """
    Set up the application logger using the file path of a JSON configuration file.

    Args :
        file_path (str): the relative file path to the logger json file
    """
    config_file = Path(file_path)
    with Path.open(config_file) as file_conf:
        logging_config = json.load(file_conf)
    logging.config.dictConfig(config=logging_config)
