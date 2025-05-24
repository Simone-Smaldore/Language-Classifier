"""Files of constants.

This script contains the constants used in the project to define the paths of the files.
"""

from pathlib import Path

PROJECT_FOLDER: str = str(Path(Path(Path(__file__).resolve()).parent).parent)
MODEL_FOLDER: str = str(Path(PROJECT_FOLDER, "model"))
CONFIGURATION_FOLDER: str = str(Path(PROJECT_FOLDER, "configurations"))
