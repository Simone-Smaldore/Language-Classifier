"""Contains the DataPreparationService to prepare data for predictions."""

import logging
import unicodedata

from scipy.sparse import csr_matrix

from language_classifier.services.model_loader_service import ModelLoaderService

logger = logging.getLogger("data_preparation_service")


class DataPreparationService:
    """Class used to prepare the data for the predictions."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.logger = logging.getLogger("data_preparation_service")

    def prepare_data(self, text: str) -> csr_matrix:
        """
        Preprocess and transform input text into a sparse feature matrix.

        This method cleans the input text using an internal cleaning function
        and then transforms the cleaned text into a sparse matrix representation
        using a preloaded vectorizer from the ModelLoaderService.

        Args:
            text (str): The raw input text to preprocess.

        Returns:
            csr_matrix: The sparse matrix representation of the processed text.

        """
        cleaned_text = self.clean_text(text)
        return ModelLoaderService().vectorizer.transform([cleaned_text])

    def clean_text(self, text: str) -> str:
        """
        Clean and normalizes a text string by lowercasing, removing digits, and stripping punctuation.

        This function performs several text preprocessing steps:
        - Converts all characters to lowercase.
        - Replaces apostrophes with spaces.
        - Removes all digits.
        - Keeps only alphanumeric characters and whitespace.
        - Cleans remaining Unicode characters while preserving accents (via `clean_text_keep_accents`).
        - Normalizes whitespace to ensure consistent spacing.

        Args:
            text (str): The input text string to be cleaned.

        Returns:
            str: The cleaned and normalized text.

        """
        text = text.lower()
        text = text.replace("'", " ")
        text = "".join(c for c in text if not c.isdigit())
        text = "".join(c for c in text if c.isalnum() or c.isspace())
        text = self._clean_text_keep_accents(text)
        return " ".join(text.split())

    def _clean_text_keep_accents(self, text: str) -> str:
        """
        Clean a string by removing special characters while preserving accents.

        This function removes all characters from the input string except letters,
        numbers, and whitespace. It retains accented characters and strips out
        special symbols, including modifier letters and punctuation.

        Args:
            text (str): The input text string to be cleaned.

        Returns:
            str: The cleaned text with accents preserved and special characters removed.

        """
        return "".join(
            c
            for c in text
            if (unicodedata.category(c).startswith(("L", "N")) or c.isspace())
            and not unicodedata.name(c, "").startswith("MODIFIER")
        )
