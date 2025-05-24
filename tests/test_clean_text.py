"""Test module for DataPreparationService.clean_text method."""

import pytest

from language_classifier.services.data_preparation_service import DataPreparationService


@pytest.mark.parametrize(
    ("input_text", "expected_output"),
    [
        ("Hello, World!", "hello world"),
        ("Ciao!!!", "ciao"),
        ("The pen is ON THE TaBle.", "the pen is on the table"),
        ("123 ABC!", "abc"),
        ("No$%^&*()Symbols", "nosymbols"),
    ],
)
def test_clean_text(input_text: str, expected_output: str) -> None:
    """
    Test the clean_text method to ensure it normalizes text correctly.

    Args:
        input_text (str): The input string to clean.
        expected_output (str): The expected cleaned output string.

    """
    data_preparation_service = DataPreparationService()
    result = data_preparation_service.clean_text(input_text)
    assert result == expected_output
