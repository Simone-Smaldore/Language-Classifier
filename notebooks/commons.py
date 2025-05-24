"""
Provides data structures, constants, and utility functions for preprocessing, vectorizing, and splitting a language detection dataset.

It includes:

- Paths to dataset and model files.
- The `Datasets` container class for storing training and test sets,
  including feature matrices, labels, and original text data.
- The `vectorize_and_split_dataset` function, which vectorizes text data,
  encodes labels, and splits data into stratified training and test sets.
- A type alias `ModelML` representing the machine learning models used,
  either a Naive Bayes classifier or a GridSearchCV instance.

Dependencies:
- numpy
- pandas
- scipy
- scikit-learn
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

PROCESSED_DATASET_FOLDER = "../data/processed"
DATASET_CLEAN_LOCATION = f"{PROCESSED_DATASET_FOLDER}/Language Detection Clean.csv"
DATASET_CLEAN_UNDERSAMPLING_LOCATION = (
    f"{PROCESSED_DATASET_FOLDER}/Language Detection Clean Undersampling.csv"
)
MODEL_FOLDER = "../models"
DATASET_FOLDER = "../data/raw"
DATASET_LOCATION = f"{DATASET_FOLDER}/Language Detection.csv"
VECTORIZERS_FOLDER = "../vectorizers"


class Datasets:
    """
    Container class to hold training and test datasets for machine learning tasks.

    This class stores feature matrices, labels, and original text data for both
    training and testing sets in a structured way.
    """

    def __init__(
        self,
        X: csr_matrix,
        y: NDArray[np.float64],
        X_t: csr_matrix,
        y_t: NDArray[np.float64],
        text: NDArray[np.str_],
        text_t: NDArray[np.str_],
    ) -> None:
        """
        Initialize the Datasets container with training and test data.

        Args:
            X (csr_matrix): Training feature matrix (sparse).
            y (NDArray[np.float64]): Training labels.
            X_t (csr_matrix): Test feature matrix (sparse).
            y_t (NDArray[np.float64]): Test labels.
            text (NDArray[np.str_]): Original training text samples.
            text_t (NDArray[np.str_]): Original test text samples.

        """
        self.X = X
        self.y = y
        self.X_t = X_t
        self.y_t = y_t
        self.text = text
        self.text_t = text_t


def vectorize_and_split_dataset(
    df: pd.DataFrame,
    vectorizer: CountVectorizer,
) -> Datasets:
    """
    Split the dataset into training and test sets and vectorizes the text data.

    This function takes a DataFrame with text data and corresponding language labels,
    applies the provided vectorizer to transform the text into feature vectors,
    encodes the language labels, and splits everything into training and test subsets
    with stratification to preserve class distribution.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'Text' and 'Language' columns.
        vectorizer (CountVectorizer): A scikit-learn vectorizer instance to transform text data.

    Returns:
        Datasets: A Datasets object containing:
            - X: Training feature vectors
            - y: Training labels
            - X_t: Test feature vectors
            - y_t: Test labels
            - text: Training text samples (original)
            - text_t: Test text samples (original)

    """
    X = vectorizer.fit_transform(df["Text"])
    y = LabelEncoder().fit_transform(df["Language"])
    X, X_t, y, y_t, text, text_t = train_test_split(
        X,
        y,
        df["Text"].values,
        test_size=0.2,
        stratify=y,
        random_state=1999,
    )
    return Datasets(X, y, X_t, y_t, text, text_t)


ModelML = MultinomialNB | GridSearchCV
