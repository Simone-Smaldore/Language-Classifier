import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
from typing import Union
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

PROCESSED_DATASET_FOLDER = "../data/processed"
DATASET_CLEAN_LOCATION = f"{PROCESSED_DATASET_FOLDER}/Language Detection Clean.csv"
DATASET_CLEAN_UNDERSAMPLING_LOCATION = (
    f"{PROCESSED_DATASET_FOLDER}/Language Detection Clean Undersampling.csv"
)
MODEL_FOLDER = "../models"
DATASET_FOLDER = "../data/raw"
DATASET_LOCATION = f"{DATASET_FOLDER}/Language Detection.csv"


class Datasets:

    def __init__(
        self,
        X: csr_matrix,
        y: NDArray[np.float64],
        X_t: csr_matrix,
        y_t: NDArray[np.float64],
        text: NDArray[np.str_],
        text_t: NDArray[np.str_],
    ):
        self.X = X
        self.y = y
        self.X_t = X_t
        self.y_t = y_t
        self.text = text
        self.text_t = text_t


def split_dataset(df: pd.DataFrame, vectorizer: CountVectorizer) -> Datasets:
    X = vectorizer.fit_transform(df["Text"])
    y = LabelEncoder().fit_transform(df["Language"])
    X, X_t, y, y_t, text, text_t = train_test_split(
        X, y, df["Text"].values, test_size=0.2, stratify=y, random_state=1999
    )
    return Datasets(X, y, X_t, y_t, text, text_t)


ModelML = Union[MultinomialNB, GridSearchCV]
