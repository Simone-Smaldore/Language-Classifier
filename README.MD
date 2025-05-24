# 🧠 Language Classifier

A Python project for detecting whether a given text is written in Italian or English using machine learning models. It includes a REST API, Jupyter notebooks for data analysis and model training, and unit tests.

--------------------------------------------------------------------

## 📚 Dataset and Modeling Approach

### Dataset

This project uses the [Language Detection Dataset](https://www.kaggle.com/datasets/basilb2s/language-detection) from Kaggle, which contains over 20,000 text samples labeled with their corresponding languages. For this project, only **English** and **Italian** samples were selected to build a binary classification model.

### Modeling Approach

Two classification algorithms were evaluated:

- **Multinomial Naive Bayes**
- **Logistic Regression**

Text preprocessing was followed by vectorization using two common strategies:

- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**

After comparing performance metrics on a validation set, both **Multinomial Naive Bayes** and **Logistic Regression** showed similar results in terms of accuracy.

> ✅ **Multinomial Naive Bayes** combined with **Bag of Words (BoW)** was ultimately selected

This choice was made not only for its competitive performance but also for its simplicity and interpretability, making it a practical and explainable solution for fast language detection between Italian and English.

--------------------------------------------------------------------


## 🚀 Getting Started

### Start the project from scratch

###### 1. Clone the repository

```bash
git clone https://github.com/your-username/language-classifier.git
cd language-classifier
```

###### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

###### 3. Install dependencies

```bash
pip install -r requirements.txt
```

###### 4. Run the API server

```bash
python language_classifier/main.py
```

### Start the project using Docker

```bash
docker build -t language-classifier .
docker run -p 5000:5000 language-classifier
```

###### Alternatively, use Docker Compose:

```bash
docker-compose up
```

--------------------------------------------------------------------

## 📡 API Endpoint

- `POST /predict`
  - Input: JSON with text (`{"text": "My English Sentence"}`)
  - Output: Predicted language (`{
    "language": "English",
    "prediction": 0
}`) 

--------------------------------------------------------------------

## 🧪 Linting and Testing


The project uses tox to manage linting and testing pipelines

Run this command for the complete pipeline:
```bash
tox
```
To execute ruff linting on the python code and on the jupyter notebooks:

```bash
tox run -e lint
```
To execute mypy type checking on the python code and on the jupyter notebooks:

```bash
tox run -e type-check
```
To execute the tests:

```bash
tox run -e test
```

--------------------------------------------------------------------

## 📊 Included Notebooks

1. **Data Exploration** – Initial dataset analysis
2. **Data Cleaning** – Text preprocessing and normalization
3. **Train Model** – Training NLP models
4. **Model Evaluation** – Metrics and performance comparisons

--------------------------------------------------------------------

## 🛠️ Requirements

- Python 3.12+
- Libraries: `scikit-learn`, `pandas`, `numpy`,`seaborn`, `flask`, `gunicorn`

