# 🧠 Language Classifier

A Python project for detecting whether a given text is written in Italian or not using machine learning models. It includes a REST API, Jupyter notebooks for data analysis and model training, and unit tests.

--------------------------------------------------------------------

## 📚 Dataset and Modeling Approach

### Dataset

This project uses the [Language Detection Dataset](https://www.kaggle.com/datasets/basilb2s/language-detection) from Kaggle, which contains over 10,000 text samples labeled with their corresponding languages. For this project, the goal was to classify sentences as either Italian or Not Italian. To address this, we created a new binary label to distinguish between the two classes.

### Modeling Approach

Two classification algorithms were evaluated:

- **Multinomial Naive Bayes**
- **Logistic Regression**

Text preprocessing was followed by vectorization using two common strategies:

- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**

The analysis included both pre-built models and vectorizers from the **scikit-learn** library as well as custom-built models developed from scratch.  

- Using scikit-learn’s implementations allowed for leveraging well-tested, efficient algorithms and vectorization techniques, facilitating rapid experimentation and benchmarking.  

- At the same time, constructing models from scratch provided deeper insights into the inner workings of the algorithms and enabled tailored modifications to better fit the specific requirements of the language identification task.  


After comparing performance metrics on a validation set, both **Multinomial Naive Bayes** and **Logistic Regression** showed similar results in terms of accuracy.

> ✅ **Multinomial Naive Bayes** combined with **Bag of Words (BoW)** was ultimately selected

This choice was made not only for its solid performance on small datasets, but also for its speed and simplicity. Naive Bayes serves as a strong baseline and a practical, interpretable solution for quick Italian language detection.

--------------------------------------------------------------------


## 🚀 Getting Started

### Start the project from scratch

###### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
```

###### 2. Install dependencies

```bash
pip install -r requirements.txt
```

###### 3. Run the API server

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
  - Input: JSON with text (`{"text": "La mia frase Italiana"}`)
  - Output: Prediction (`{
    "prediction": 1
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
3b. **Train Model Scratch Algo** – Training NLP models using models implemented from scratch
4. **Model Evaluation** – Metrics and performance comparisons
4b. **Model Evaluation Scratch Algo** – Metrics and performance comparisons for models implemented from scratch

--------------------------------------------------------------------

## 🛠️ Requirements

- Python 3.12+
- Libraries: `scikit-learn`, `pandas`, `numpy`,`seaborn`, `flask`, `gunicorn`

