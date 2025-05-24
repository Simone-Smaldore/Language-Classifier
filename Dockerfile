FROM python:3.12-slim
WORKDIR /
COPY requirements.txt .
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "language_classifier.main_docker:app"]