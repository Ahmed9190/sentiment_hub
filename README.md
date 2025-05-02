# SentimentHub

A production-ready, modular sentiment analysis project with MLOps best practices.

## 🚀 Overview

SentimentHub is a complete pipeline for sentiment analysis on IMDB movie reviews, covering:

- Data exploration and preprocessing
- Model training and evaluation
- Version control for data/models (DVC)
- Serving predictions via FastAPI API
- Automated testing

## 📂 Project Structure

```
SentimentHub/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── api/
│   ├── data_preprocessing.py
│   └── model_training.py
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
├── .dvcignore
└── LICENSE
```

## ⚡️ Quickstart

### 1. Clone & Setup

```
git clone
cd SentimentHub
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Download Data

Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data) and put it in `data/imdb.csv`.

### 3. Train the Model

```
python src/model_training.py
```

### 4. Track Data & Model with DVC

```
dvc add data/imdb.csv
dvc add models/model.pkl
dvc add models/vectorizer.pkl
git add data/imdb.csv.dvc models/model.pkl.dvc models/vectorizer.pkl.dvc .dvcignore dvc.yaml dvc.lock
git commit -m "Track data and models with DVC"
```

### 5. Serve the Model as an API

```
uvicorn src.api.main:app --reload
```

Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive docs.

### 6. Run Tests

```
pytest
```

## 🧪 Example API Call

```
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\": \"I love this movie\!\"}"
```

Response:

```
{
  "sentiment": "positive",
  "confidence": 0.9070903087882695
}
```

## 🛠️ Tech Stack

- Python, pandas, scikit-learn, joblib
- DVC for data/model versioning
- FastAPI & Uvicorn for API
- pytest for testing

## 📄 License

MIT License (see [LICENSE](LICENSE) for details)
