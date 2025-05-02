import pandas as pd


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["review", "sentiment"])
    df["sentiment_numeric"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df.dropna(subset=["sentiment_numeric"])
    X = df["review"]
    y = df["sentiment_numeric"].astype(int)
    return X, y
