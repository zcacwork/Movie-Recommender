import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle

def rmse():
    ratings = pd.read_csv("data/ratings.csv")
    preds, user_item = pickle.load(open("models/collab_model.pkl", "rb"))

    actual = []
    predicted = []

    for row in ratings.itertuples():
        user = row.userId - 1
        movie = list(user_item.columns).index(row.movieId)

        actual.append(row.rating)
        predicted.append(preds[user][movie])

    return np.sqrt(mean_squared_error(actual, predicted))

if __name__ == "__main__":
    print("RMSE:", rmse())