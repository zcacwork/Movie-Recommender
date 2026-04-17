import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import pickle

def train_collaborative():
    ratings = pd.read_csv("data/ratings.csv")

    user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    matrix = user_item.values
    mean = np.mean(matrix, axis=1)

    norm_matrix = matrix - mean.reshape(-1, 1)

    U, sigma, Vt = svds(norm_matrix, k=50)
    sigma = np.diag(sigma)

    preds = np.dot(np.dot(U, sigma), Vt) + mean.reshape(-1, 1)

    pickle.dump((preds, user_item), open("models/collab_model.pkl", "wb"))

if __name__ == "__main__":
    train_collaborative()