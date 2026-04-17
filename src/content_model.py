import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_content():
    movies = pd.read_csv("data/movies.csv")

    movies['genres'] = movies['genres'].str.replace("|", " ")
    
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(movies['genres'])

    similarity = cosine_similarity(matrix)

    pickle.dump((movies, similarity), open("models/content_model.pkl", "wb"))

if __name__ == "__main__":
    train_content()