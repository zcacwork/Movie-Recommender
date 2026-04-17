import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")
st.markdown("## 🍿 Personalized Movie Recommendations")

st.write("App started...")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

movies, ratings = load_data()
st.write("Data loaded ✅")

# =========================
# TRAIN MODELS (AUTO)
# =========================
@st.cache_resource
def train_models():
    st.write("Training models... ⏳")

    # ---------- CONTENT MODEL ----------
    movies['genres'] = movies['genres'].str.replace("|", " ")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    similarity = cosine_similarity(tfidf_matrix)

    # ---------- COLLAB MODEL ----------
    user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    matrix = user_item.values
    mean = np.mean(matrix, axis=1)
    norm_matrix = matrix - mean.reshape(-1, 1)

    # 🔥 REDUCED K (IMPORTANT → prevents blank screen)
    U, sigma, Vt = svds(norm_matrix, k=20)
    sigma = np.diag(sigma)

    preds = np.dot(np.dot(U, sigma), Vt) + mean.reshape(-1, 1)

    st.write("Models trained ✅")

    return similarity, preds, user_item

with st.spinner("Training models... please wait ⏳"):
    similarity, preds, user_item = train_models()

# =========================
# RECOMMENDATION FUNCTION
# =========================
def hybrid_recommend(user_id, movie_name, alpha=0.6):
    idx = movies[movies['title'] == movie_name].index[0]

    content_scores = list(enumerate(similarity[idx]))
    user_row = user_id - 1
    collab_scores = preds[user_row]

    scores = []
    for i in range(len(collab_scores)):
        score = alpha * content_scores[i][1] + (1 - alpha) * collab_scores[i]
        scores.append((i, score))

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[1:10]

    return [movies.iloc[i[0]].title for i in ranked]

# =========================
# UI INPUT
# =========================
user_id = st.number_input("Enter User ID", min_value=1, max_value=600, value=1)
movie_name = st.selectbox("Select a movie", movies['title'].values)

# =========================
# BUTTON ACTION
# =========================
import requests

API_KEY = "YOUR_TMDB_API_KEY"

def fetch_poster(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    data = requests.get(url).json()

    if data["results"]:
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path

    return "https://via.placeholder.com/500x750?text=No+Image"


if st.button("🍿 Recommend"):
    with st.spinner("Fetching recommendations..."):
        results = hybrid_recommend(user_id, movie_name)

    st.subheader("🎯 Top Picks For You")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            poster = fetch_poster(results[i])
            st.image(poster)
            st.caption(results[i])

st.write("UI Loaded Successfully ✅")
