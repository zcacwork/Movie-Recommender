import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

st.set_page_config(page_title="Movie Recommender", layout="wide")

# =========================
# UI HEADER
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Recommender System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align:center;'>🍿 Personalized Movie Recommendations</h3>",
    unsafe_allow_html=True
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models():
    # Content-based
    movies['genres'] = movies['genres'].str.replace("|", " ")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    similarity = cosine_similarity(tfidf_matrix)

    # Collaborative (optimized)
    user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    matrix = user_item.values
    mean = np.mean(matrix, axis=1)
    norm_matrix = matrix - mean.reshape(-1, 1)

    # 🔥 Reduced k for speed
    U, sigma, Vt = svds(norm_matrix, k=20)
    sigma = np.diag(sigma)

    preds = np.dot(np.dot(U, sigma), Vt) + mean.reshape(-1, 1)

    return similarity, preds, user_item

with st.spinner("Training recommendation model... ⏳"):
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

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    return [movies.iloc[i[0]].title for i in ranked]

# =========================
# POSTER FUNCTION (NO API)
# =========================
def fetch_poster(movie_name):
    return "https://via.placeholder.com/300x450?text=" + movie_name.replace(" ", "+")

# =========================
# USER INPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input("Enter User ID", min_value=1, max_value=600, value=1)

with col2:
    movie_name = st.selectbox("Select a movie", movies['title'].values)

# =========================
# BUTTON ACTION
# =========================
if st.button("🍿 Recommend Movies"):

    with st.spinner("Finding best movies for you... 🎯"):
        results = hybrid_recommend(user_id, movie_name)

    st.markdown("## 🎯 Top Picks For You")

    # Display in grid (Netflix style)
    cols = st.columns(5)

    for i in range(10):
        with cols[i % 5]:
            st.image(fetch_poster(results[i]))
            st.caption(results[i])

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Machine Learning</p>",
    unsafe_allow_html=True
)
