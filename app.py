import streamlit as st
import pandas as pd
import pickle

# Load models
movies, similarity = pickle.load(open("models/content_model.pkl", "rb"))
preds, user_item = pickle.load(open("models/collab_model.pkl", "rb"))

st.title("🎬 Movie Recommender System")

user_id = st.number_input("Enter User ID", min_value=1, max_value=600, value=1)

movie_name = st.selectbox("Select a movie", movies['title'].values)

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

if st.button("Recommend"):
    results = hybrid_recommend(user_id, movie_name)

    st.subheader("Top Recommendations:")
    for movie in results:
        st.write(movie)