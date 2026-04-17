import pickle
import pandas as pd

movies, similarity = pickle.load(open("models/content_model.pkl", "rb"))
preds, user_item = pickle.load(open("models/collab_model.pkl", "rb"))

def content_score(movie_name):
    idx = movies[movies['title'] == movie_name].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    return dict(sim_scores)

def hybrid_recommend(user_id, movie_name, alpha=0.6):
    content_scores = content_score(movie_name)

    user_row = user_id - 1
    collab_scores = preds[user_row]

    scores = []

    for i in range(len(collab_scores)):
        score = alpha * content_scores.get(i, 0) + (1 - alpha) * collab_scores[i]
        scores.append((i, score))

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[1:10]

    return [movies.iloc[i[0]].title for i in ranked]