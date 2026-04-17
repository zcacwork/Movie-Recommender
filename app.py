import os
import pickle
from src.content_model import train_content
from src.collaborative_model import train_collaborative

# Train models if not present
if not os.path.exists("models/content_model.pkl"):
    train_content()

if not os.path.exists("models/collab_model.pkl"):
    train_collaborative()

movies, similarity = pickle.load(open("models/content_model.pkl", "rb"))
preds, user_item = pickle.load(open("models/collab_model.pkl", "rb"))
