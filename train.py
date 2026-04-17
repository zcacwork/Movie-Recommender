from src.content_model import train_content
from src.collaborative_model import train_collaborative

print("Training content model...")
train_content()

print("Training collaborative model...")
train_collaborative()

print("Done!")