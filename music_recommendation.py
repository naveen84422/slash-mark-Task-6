import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
tracks = pd.read_csv('tracks.csv')  # Replace with actual dataset path
user_ratings = pd.read_csv('user_ratings.csv')  # Replace with actual dataset path

# Display first few rows
print(tracks.head())
print(user_ratings.head())

# Data Preprocessing
tracks.fillna('', inplace=True)

# Content-Based Filtering (TF-IDF on track metadata)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(tracks['genre'] + ' ' + tracks['artist'] + ' ' + tracks['album'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to Recommend Songs Based on Content
def recommend_songs(song_title, top_n=10):
    if song_title not in tracks['title'].values:
        return "Song not found in database."
    idx = tracks[tracks['title'] == song_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]
    return tracks.iloc[song_indices][['title', 'artist', 'genre']]

# Collaborative Filtering (Matrix Factorization)
user_item_matrix = user_ratings.pivot(index='user_id', columns='track_id', values='rating').fillna(0)
U, sigma, Vt = svds(user_item_matrix, k=50)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Function to Recommend Songs Based on User History
def recommend_for_user(user_id, top_n=10):
    if user_id not in predicted_ratings_df.index:
        return "User not found."
    user_ratings = predicted_ratings_df.loc[user_id].sort_values(ascending=False).head(top_n)
    return tracks[tracks['track_id'].isin(user_ratings.index)][['title', 'artist', 'genre']]

# Visualizing Top Rated Songs
average_ratings = user_ratings.groupby('track_id')['rating'].mean().sort_values(ascending=False)
top_rated_tracks = tracks[tracks['track_id'].isin(average_ratings.head(10).index)]
plt.figure(figsize=(10,5))
sns.barplot(x=top_rated_tracks['title'], y=average_ratings.head(10).values)
plt.xticks(rotation=90)
plt.title("Top Rated Songs")
plt.xlabel("Songs")
plt.ylabel("Average Rating")
plt.show()

# Function to Find Similar Users
def similar_users(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return "User not found."
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, user_item_matrix)
    similar_users = list(enumerate(similarity_scores[0]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [user_item_matrix.index[i[0]] for i in similar_users]

# Example Usage
print("Content-Based Recommendations:")
print(recommend_songs('Shape of You'))

print("\nCollaborative Filtering Recommendations:")
print(recommend_for_user(1))

print("\nSimilar Users to User 1:")
print(similar_users(1))
