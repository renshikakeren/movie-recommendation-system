# recommender.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load and prepare data (you can call this from data_preprocessing.py)
def load_data(file_path='data/movies.csv'):
    df = pd.read_csv(file_path)
    df['combined_features'] = df['genres'] + ' ' + df['title']
    return df

# Initialize the Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_recommendations(user_query, top_n=5, movie_df=None, movie_embeddings=None):
    """Generates movie recommendations based on a user query."""
    if movie_df is None or movie_embeddings is None:
        movie_df = load_data()
        features = movie_df['combined_features'].tolist()
        movie_embeddings = model.encode(features, convert_to_tensor=False)

    # Encode the user query
    query_embedding = model.encode(user_query, convert_to_tensor=False).reshape(1, -1)

    # Calculate cosine similarity between the query and all movies
    similarities = cosine_similarity(query_embedding, movie_embeddings)[0]

    # Get the top N most similar movies
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    recommended_movies = movie_df.iloc[top_indices]
    return recommended_movies.to_dict('records')

if __name__ == '__main__':
    # Example usage
    df = load_data()
    features = df['combined_features'].tolist()
    embeddings = model.encode(features, convert_to_tensor=False)

    recommendations = get_recommendations("sci-fi movies from the 90s", movie_df=df, movie_embeddings=embeddings)
    print("Recommendations:", recommendations)