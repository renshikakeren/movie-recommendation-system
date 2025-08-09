# data_preprocessing.py
import pandas as pd

def load_and_prepare_data(file_path='data/movies.csv'):
    """Loads movie data and prepares it for embedding generation."""
    try:
        movies_df = pd.read_csv(file_path)
        # Combine relevant fields for the embedding model
        movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['title']
        return movies_df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

if __name__ == '__main__':
    df = load_and_prepare_data()
    if df is not None:
        print(df.head())