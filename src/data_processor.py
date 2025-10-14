import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path

class DataProcessor:
    """Handles loading and preprocessing of MovieLens data"""
    
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.rating_matrix = None
        self.n_users = 0
        self.n_items = 0
        
        # Setup directories
        self.raw_data_path = Path('data/raw')
        self.processed_data_path = Path('data/processed')
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load ratings and movies data"""
        # Load ratings
        ratings_file = self.raw_data_path / 'u.data'
        self.ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        # Load movies
        movies_file = self.raw_data_path / 'u.item'
        movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                     ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        self.movies = pd.read_csv(
            movies_file,
            sep='|',
            names=movie_cols,
            encoding='latin-1'
        )
        
        # Create genre string
        genre_cols = movie_cols[5:]
        self.movies['genres'] = self.movies[genre_cols].apply(
            lambda x: '|'.join([genre_cols[i] for i, val in enumerate(x) if val == 1]),
            axis=1
        )
        
        # Create rating matrix
        self._create_rating_matrix()
        
        # Save processed data
        self._save_processed_data()
    
    def _create_rating_matrix(self):
        """Create sparse user-item rating matrix"""
        self.n_users = self.ratings['user_id'].max()
        self.n_items = self.ratings['item_id'].max()
        
        # Create sparse matrix (user_id and item_id are 1-indexed, convert to 0-indexed)
        self.rating_matrix = csr_matrix(
            (self.ratings['rating'], 
             (self.ratings['user_id'] - 1, self.ratings['item_id'] - 1)),
            shape=(self.n_users, self.n_items)
        )
    
    def _save_processed_data(self):
        """Save processed data for faster loading"""
        # Save ratings
        self.ratings.to_csv(
            self.processed_data_path / 'ratings.csv',
            index=False
        )
        
        # Save movies
        self.movies[['item_id', 'title', 'genres']].to_csv(
            self.processed_data_path / 'movies.csv',
            index=False
        )
        
        # Save rating matrix
        with open(self.processed_data_path / 'rating_matrix.pkl', 'wb') as f:
            pickle.dump(self.rating_matrix, f)
    
    def get_movie_info(self, item_idx):
        """Get movie information by item index (0-indexed)"""
        item_id = item_idx + 1  # Convert to 1-indexed
        movie = self.movies[self.movies['item_id'] == item_id].iloc[0]
        return {
            'title': movie['title'],
            'genres': movie['genres']
        }
    
    def get_top_movies(self, n=10):
        """Get top N most rated movies"""
        movie_ratings = self.ratings.groupby('item_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_ratings.columns = ['item_id', 'avg_rating', 'num_ratings']
        
        # Filter movies with at least 50 ratings
        movie_ratings = movie_ratings[movie_ratings['num_ratings'] >= 50]
        top_movies = movie_ratings.nlargest(n, 'num_ratings')
        
        # Merge with movie titles
        result = top_movies.merge(
            self.movies[['item_id', 'title', 'genres']], 
            on='item_id'
        )
        
        return result[['title', 'genres', 'avg_rating', 'num_ratings']]