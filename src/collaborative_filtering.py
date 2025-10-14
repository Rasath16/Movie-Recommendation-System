import numpy as np
from scipy.stats import pearsonr

class UserBasedCF:
    """User-Based Collaborative Filtering"""
    
    def __init__(self, rating_matrix):
        self.rating_matrix = rating_matrix
        self.n_users, self.n_items = rating_matrix.shape
    
    def compute_similarity(self, user_idx, metric='cosine'):
        """Compute similarity between user and all other users"""
        user_ratings = self.rating_matrix[user_idx].toarray().flatten()
        similarities = np.zeros(self.n_users)
        
        for i in range(self.n_users):
            if i == user_idx:
                continue
            
            other_ratings = self.rating_matrix[i].toarray().flatten()
            
            # Find common rated items
            common = (user_ratings > 0) & (other_ratings > 0)
            
            if np.sum(common) < 3:  # Need at least 3 common items
                continue
            
            if metric == 'cosine':
                norm_user = np.linalg.norm(user_ratings[common])
                norm_other = np.linalg.norm(other_ratings[common])
                if norm_user > 0 and norm_other > 0:
                    similarities[i] = np.dot(user_ratings[common], other_ratings[common]) / (norm_user * norm_other)
            
            elif metric == 'pearson':
                if np.sum(common) > 1:
                    corr, _ = pearsonr(user_ratings[common], other_ratings[common])
                    similarities[i] = corr if not np.isnan(corr) else 0
        
        return similarities
    
    def recommend(self, user_idx, n=10, metric='cosine'):
        """Generate top-N recommendations"""
        similarities = self.compute_similarity(user_idx, metric)
        user_ratings = self.rating_matrix[user_idx].toarray().flatten()
        unrated = np.where(user_ratings == 0)[0]
        
        predictions = []
        for item_idx in unrated:
            # Get ratings for this item from similar users
            item_ratings = self.rating_matrix[:, item_idx].toarray().flatten()
            rated_users = item_ratings > 0
            
            # Weighted average of ratings from similar users
            if np.any(rated_users):
                weights = similarities[rated_users]
                ratings = item_ratings[rated_users]
                
                if np.sum(np.abs(weights)) > 0:
                    pred = np.dot(weights, ratings) / np.sum(np.abs(weights))
                    predictions.append((item_idx, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class ItemBasedCF:
    """Item-Based Collaborative Filtering"""
    
    def __init__(self, rating_matrix):
        self.rating_matrix = rating_matrix.T  # Transpose for item-based
        self.n_items, self.n_users = self.rating_matrix.shape
    
    def compute_similarity(self, item_idx, metric='cosine'):
        """Compute similarity between item and all other items"""
        item_ratings = self.rating_matrix[item_idx].toarray().flatten()
        similarities = np.zeros(self.n_items)
        
        for i in range(self.n_items):
            if i == item_idx:
                continue
            
            other_ratings = self.rating_matrix[i].toarray().flatten()
            
            # Find common users who rated both items
            common = (item_ratings > 0) & (other_ratings > 0)
            
            if np.sum(common) < 3:
                continue
            
            if metric == 'cosine':
                norm_item = np.linalg.norm(item_ratings[common])
                norm_other = np.linalg.norm(other_ratings[common])
                if norm_item > 0 and norm_other > 0:
                    similarities[i] = np.dot(item_ratings[common], other_ratings[common]) / (norm_item * norm_other)
            
            elif metric == 'pearson':
                if np.sum(common) > 1:
                    corr, _ = pearsonr(item_ratings[common], other_ratings[common])
                    similarities[i] = corr if not np.isnan(corr) else 0
        
        return similarities
    
    def recommend(self, user_idx, n=10, metric='cosine'):
        """Generate top-N recommendations"""
        user_ratings = self.rating_matrix[:, user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]
        unrated_items = np.where(user_ratings == 0)[0]
        
        item_scores = np.zeros(self.n_items)
        
        for rated_item in rated_items:
            similarities = self.compute_similarity(rated_item, metric)
            rating = user_ratings[rated_item]
            item_scores += similarities * rating
        
        # Get predictions for unrated items
        predictions = [(item_idx, item_scores[item_idx]) for item_idx in unrated_items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]