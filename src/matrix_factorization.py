import numpy as np

class SVD:
    """Matrix Factorization using SVD with Gradient Descent"""
    
    def __init__(self, rating_matrix, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.rating_matrix = rating_matrix
        self.n_users, self.n_items = rating_matrix.shape
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        
        # Initialize factor matrices
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, n_factors))
        
        # Biases
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = rating_matrix.data.mean()
    
    def train(self):
        """Train the model using SGD"""
        rows, cols = self.rating_matrix.nonzero()
        ratings = self.rating_matrix.data
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(rows))
            
            for idx in indices:
                u, i, r = rows[idx], cols[idx], ratings[idx]
                
                # Prediction
                pred = (self.global_bias + self.user_bias[u] + 
                       self.item_bias[i] + 
                       np.dot(self.user_factors[u], self.item_factors[i]))
                
                # Error
                error = r - pred
                
                # Update biases
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])
                
                # Update factors
                user_f = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (error * user_f - self.reg * self.item_factors[i])
    
    def predict(self, user_idx, item_idx):
        """Predict rating for user-item pair"""
        pred = (self.global_bias + self.user_bias[user_idx] + 
                self.item_bias[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        return np.clip(pred, 1, 5)
    
    def recommend(self, user_idx, n=10):
        """Generate top-N recommendations"""
        user_ratings = self.rating_matrix[user_idx].toarray().flatten()
        unrated = np.where(user_ratings == 0)[0]
        
        predictions = []
        for item_idx in unrated:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]