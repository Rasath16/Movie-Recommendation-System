import numpy as np
from sklearn.model_selection import train_test_split

def evaluate_model(model, rating_matrix, k=10, method='user', test_size=0.2, threshold=4.0):
    """
    Evaluate recommendation model using Precision@K and Recall@K
    
    Parameters:
    -----------
    model : object
        Recommendation model with recommend() method
    rating_matrix : scipy.sparse matrix
        User-item rating matrix
    k : int
        Number of recommendations
    method : str
        'user', 'item', or 'svd'
    test_size : float
        Proportion of data for testing
    threshold : float
        Rating threshold for relevant items
    
    Returns:
    --------
    precision : float
        Average precision@k
    recall : float
        Average recall@k
    """
    # Create train-test split
    rows, cols = rating_matrix.nonzero()
    ratings = rating_matrix.data
    
    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    # Group test data by user
    test_data = {}
    for idx in test_idx:
        u, i, r = rows[idx], cols[idx], ratings[idx]
        if u not in test_data:
            test_data[u] = []
        if r >= threshold:
            test_data[u].append(i)
    
    # Evaluate each user
    precisions = []
    recalls = []
    
    for user_idx in test_data:
        if len(test_data[user_idx]) == 0:
            continue
        
        try:
            # Get recommendations
            recommendations = model.recommend(user_idx, k)
            rec_items = [item_idx for item_idx, _ in recommendations]
            relevant_items = set(test_data[user_idx])
            
            # Calculate metrics
            hits = len(set(rec_items) & relevant_items)
            
            precision = hits / len(rec_items) if len(rec_items) > 0 else 0
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        except:
            continue
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    return avg_precision, avg_recall