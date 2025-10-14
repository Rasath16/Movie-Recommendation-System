import streamlit as st
import sys
import os

# Add src to path - multiple approaches for reliability
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import modules
try:
    from data_processor import DataProcessor
    from collaborative_filtering import UserBasedCF, ItemBasedCF
    from matrix_factorization import SVD
    from evaluator import evaluate_model
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Please ensure all files are in the correct directory structure.")
    st.stop()

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Title
st.title("🎬 Movie Recommendation System")
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.processor = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data loading
    st.subheader("1. Load Data")
    if st.button("Load MovieLens Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                processor = DataProcessor()
                processor.load_data()
                st.session_state.processor = processor
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    if st.session_state.data_loaded:
        st.divider()
        
        # Algorithm selection
        st.subheader("2. Select Algorithm")
        algorithm = st.selectbox(
            "Method",
            ["User-Based CF", "Item-Based CF", "Matrix Factorization (SVD)"]
        )
        
        st.subheader("3. Parameters")
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=1)
        top_k = st.slider("Top K Recommendations", 5, 20, 10)
        
        if algorithm in ["User-Based CF", "Item-Based CF"]:
            similarity = st.selectbox("Similarity", ["cosine", "pearson"])
        else:
            n_factors = st.slider("Latent Factors", 20, 100, 50)
            n_epochs = st.slider("Epochs", 10, 30, 20)

# Main content
if st.session_state.data_loaded:
    processor = st.session_state.processor
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Users", processor.n_users)
    with col2:
        st.metric("Movies", processor.n_items)
    with col3:
        st.metric("Ratings", len(processor.ratings))
    with col4:
        sparsity = 1 - (len(processor.ratings) / (processor.n_users * processor.n_items))
        st.metric("Sparsity", f"{sparsity:.1%}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "📈 Evaluation", "📉 Analysis"])
    
    with tab1:
        st.subheader(f"Top {top_k} Recommendations for User {user_id}")
        
        if st.button("Generate Recommendations", type="primary", use_container_width=True):
            with st.spinner("Computing recommendations..."):
                try:
                    # Get recommendations based on selected algorithm
                    if algorithm == "User-Based CF":
                        model = UserBasedCF(processor.rating_matrix)
                        recs = model.recommend(user_id - 1, top_k, similarity)
                    elif algorithm == "Item-Based CF":
                        model = ItemBasedCF(processor.rating_matrix)
                        recs = model.recommend(user_id - 1, top_k, similarity)
                    else:
                        model = SVD(processor.rating_matrix, n_factors, n_epochs)
                        model.train()
                        recs = model.recommend(user_id - 1, top_k)
                    
                    # Display recommendations
                    for idx, (movie_idx, score) in enumerate(recs, 1):
                        movie_info = processor.get_movie_info(movie_idx)
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{idx}. {movie_info['title']}**")
                            st.caption(f"Genres: {movie_info['genres']}")
                        with col2:
                            st.metric("Score", f"{score:.2f}")
                        if idx < len(recs):
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Model Evaluation")
        
        eval_k = st.slider("Evaluate at K", 5, 20, 10, key="eval_k")
        
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating..."):
                try:
                    if algorithm == "User-Based CF":
                        model = UserBasedCF(processor.rating_matrix)
                        precision, recall = evaluate_model(model, processor.rating_matrix, eval_k, 'user')
                    elif algorithm == "Item-Based CF":
                        model = ItemBasedCF(processor.rating_matrix)
                        precision, recall = evaluate_model(model, processor.rating_matrix, eval_k, 'item')
                    else:
                        model = SVD(processor.rating_matrix, n_factors, n_epochs)
                        model.train()
                        precision, recall = evaluate_model(model, processor.rating_matrix, eval_k, 'svd')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Precision@{eval_k}", f"{precision:.4f}")
                    with col2:
                        st.metric(f"Recall@{eval_k}", f"{recall:.4f}")
                    with col3:
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        st.metric("F1-Score", f"{f1:.4f}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab3:
        st.subheader("Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Rating Distribution")
            rating_dist = processor.ratings['rating'].value_counts().sort_index()
            st.bar_chart(rating_dist)
        
        with col2:
            st.markdown("#### Top 10 Most Rated Movies")
            top_movies = processor.get_top_movies(10)
            st.dataframe(top_movies, use_container_width=True)

else:
    st.info("👈 Click 'Load MovieLens Data' in the sidebar to start!")
    st.markdown("""
    ### Instructions:
    1. Ensure `u.data` and `u.item` files are in `data/raw/` directory
    2. Click the 'Load MovieLens Data' button
    3. Select an algorithm and configure parameters
    4. Generate recommendations and evaluate performance
    
    ### Algorithms Available:
    - **User-Based Collaborative Filtering**: Recommends based on similar users
    - **Item-Based Collaborative Filtering**: Recommends based on similar movies
    - **Matrix Factorization (SVD)**: Advanced latent factor model
    """)