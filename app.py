"""
Movie Recommendation System - Streamlit App
Interactive web application for movie recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E50914;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #564d4d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: ff0000;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load models and data"""
    try:
        with open('recommendation_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        comparison_df = pd.read_csv('model_comparison.csv')
        
        return models, comparison_df
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run the training script first!")
        st.stop()

models, comparison_df = load_data()

# Extract components
item_similarity_df = models['item_similarity']
svd_model = models['svd_model']
train_matrix = models['train_matrix']
movies = models['movies']
ratings = models['ratings']

# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================

def get_movie_info(movie_id):
    """Get movie information"""
    movie = movies[movies['movie_id'] == movie_id]
    if len(movie) > 0:
        return movie.iloc[0]
    return None

def item_based_recommendations(user_id, n=10):
    """Generate recommendations using item-based CF"""
    user_ratings = ratings[ratings['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        # Return popular movies for cold start
        popular = ratings.groupby('item_id').size().sort_values(ascending=False).head(n)
        return popular.index.tolist()
    
    item_scores = {}
    for _, row in user_ratings.iterrows():
        item_id = row['item_id']
        if item_id in item_similarity_df.index:
            similar_items = item_similarity_df[item_id].sort_values(ascending=False)[1:51]
            
            for sim_item, similarity in similar_items.items():
                if sim_item not in user_ratings['item_id'].values:
                    if sim_item not in item_scores:
                        item_scores[sim_item] = 0
                    item_scores[sim_item] += similarity * row['rating']
    
    if len(item_scores) == 0:
        popular = ratings.groupby('item_id').size().sort_values(ascending=False).head(n)
        return popular.index.tolist()
    
    recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [item_id for item_id, score in recommendations]

def get_user_rated_movies(user_id, n=5):
    """Get movies rated by user"""
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('rating', ascending=False).head(n)
    return user_ratings

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🎬 Get Recommendations",
    "📊 Data Insights",
    "🤖 Model Performance",
    "ℹ️ About Project"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.metric("Total Movies", f"{movies['movie_id'].nunique():,}")
st.sidebar.metric("Total Users", f"{ratings['user_id'].nunique():,}")
st.sidebar.metric("Total Ratings", f"{len(ratings):,}")
st.sidebar.metric("Avg Rating", f"{ratings['rating'].mean():.2f}/5.0")

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML-Powered Personalized Movie Suggestions</div>', unsafe_allow_html=True)
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.icons8.com/clouds/400/movie-projector.png", width=200)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Personalized</h3>
            <p>Get movie recommendations tailored to your taste based on collaborative filtering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered</h3>
            <p>Multiple ML models including SVD matrix factorization and hybrid approaches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data-Driven</h3>
            <p>Built on MovieLens 100K dataset with 100,000 real user ratings</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Overview
    st.markdown("### 📚 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### MovieLens 100K Dataset")
        st.markdown("""
        - **943** unique users
        - **1,682** movies
        - **100,000** ratings
        - Rating scale: **1-5 stars**
        - Time period: **1995-1998**
        - Genres: **19** different categories
        """)
    
    with col2:
        # Rating distribution
        rating_dist = ratings['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_dist.index,
            y=rating_dist.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title='Rating Distribution',
            color=rating_dist.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("### 🚀 Getting Started")
    st.info("👈 Use the sidebar to navigate through different sections and explore the recommendation system!")

# ============================================================================
# PAGE 2: GET RECOMMENDATIONS
# ============================================================================

elif page == "🎬 Get Recommendations":
    st.markdown("## 🎬 Get Your Movie Recommendations")
    st.markdown("Select a user ID to see personalized movie recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        user_id = st.selectbox(
            "Select User ID",
            options=sorted(ratings['user_id'].unique()),
            index=0
        )
        
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10
        )
        
        if st.button("🎯 Get Recommendations", type="primary"):
            st.session_state.show_recommendations = True
    
    if 'show_recommendations' in st.session_state and st.session_state.show_recommendations:
        st.markdown("---")
        
        # Show user's rated movies
        st.markdown(f"### 🎬 User {user_id}'s Top Rated Movies")
        user_rated = get_user_rated_movies(user_id, 5)
        
        if len(user_rated) > 0:
            cols = st.columns(5)
            for idx, (_, movie_row) in enumerate(user_rated.iterrows()):
                movie_info = get_movie_info(movie_row['item_id'])
                if movie_info is not None:
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <b>{movie_info['title'][:30]}</b><br>
                            ⭐ {movie_row['rating']:.1f}/5
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Generate recommendations
        st.markdown(f"### 🎯 Recommended Movies for User {user_id}")
        
        with st.spinner("Generating recommendations..."):
            recommended_ids = item_based_recommendations(user_id, num_recommendations)
        
        # Display recommendations
        for i, movie_id in enumerate(recommended_ids, 1):
            movie_info = get_movie_info(movie_id)
            if movie_info is not None:
                # Get genre info
                genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 
                             'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                             'Sci-Fi', 'Thriller', 'War', 'Western']
                genres = [g for g in genre_cols if movie_info[g] == 1]
                genre_str = ", ".join(genres[:3]) if genres else "Unknown"
                
                # Get average rating
                movie_ratings = ratings[ratings['item_id'] == movie_id]
                avg_rating = movie_ratings['rating'].mean() if len(movie_ratings) > 0 else 0
                num_ratings = len(movie_ratings)
                
                with st.container():
                    col1, col2, col3 = st.columns([0.5, 3, 1])
                    with col1:
                        st.markdown(f"### {i}")
                    with col2:
                        st.markdown(f"**{movie_info['title']}**")
                        st.caption(f"🎭 {genre_str}")
                    with col3:
                        st.metric("Avg Rating", f"{avg_rating:.1f}⭐", f"{num_ratings} ratings")
                    st.markdown("---")

# ============================================================================
# PAGE 3: DATA INSIGHTS
# ============================================================================

elif page == "📊 Data Insights":
    st.markdown("## 📊 Data Insights & Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["📈 Rating Analytics", "🎭 Genre Analysis", "👥 User Behavior"])
    
    with tab1:
        st.markdown("### Rating Distribution & Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            rating_counts = ratings['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Number of Ratings'},
                title='Distribution of Ratings',
                text=rating_counts.values,
                color=rating_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Movie popularity
            movie_stats = ratings.groupby('item_id').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            movie_stats.columns = ['movie_id', 'rating_count', 'avg_rating']
            
            fig = px.scatter(
                movie_stats,
                x='rating_count',
                y='avg_rating',
                title='Movie Popularity vs Average Rating',
                labels={'rating_count': 'Number of Ratings', 'avg_rating': 'Average Rating'},
                opacity=0.6,
                color='avg_rating',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top movies
        st.markdown("### 🏆 Top Rated Movies (min 50 ratings)")
        movie_stats_merged = movie_stats.merge(movies[['movie_id', 'title']], on='movie_id')
        top_movies = movie_stats_merged[movie_stats_merged['rating_count'] >= 50].nlargest(10, 'avg_rating')
        
        fig = px.bar(
            top_movies,
            x='avg_rating',
            y='title',
            orientation='h',
            title='Top 10 Highest Rated Movies',
            labels={'avg_rating': 'Average Rating', 'title': 'Movie'},
            text='avg_rating',
            color='avg_rating',
            color_continuous_scale='RdYlGn'
        )
        fig.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
        fig.update_layout(showlegend=False, height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Genre Analysis")
        
        # Genre distribution
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 
                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                     'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genre_counts = movies[genre_cols].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title='Number of Movies per Genre',
                labels={'x': 'Number of Movies', 'y': 'Genre'},
                text=genre_counts.values,
                color=genre_counts.values,
                color_continuous_scale='Teal'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title='Genre Distribution',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### User Behavior Analysis")
        
        # User activity
        user_activity = ratings.groupby('user_id').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                x=user_activity.values,
                nbins=50,
                title='Distribution of User Activity',
                labels={'x': 'Number of Ratings per User', 'y': 'Number of Users'},
                color_discrete_sequence=['#636EFA']
            )
            fig.add_vline(x=user_activity.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {user_activity.mean():.1f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.markdown("#### User Activity Statistics")
            st.metric("Average ratings per user", f"{user_activity.mean():.1f}")
            st.metric("Median ratings per user", f"{user_activity.median():.0f}")
            st.metric("Most active user", f"{user_activity.max()} ratings")
            st.metric("Least active user", f"{user_activity.min()} ratings")

# ============================================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================================

elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance Comparison")
    
    st.markdown("""
    We trained and compared **4 different recommendation algorithms**:
    1. **User-Based Collaborative Filtering** - Recommends based on similar users
    2. **Item-Based Collaborative Filtering** - Recommends based on similar movies
    3. **SVD Matrix Factorization** - Uses latent factor decomposition
    4. **Hybrid Model** - Combines item-based and SVD approaches
    """)
    
    st.markdown("---")
    
    # Model comparison table
    st.markdown("### 📊 Model Comparison Results")
    
    styled_df = comparison_df.style.highlight_max(
        subset=['Precision@10', 'Coverage'],
        color='lightgreen'
    ).highlight_min(
        subset=['Training Time (s)'],
        color='lightgreen'
    ).format({
        'Precision@10': '{:.4f}',
        'Coverage': '{:.4f}',
        'Training Time (s)': '{:.2f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=200)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Precision@10',
            title='Model Precision@10 Comparison',
            text='Precision@10',
            color='Precision@10',
            color_continuous_scale='Greens'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Coverage',
            title='Model Coverage Comparison',
            text='Coverage',
            color='Coverage',
            color_continuous_scale='Blues'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    best_model_idx = comparison_df['Precision@10'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]['Model']
    best_precision = comparison_df.iloc[best_model_idx]['Precision@10']
    
    st.success(f"🏆 **Best Performing Model: {best_model}** with Precision@10 of {best_precision:.4f}")
    
    st.markdown("---")
    
    # Evaluation metrics explanation
    st.markdown("### 📚 Evaluation Metrics Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Precision@10**
        - Measures accuracy of top-10 recommendations
        - Higher is better
        - Indicates how many recommended movies the user actually likes
        """)
    
    with col2:
        st.markdown("""
        **Coverage**
        - Percentage of movies that can be recommended
        - Higher is better
        - Indicates diversity of recommendations
        """)

# ============================================================================
# PAGE 5: ABOUT PROJECT
# ============================================================================

elif page == "ℹ️ About Project":
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **Movie Recommendation System** was built as part of an ML internship journey.
    It demonstrates the implementation of various collaborative filtering techniques
    for building personalized recommendation engines.
    
    ### 🛠️ Technologies Used
    
    - **Python** - Core programming language
    - **Pandas & NumPy** - Data manipulation and analysis
    - **Scikit-learn** - Machine learning algorithms (SVD, similarity metrics)
    - **Streamlit** - Interactive web application framework
    - **Plotly** - Interactive data visualizations
    
    ### 📊 Dataset
    
    **MovieLens 100K Dataset** from GroupLens Research
    - 100,000 ratings from 943 users on 1,682 movies
    - Rating scale: 1-5 stars
    - Rich metadata including genres and timestamps
    
    ### 🤖 Recommendation Approaches
    
    #### 1. User-Based Collaborative Filtering
    - Finds users with similar rating patterns
    - Recommends movies liked by similar users
    - Good for capturing user preferences
    
    #### 2. Item-Based Collaborative Filtering
    - Identifies similar movies based on user ratings
    - Recommends movies similar to ones user liked
    - More stable than user-based approach
    
    #### 3. Matrix Factorization (SVD)
    - Decomposes user-item matrix into latent factors
    - Captures hidden patterns in user preferences
    - Handles sparsity well
    
    #### 4. Hybrid Model
    - Combines item-based and SVD approaches
    - Leverages strengths of multiple algorithms
    - Often provides best overall performance
    
    ### 📈 Project Workflow
    
    1. **Data Exploration** - Analyzed dataset characteristics and patterns
    2. **Data Preprocessing** - Created user-item matrices
    3. **Model Training** - Implemented 4 different algorithms
    4. **Evaluation** - Compared models using Precision@K and Coverage
    5. **Deployment** - Built interactive Streamlit application
    
    ### 🎓 Learning Outcomes
    
    - Understanding of collaborative filtering techniques
    - Experience with recommendation system evaluation
    - Practical implementation of matrix factorization
    - Building interactive ML applications with Streamlit
    
    ### 📬 Contact
    
    Built with ❤️ as part of ML internship journey
    """)
    
    st.markdown("---")
    st.markdown("### 🌟 Thank you for exploring this project!")