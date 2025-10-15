import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import requests

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
        background-color: #151B54;
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


# LOAD MODELS AND DATA


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

# HELPER FUNCTIONS

import requests
import re

@st.cache_data
def get_movie_poster(title):
    """Get movie poster from TMDb API with caching"""
    try:
        TMDB_API_KEY = "7891144d4b5142e348389f3caeef27f3"
        url = "https://api.themoviedb.org/3/search/movie"
        
        # Extract year from title (e.g., "Toy Story (1995)")
        year = None
        year_match = re.search(r'\((\d{4})\)', title)
        if year_match:
            year = year_match.group(1)
            clean_title = title.replace(year_match.group(0), '').strip()
        else:
            clean_title = title
        
        # Build params
        params = {
            'api_key': TMDB_API_KEY,
            'query': clean_title,
            'include_adult': False
        }
        if year:
            params['year'] = year
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        # Try to find poster
        if data.get('results') and len(data['results']) > 0:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        # Fallback to placeholder
        safe_title = clean_title.replace(' ', '+')[:20]
        return f"https://via.placeholder.com/300x450/667eea/ffffff?text={safe_title}"
        
    except:
        safe_title = title.replace(' ', '+').replace('(', '').replace(')', '')[:20]
        return f"https://via.placeholder.com/300x450/667eea/ffffff?text={safe_title}"

def get_movie_info(movie_id):
    """Get movie information"""
    movie = movies[movies['movie_id'] == movie_id]
    if len(movie) > 0:
        return movie.iloc[0]
    return None

def get_user_rated_movies(user_id, n=5):
    """Get movies rated by user"""
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('rating', ascending=False).head(n)
    return user_ratings

def search_movies(query):
    """Search movies by title"""
    query_lower = query.lower()
    results = movies[movies['title'].str.lower().str.contains(query_lower, na=False)]
    return results

def get_similar_movies(movie_id, n=10):
    """Get similar movies based on item similarity"""
    if movie_id not in item_similarity_df.index:
        return []
    
    similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]
    return similar_movies.index.tolist()

def item_based_recommendations(user_id, n=10):
    """Generate recommendations using item-based CF"""
    user_ratings = ratings[ratings['user_id'] == user_id]
    
    if len(user_ratings) == 0:
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


# SIDEBAR


st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🎬 Get Recommendations",
    "📊 Data Insights",
    "🤖 Model Performance",
    
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.metric("Total Movies", f"{movies['movie_id'].nunique():,}")
st.sidebar.metric("Total Users", f"{ratings['user_id'].nunique():,}")
st.sidebar.metric("Total Ratings", f"{len(ratings):,}")
st.sidebar.metric("Avg Rating", f"{ratings['rating'].mean():.2f}/5.0")

# PAGE 1: HOME

if page == "🏠 Home":
    st.markdown('<div class="main-header">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML-Powered Personalized Movie Suggestions</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Personalized</h3>
            <p>Get movie recommendations tailored to your taste using collaborative filtering</p>
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
            <h3>🎥 Movie Search</h3>
            <p>Search any movie and get similar recommendations - perfect for new users!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Overview
    st.markdown("### 📚 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### MovieLens 100K Dataset")
        st.markdown(f"""
        - **{ratings['user_id'].nunique()}** unique users
        - **{movies['movie_id'].nunique()}** movies
        - **{len(ratings):,}** ratings
        - Rating scale: **1-5 stars**
        - Genres: **19** different categories
        """)
    
    with col2:
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
    st.info("👈 Use the sidebar to navigate and explore the recommendation system!")


# PAGE 2: GET RECOMMENDATIONS


elif page == "🎬 Get Recommendations":
    st.markdown("## 🎬 Get Your Movie Recommendations")
    
    rec_mode = st.radio(
        "Choose Recommendation Mode:",
        ["👤 User-Based Recommendations", "🎥 Movie-Based Recommendations"],
        horizontal=True
    )
    
    st.markdown("---")
    

    # MODE 1: USER-BASED RECOMMENDATIONS
   
    if rec_mode == "👤 User-Based Recommendations":
        st.markdown("### Select a user to see personalized recommendations")
        
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
            
            show_posters = st.checkbox("Show Movie Posters", value=True)
            
            if st.button("🎯 Get Recommendations", type="primary"):
                st.session_state.show_user_recs = True
                st.session_state.user_id = user_id
                st.session_state.num_recs = num_recommendations
                st.session_state.show_posters = show_posters
        
        if 'show_user_recs' in st.session_state and st.session_state.show_user_recs:
            st.markdown("---")
            
            # Show user's rated movies
            st.markdown(f"### 🎬 User {st.session_state.user_id}'s Top Rated Movies")
            user_rated = get_user_rated_movies(st.session_state.user_id, 5)
            
            if len(user_rated) > 0:
                cols = st.columns(5)
                for idx, (_, movie_row) in enumerate(user_rated.iterrows()):
                    movie_info = get_movie_info(movie_row['item_id'])
                    if movie_info is not None:
                        with cols[idx]:
                            if st.session_state.show_posters:
                                st.image(get_movie_poster(movie_info['title']), use_container_width=True)
                            st.markdown(f"**{movie_info['title'][:30]}**")
                            st.caption(f"⭐ {movie_row['rating']:.1f}/5")
            
            st.markdown("---")
            st.markdown(f"### 🎯 Recommended for User {st.session_state.user_id}")
            
            with st.spinner("Generating recommendations..."):
                recommended_ids = item_based_recommendations(st.session_state.user_id, st.session_state.num_recs)
            
            if st.session_state.show_posters:
                cols_per_row = 5
                for i in range(0, len(recommended_ids), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, movie_id in enumerate(recommended_ids[i:i+cols_per_row]):
                        movie_info = get_movie_info(movie_id)
                        if movie_info is not None:
                            with cols[j]:
                                st.image(get_movie_poster(movie_info['title']), use_container_width=True)
                                st.markdown(f"**{movie_info['title']}**")
                                
                                genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 
                                             'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                                             'Sci-Fi', 'Thriller', 'War', 'Western']
                                genres = [g for g in genre_cols if movie_info[g] == 1]
                                genre_str = ", ".join(genres[:2]) if genres else "Unknown"
                                
                                movie_ratings = ratings[ratings['item_id'] == movie_id]
                                avg_rating = movie_ratings['rating'].mean() if len(movie_ratings) > 0 else 0
                                
                                st.caption(f"🎭 {genre_str}")
                                st.caption(f"⭐ {avg_rating:.1f}/5")
            else:
                for i, movie_id in enumerate(recommended_ids, 1):
                    movie_info = get_movie_info(movie_id)
                    if movie_info is not None:
                        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 
                                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                                     'Sci-Fi', 'Thriller', 'War', 'Western']
                        genres = [g for g in genre_cols if movie_info[g] == 1]
                        genre_str = ", ".join(genres[:3]) if genres else "Unknown"
                        
                        movie_ratings = ratings[ratings['item_id'] == movie_id]
                        avg_rating = movie_ratings['rating'].mean() if len(movie_ratings) > 0 else 0
                        num_ratings = len(movie_ratings)
                        
                        col1, col2, col3 = st.columns([0.5, 3, 1])
                        with col1:
                            st.markdown(f"### {i}")
                        with col2:
                            st.markdown(f"**{movie_info['title']}**")
                            st.caption(f"🎭 {genre_str}")
                        with col3:
                            st.metric("Rating", f"{avg_rating:.1f}⭐", f"{num_ratings} votes")
                        st.markdown("---")
    
    
    # MODE 2: MOVIE-BASED RECOMMENDATIONS
    
    else:
        st.markdown("### 🎬 Search for a movie and get similar recommendations!")
        st.info("👋 New user? Just search for any movie you like!")
        
        search_query = st.text_input(
            "🔍 Search for a movie",
            placeholder="e.g., Star Wars, Toy Story, The Matrix...",
            key="movie_search"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            num_similar = st.slider(
                "Number of Recommendations",
                min_value=5,
                max_value=20,
                value=12
            )
        with col2:
            show_posters_movie = st.checkbox("Show Movie Posters", value=True)
        
        if search_query:
            search_results = search_movies(search_query)
            
            if len(search_results) == 0:
                st.warning("❌ No movies found. Try a different search term.")
            else:
                st.markdown("---")
                st.success(f"✅ Found {len(search_results)} movie(s)")
                
                st.markdown("### 🎯 Select Your Movie")
                
                if show_posters_movie and len(search_results) <= 10:
                    cols_per_row = 5
                    for i in range(0, min(len(search_results), 10), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, (idx, movie) in enumerate(list(search_results.iterrows())[i:i+cols_per_row]):
                            with cols[j]:
                                st.image(get_movie_poster(movie['title']), use_container_width=True)
                                if st.button(f"Select", key=f"sel_{movie['movie_id']}", use_container_width=True):
                                    st.session_state.selected_movie_id = movie['movie_id']
                                    st.session_state.selected_movie_title = movie['title']
                                st.caption(movie['title'][:40])
                else:
                    movie_options = {f"{row['title']}": row['movie_id'] 
                                   for _, row in search_results.iterrows()}
                    selected_movie_display = st.selectbox(
                        "Choose a movie:",
                        options=list(movie_options.keys())
                    )
                    
                    if st.button("🎬 Select This Movie", type="primary"):
                        st.session_state.selected_movie_id = movie_options[selected_movie_display]
                        selected_movie = search_results[search_results['movie_id'] == st.session_state.selected_movie_id].iloc[0]
                        st.session_state.selected_movie_title = selected_movie['title']
                
                if 'selected_movie_id' in st.session_state:
                    st.markdown("---")
                    selected_movie = movies[movies['movie_id'] == st.session_state.selected_movie_id].iloc[0]
                    
                    st.markdown("### 🎬 You Selected:")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if show_posters_movie:
                            st.image(get_movie_poster(selected_movie['title']), use_container_width=True)
                    
                    with col2:
                        st.markdown(f"## {selected_movie['title']}")
                        
                        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 
                                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                                     'Sci-Fi', 'Thriller', 'War', 'Western']
                        genres = [g for g in genre_cols if selected_movie[g] == 1]
                        if genres:
                            st.markdown(f"**🎭 Genres:** {', '.join(genres)}")
                        
                        movie_ratings = ratings[ratings['item_id'] == st.session_state.selected_movie_id]
                        if len(movie_ratings) > 0:
                            avg_rating = movie_ratings['rating'].mean()
                            num_ratings = len(movie_ratings)
                            st.markdown(f"**⭐ Rating:** {avg_rating:.2f}/5.0 ({num_ratings} votes)")
                    
                    st.markdown("---")
                    st.markdown(f"### 🎯 Because you like **{selected_movie['title']}**:")
                    
                    with st.spinner("Finding similar movies..."):
                        similar_movie_ids = get_similar_movies(st.session_state.selected_movie_id, num_similar)
                    
                    if len(similar_movie_ids) == 0:
                        st.warning("Sorry, couldn't find similar movies.")
                    else:
                        if show_posters_movie:
                            cols_per_row = 4
                            for i in range(0, len(similar_movie_ids), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, movie_id in enumerate(similar_movie_ids[i:i+cols_per_row]):
                                    movie_info = get_movie_info(movie_id)
                                    if movie_info is not None:
                                        with cols[j]:
                                            st.image(get_movie_poster(movie_info['title']), use_container_width=True)
                                            st.markdown(f"**{movie_info['title']}**")
                                            
                                            genres = [g for g in genre_cols if movie_info[g] == 1]
                                            genre_str = ", ".join(genres[:2]) if genres else "Unknown"
                                            st.caption(f"🎭 {genre_str}")
                                            
                                            movie_ratings = ratings[ratings['item_id'] == movie_id]
                                            if len(movie_ratings) > 0:
                                                avg_rating = movie_ratings['rating'].mean()
                                                st.caption(f"⭐ {avg_rating:.1f}/5.0")
                                            
                                            similarity = item_similarity_df.loc[st.session_state.selected_movie_id, movie_id]
                                            st.progress(similarity, text=f"{similarity:.0%} match")
                        else:
                            for rank, movie_id in enumerate(similar_movie_ids, 1):
                                movie_info = get_movie_info(movie_id)
                                if movie_info is not None:
                                    col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1])
                                    
                                    with col1:
                                        st.markdown(f"### {rank}")
                                    with col2:
                                        st.markdown(f"**{movie_info['title']}**")
                                        genres = [g for g in genre_cols if movie_info[g] == 1]
                                        genre_str = ", ".join(genres[:3]) if genres else "Unknown"
                                        st.caption(f"🎭 {genre_str}")
                                    with col3:
                                        movie_ratings = ratings[ratings['item_id'] == movie_id]
                                        if len(movie_ratings) > 0:
                                            avg_rating = movie_ratings['rating'].mean()
                                            st.metric("Rating", f"{avg_rating:.1f}⭐")
                                    with col4:
                                        similarity = item_similarity_df.loc[st.session_state.selected_movie_id, movie_id]
                                        st.metric("Match", f"{similarity:.0%}")
                                    st.markdown("---")
                        
                        if st.button("🔄 Try Another Movie"):
                            del st.session_state.selected_movie_id
                            del st.session_state.selected_movie_title
                            st.rerun()

# PAGE 3: DATA INSIGHTS


elif page == "📊 Data Insights":
    st.markdown("## 📊 Data Insights & Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["📈 Ratings", "🎭 Genres", "👥 Users"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            rating_counts = ratings['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                title='Rating Distribution',
                text=rating_counts.values,
                color=rating_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            movie_stats = ratings.groupby('item_id').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            movie_stats.columns = ['movie_id', 'rating_count', 'avg_rating']
            
            fig = px.scatter(
                movie_stats,
                x='rating_count',
                y='avg_rating',
                title='Popularity vs Rating',
                labels={'rating_count': 'Number of Ratings', 'avg_rating': 'Avg Rating'},
                opacity=0.6,
                color='avg_rating',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏆 Top Rated Movies (min 50 ratings)")
        movie_stats_merged = movie_stats.merge(movies[['movie_id', 'title']], on='movie_id')
        top_movies = movie_stats_merged[movie_stats_merged['rating_count'] >= 50].nlargest(10, 'avg_rating')
        
        fig = px.bar(
            top_movies,
            x='avg_rating',
            y='title',
            orientation='h',
            title='Top 10 Movies',
            text='avg_rating',
            color='avg_rating',
            color_continuous_scale='RdYlGn'
        )
        fig.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
        fig.update_layout(showlegend=False, height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
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
                title='Movies per Genre',
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
        user_activity = ratings.groupby('user_id').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                x=user_activity.values,
                nbins=50,
                title='User Activity Distribution',
                labels={'x': 'Ratings per User', 'y': 'Number of Users'},
                color_discrete_sequence=['#636EFA']
            )
            fig.add_vline(x=user_activity.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {user_activity.mean():.1f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### User Statistics")
            st.metric("Avg ratings/user", f"{user_activity.mean():.1f}")
            st.metric("Median ratings/user", f"{user_activity.median():.0f}")
            st.metric("Most active user", f"{user_activity.max()} ratings")
            st.metric("Least active user", f"{user_activity.min()} ratings")


# PAGE 4: MODEL PERFORMANCE


elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance Comparison")
    
    st.markdown("""
    **4 Recommendation Algorithms Compared:**
    1. **User-Based CF** - Similar users
    2. **Item-Based CF** - Similar movies
    3. **SVD (Matrix Factorization)** - Latent factors
    4. **Hybrid** - Combined approach
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Results")
    
    styled_df = comparison_df.style.highlight_max(
        subset=['Precision@10', 'Coverage'],
        color='lightgreen'
    ).format({
        'Precision@10': '{:.4f}',
        'Coverage': '{:.4f}',
        'Training Time (s)': '{:.2f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Precision@10',
            title='Precision@10',
            text='Precision@10',
            color='Precision@10',
            color_continuous_scale='Greens'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Coverage',
            title='Coverage',
            text='Coverage',
            color='Coverage',
            color_continuous_scale='Blues'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    best_idx = comparison_df['Precision@10'].idxmax()
    best_model = comparison_df.iloc[best_idx]['Model']
    st.success(f"🏆 **Best Model: {best_model}**")

