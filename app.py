import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import re
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
        background-color: #0A1172;
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
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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

@st.cache_data
def load_content_based_db():
    """Load pre-built content recommendation database"""
    try:
        with open('tmdb_content_recommendations.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

models, comparison_df = load_data()

# Extract components
item_similarity_df = models['item_similarity']
svd_model = models['svd_model']
train_matrix = models['train_matrix']
movies = models['movies']
ratings = models['ratings']


# HELPER FUNCTIONS

@st.cache_data
def get_movie_poster(title):
    """Get movie poster from TMDb API with caching"""
    try:
        TMDB_API_KEY = "7891144d4b5142e348389f3caeef27f3"
        url = "https://api.themoviedb.org/3/search/movie"
        
        # Extract year from title
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
                return f"https://image.tmdb.org/t/p/w300{poster_path}"  # w300 for smaller size
        
        # Fallback to placeholder
        safe_title = clean_title.replace(' ', '+')[:15]
        return f"https://via.placeholder.com/300x450/667eea/ffffff?text={safe_title}"
        
    except:
        safe_title = title.replace(' ', '+').replace('(', '').replace(')', '')[:15]
        return f"https://via.placeholder.com/300x450/667eea/ffffff?text={safe_title}"

def get_tmdb_poster(poster_path):
    """Get full poster URL from TMDb"""
    if poster_path:
        return f"https://image.tmdb.org/t/p/w300{poster_path}"  # w300 for smaller size
    return None

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

@st.cache_data(ttl=3600)
def search_tmdb_movies(query):
    """Search ANY movie on TMDb"""
    try:
        TMDB_API_KEY = "7891144d4b5142e348389f3caeef27f3"
        url = "https://api.themoviedb.org/3/search/movie"
        
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'include_adult': False,
            'page': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get('results'):
            tmdb_movies = []
            for movie in data['results'][:20]:
                tmdb_movies.append({
                    'tmdb_id': movie['id'],
                    'title': f"{movie['title']} ({movie.get('release_date', '')[:4]})" if movie.get('release_date') else movie['title'],
                    'original_title': movie['title'],
                    'poster_path': movie.get('poster_path'),
                    'overview': movie.get('overview', 'No overview available'),
                    'vote_average': movie.get('vote_average', 0),
                    'release_date': movie.get('release_date', ''),
                    'genre_ids': movie.get('genre_ids', [])
                })
            return pd.DataFrame(tmdb_movies)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def search_movies(query):
    """Search movies in dataset AND TMDb API"""
    query_lower = query.lower()
    local_results = movies[movies['title'].str.lower().str.contains(query_lower, na=False)]
    tmdb_results = search_tmdb_movies(query)
    return local_results, tmdb_results

def get_similar_movies(movie_id, n=10):
    """Get similar movies based on item similarity"""
    if movie_id not in item_similarity_df.index:
        return []
    
    similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]
    return similar_movies.index.tolist()

def get_content_based_recommendations(movie_title, n=10):
    """Get recommendations using content-based filtering"""
    db = load_content_based_db()
    
    if db is None:
        return []
    
    movies_df = db['movies']
    similarity_matrix = db['similarity_matrix']
    
    matches = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if len(matches) == 0:
        return []
    
    idx = matches.index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df.iloc[movie_indices].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]
    
    return recommendations

@st.cache_data
def get_tmdb_movie_details(tmdb_id):
    """Get detailed movie info from TMDb"""
    try:
        TMDB_API_KEY = "7891144d4b5142e348389f3caeef27f3"
        
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {'api_key': TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=5)
        movie_details = response.json()
        
        similar_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/recommendations"
        similar_response = requests.get(similar_url, params=params, timeout=5)
        similar_movies = similar_response.json().get('results', [])
        
        return movie_details, similar_movies
    except:
        return None, []

def get_genre_name(genre_ids):
    """Convert TMDb genre IDs to names"""
    genre_map = {
        28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
        80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
        14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
        9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 10770: "TV Movie",
        53: "Thriller", 10752: "War", 37: "Western"
    }
    return [genre_map.get(gid, '') for gid in genre_ids if gid in genre_map]

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

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", [
    " Home",
    " Get Recommendations",
    " Data Insights",
    " Model Performance",
    
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.metric("Total Movies", f"{movies['movie_id'].nunique():,}")
st.sidebar.metric("Total Users", f"{ratings['user_id'].nunique():,}")
st.sidebar.metric("Total Ratings", f"{len(ratings):,}")
st.sidebar.metric("Avg Rating", f"{ratings['rating'].mean():.2f}/5.0")


# PAGE 1: HOME


if page == " Home":
    st.markdown('<div class="main-header">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML-Powered Personalized Movie Suggestions</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
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
            <h3>🎥 Unlimited Movies</h3>
            <p>Search ANY movie from TMDb database and get instant recommendations!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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


elif page == " Get Recommendations":
    st.markdown("## 🎬 Get Your Movie Recommendations")
    
    rec_mode = st.radio(
        "Choose Recommendation Mode:",
        [" User-Based Recommendations", " Movie-Based Recommendations"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # MODE 1: USER-BASED RECOMMENDATIONS
   
    if rec_mode == " User-Based Recommendations":
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
            
            if st.button(" Get Recommendations", type="primary"):
                st.session_state.show_user_recs = True
                st.session_state.user_id = user_id
                st.session_state.num_recs = num_recommendations
                st.session_state.show_posters = show_posters
        
        if 'show_user_recs' in st.session_state and st.session_state.show_user_recs:
            st.markdown("---")
            
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
            
            genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 
                         'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                         'Sci-Fi', 'Thriller', 'War', 'Western']
            
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
        st.markdown("### 🎬 Search for ANY movie and get recommendations!")
        st.info("💡 Search for any movie - even new releases! We'll find similar movies for you.")
        
        search_query = st.text_input(
            "🔍 Search for any movie",
            placeholder="e.g., Avatar, Inception, Barbie, Oppenheimer...",
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
        
        if search_query and len(search_query) > 2:
            with st.spinner("🔍 Searching for movies..."):
                local_results, tmdb_results = search_movies(search_query)
            
            total_results = len(local_results) + len(tmdb_results)
            
            if total_results == 0:
                st.warning("❌ No movies found. Try a different search term.")
                st.info("💡 Tip: Try popular movies like 'Star Wars', 'Marvel', 'Harry Potter'")
            else:
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ Found {total_results} movie(s)")
                with col2:
                    if len(local_results) > 0:
                        st.info(f"📀 {len(local_results)} in our database")
                    if len(tmdb_results) > 0:
                        st.info(f"🌐 {len(tmdb_results)} from TMDb")
                
                st.markdown("### 🎯 Select a Movie")
                
                if len(local_results) > 0 and len(tmdb_results) > 0:
                    tab1, tab2 = st.tabs([f"📀 Our Database ({len(local_results)})", f"🌐 All Movies ({len(tmdb_results)})"])
                elif len(local_results) > 0:
                    tab1 = st.container()
                    tab2 = None
                else:
                    tab1 = None
                    tab2 = st.container()
                
                # LOCAL DATABASE RESULTS
                if len(local_results) > 0 and tab1 is not None:
                    with tab1:
                        st.markdown("**Movies in our trained database (best recommendations)**")
                        
                        if show_posters_movie and len(local_results) <= 10:
                            cols_per_row = 5
                            for i in range(0, len(local_results), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, (idx, movie) in enumerate(list(local_results.iterrows())[i:i+cols_per_row]):
                                    with cols[j]:
                                        st.image(get_movie_poster(movie['title']), use_container_width=True)
                                        if st.button("Select", key=f"local_{movie['movie_id']}", use_container_width=True):
                                            st.session_state.selected_movie_source = 'local'
                                            st.session_state.selected_movie_id = movie['movie_id']
                                            st.session_state.selected_movie_title = movie['title']
                                        st.caption(movie['title'][:40])
                        else:
                            movie_options = {f"{row['title']}": row['movie_id'] 
                                           for _, row in local_results.iterrows()}
                            selected = st.selectbox("Choose:", list(movie_options.keys()), key="local_select")
                            
                            if st.button("🎬 Select This Movie", type="primary", key="local_btn"):
                                st.session_state.selected_movie_source = 'local'
                                st.session_state.selected_movie_id = movie_options[selected]
                                st.session_state.selected_movie_title = selected
                
                # TMDB RESULTS
                if len(tmdb_results) > 0 and tab2 is not None:
                    with tab2:
                        st.markdown("**Search any movie from TMDb database**")
                        
                        if show_posters_movie and len(tmdb_results) <= 10:
                            cols_per_row = 5
                            for i in range(0, len(tmdb_results), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, (idx, movie) in enumerate(list(tmdb_results.iterrows())[i:i+cols_per_row]):
                                    with cols[j]:
                                        poster_url = get_tmdb_poster(movie['poster_path'])
                                        if poster_url:
                                            st.image(poster_url, use_container_width=True)
                                        else:
                                            st.image(get_movie_poster(movie['title']), use_container_width=True)
                                        
                                        if st.button("Select", key=f"tmdb_{movie['tmdb_id']}", use_container_width=True):
                                            st.session_state.selected_movie_source = 'tmdb'
                                            st.session_state.selected_tmdb_id = movie['tmdb_id']
                                            st.session_state.selected_movie_title = movie['title']
                                        st.caption(movie['title'][:40])
                        else:
                            movie_options = {f"{row['title']} ⭐{row['vote_average']:.1f}": row['tmdb_id'] 
                                           for _, row in tmdb_results.iterrows()}
                            selected = st.selectbox("Choose:", list(movie_options.keys()), key="tmdb_select")
                            
                            if st.button("🎬 Select This Movie", type="primary", key="tmdb_btn"):
                                st.session_state.selected_movie_source = 'tmdb'
                                st.session_state.selected_tmdb_id = movie_options[selected]
                                st.session_state.selected_movie_title = selected.split(' ⭐')[0]
                
                # SHOW RECOMMENDATIONS
                if 'selected_movie_source' in st.session_state:
                    st.markdown("---")
                    
                    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 
                                 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                                 'Sci-Fi', 'Thriller', 'War', 'Western']
                    
                    # LOCAL MOVIE RECOMMENDATIONS
                    if st.session_state.selected_movie_source == 'local':
                        selected_movie = movies[movies['movie_id'] == st.session_state.selected_movie_id].iloc[0]
                        
                        st.markdown("### 🎬 You Selected:")
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            if show_posters_movie:
                                st.image(get_movie_poster(selected_movie['title']), use_container_width=True)
                        
                        with col2:
                            st.markdown(f"## {selected_movie['title']}")
                            
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
                            st.warning("Sorry, couldn't find similar movies in our database.")
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
                    
                    # TMDB MOVIE RECOMMENDATIONS
                    else:
                        st.markdown("### 🎬 You Selected:")
                        
                        with st.spinner("Loading movie details..."):
                            movie_details, similar_movies = get_tmdb_movie_details(st.session_state.selected_tmdb_id)
                        
                        if movie_details:
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                if show_posters_movie:
                                    poster_url = get_tmdb_poster(movie_details.get('poster_path'))
                                    if poster_url:
                                        st.image(poster_url, use_container_width=True)
                            
                            with col2:
                                st.markdown(f"## {movie_details.get('title')}")
                                
                                if movie_details.get('genres'):
                                    genre_names = [g['name'] for g in movie_details['genres']]
                                    st.markdown(f"**🎭 Genres:** {', '.join(genre_names)}")
                                
                                if movie_details.get('vote_average'):
                                    st.markdown(f"**⭐ TMDb Rating:** {movie_details['vote_average']:.1f}/10 ({movie_details.get('vote_count', 0):,} votes)")
                                
                                if movie_details.get('overview'):
                                    with st.expander("📖 Overview"):
                                        st.write(movie_details['overview'])
                            
                            st.markdown("---")
                            st.markdown(f"### 🎯 Similar Movies:")
                            
                            if len(similar_movies) == 0:
                                st.warning("No similar movies found.")
                            else:
                                similar_movies = similar_movies[:num_similar]
                                
                                if show_posters_movie:
                                    cols_per_row = 4
                                    for i in range(0, len(similar_movies), cols_per_row):
                                        cols = st.columns(cols_per_row)
                                        for j, movie in enumerate(similar_movies[i:i+cols_per_row]):
                                            with cols[j]:
                                                poster_url = get_tmdb_poster(movie.get('poster_path'))
                                                if poster_url:
                                                    st.image(poster_url, use_container_width=True)
                                                else:
                                                    st.image(get_movie_poster(movie['title']), use_container_width=True)
                                                
                                                st.markdown(f"**{movie['title']}**")
                                                
                                                genres = get_genre_name(movie.get('genre_ids', []))
                                                if genres:
                                                    st.caption(f"🎭 {', '.join(genres[:2])}")
                                                
                                                if movie.get('vote_average'):
                                                    st.caption(f"⭐ {movie['vote_average']:.1f}/10")
                                                
                                                if movie.get('release_date'):
                                                    st.caption(f"📅 {movie['release_date'][:4]}")
                                else:
                                    for rank, movie in enumerate(similar_movies, 1):
                                        col1, col2, col3 = st.columns([0.5, 3, 1])
                                        
                                        with col1:
                                            st.markdown(f"### {rank}")
                                        with col2:
                                            st.markdown(f"**{movie['title']}**")
                                            genres = get_genre_name(movie.get('genre_ids', []))
                                            if genres:
                                                st.caption(f"🎭 {', '.join(genres)}")
                                            if movie.get('release_date'):
                                                st.caption(f"📅 {movie['release_date'][:4]}")
                                        with col3:
                                            if movie.get('vote_average'):
                                                st.metric("Rating", f"{movie['vote_average']:.1f}/10")
                                        st.markdown("---")
                        
                        if st.button("🔄 Try Another Movie"):
                            for key in ['selected_movie_source', 'selected_movie_id', 'selected_tmdb_id', 'selected_movie_title']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()


# PAGE 3: DATA INSIGHTS


elif page == " Data Insights":
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


elif page == " Model Performance":
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


