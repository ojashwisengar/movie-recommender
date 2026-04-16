import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ─── Path Setup ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ─── Load Data & Train Models ──────────────────────────────
@st.cache_resource
def load_models():
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    popular_movies = pd.read_csv(os.path.join(DATA_DIR, 'popular_movies.csv'))

    # Content Based — TF-IDF only, NO full cosine matrix
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    # Collaborative — SVD with smaller k to save memory
    user_item = ratings.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)

    user_ids = user_item.index.tolist()
    movie_ids = user_item.columns.tolist()

    sparse_matrix = csr_matrix(user_item.values, dtype=float)
    U, sigma, Vt = svds(sparse_matrix, k=20)  # reduced from 50 to 20
    sigma = np.diag(sigma)

    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_ids,
        columns=movie_ids
    )

    # Return tfidf_matrix instead of cosine_sim
    return movies, ratings, popular_movies, tfidf_matrix, indices, predicted_df, movie_ids

# Load everything — shows spinner while training
with st.spinner("🔄 Loading models... (first load takes ~30 seconds)"):
    movies, ratings, popular_movies, tfidf_matrix, indices, predicted_df, movie_ids = load_models()

# ─── Helper Functions ──────────────────────────────────────
def get_content_scores(title, top_n=50):
    if title not in indices:
        return None
    idx = indices[title]
    
    # Calculate similarity only for this one movie — saves memory!
    movie_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()
    
    # Get top matches
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
    similar_scores = sim_scores[similar_indices]
    
    result = movies.iloc[similar_indices][['movieId', 'title', 'genres']].copy()
    result['content_score'] = similar_scores
    return result

def get_collab_score(user_id, movie_id):
    if user_id not in predicted_df.index:
        return 3.5  # default average
    if movie_id not in predicted_df.columns:
        return 3.5
    return predicted_df.loc[user_id, movie_id]

def hybrid_recommendations(user_id=None, title=None, top_n=10):
    # Case 4: No input → Popularity
    if user_id is None and title is None:
        return popular_movies[['title', 'genres', 'weighted_score']].head(top_n)

    # Case 2: Only title → Content based
    if user_id is None and title is not None:
        content_recs = get_content_scores(title, top_n=top_n)
        if content_recs is None:
            return None
        content_recs = content_recs.merge(
            popular_movies[['movieId', 'weighted_score']],
            on='movieId', how='left'
        )
        content_recs['weighted_score'] = content_recs['weighted_score'].fillna(0)
        content_recs = content_recs.sort_values(
            ['content_score', 'weighted_score'], ascending=[False, False]
        )
        return content_recs[['title', 'genres', 'content_score']].head(top_n)

    # Case 3: Only user_id → Collaborative
    if user_id is not None and title is None:
        if user_id not in predicted_df.index:
            st.warning("User ID not found. Showing popular movies instead.")
            return popular_movies[['title', 'genres', 'weighted_score']].head(top_n)

        rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
        user_preds = predicted_df.loc[user_id]
        user_preds = user_preds[~user_preds.index.isin(rated_movies)]
        top_movies = user_preds.sort_values(ascending=False).head(top_n)
        result = pd.DataFrame({'movieId': top_movies.index, 'predicted_rating': top_movies.values})
        result = result.merge(movies, on='movieId')
        return result[['title', 'genres', 'predicted_rating']].head(top_n)

    # Case 1: Both → Full Hybrid
    content_recs = get_content_scores(title, top_n=50)
    if content_recs is None:
        return None

    collab_scores = []
    for _, row in content_recs.iterrows():
        score = get_collab_score(user_id, row['movieId'])
        collab_scores.append(round(score, 3))

    content_recs['collab_score'] = collab_scores

    content_recs['content_norm'] = (
        content_recs['content_score'] - content_recs['content_score'].min()
    ) / (content_recs['content_score'].max() - content_recs['content_score'].min() + 1e-9)

    content_recs['collab_norm'] = (
        content_recs['collab_score'] - content_recs['collab_score'].min()
    ) / (content_recs['collab_score'].max() - content_recs['collab_score'].min() + 1e-9)

    content_recs['hybrid_score'] = (
        0.4 * content_recs['content_norm'] +
        0.6 * content_recs['collab_norm']
    )

    content_recs = content_recs.sort_values('hybrid_score', ascending=False)
    return content_recs[['title', 'genres', 'content_score',
                          'collab_score', 'hybrid_score']].head(top_n)

# ─── UI ────────────────────────────────────────────────────
st.title("🎬 Movie Recommender System")
st.markdown("*Hybrid recommender using Content-Based + Collaborative Filtering*")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("- 🎯 **Movie only** → Content-based recommendations")
st.sidebar.markdown("- 👤 **User only** → Collaborative filtering")
st.sidebar.markdown("- 🔥 **Both** → Full hybrid recommendations")
st.sidebar.markdown("- ⭐ **Neither** → Most popular movies")
st.sidebar.markdown("---")
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# Main inputs
col1, col2 = st.columns(2)
with col1:
    movie_list = ["None"] + sorted(movies['title'].tolist())
    selected_movie = st.selectbox("🎬 Select a Movie (optional)", movie_list)
with col2:
    user_id_input = st.text_input("👤 Enter User ID (1-610, optional)", value="")

if st.button("🚀 Get Recommendations", type="primary"):
    title = None if selected_movie == "None" else selected_movie
    user_id = None
    if user_id_input.strip() != "":
        try:
            user_id = int(user_id_input.strip())
            if user_id < 1 or user_id > 610:
                st.error("User ID must be between 1 and 610")
                st.stop()
        except ValueError:
            st.error("Please enter a valid number for User ID")
            st.stop()

    if user_id and title:
        st.success(f"🔥 Full Hybrid Mode — User {user_id} + {title}")
    elif title:
        st.info(f"🎯 Content-Based Mode — Similar to {title}")
    elif user_id:
        st.info(f"👤 Collaborative Mode — Personalized for User {user_id}")
    else:
        st.info("⭐ Showing Most Popular Movies")

    with st.spinner("Finding recommendations..."):
        results = hybrid_recommendations(user_id=user_id, title=title, top_n=top_n)

    if results is None:
        st.error("Movie not found. Please try another title.")
    else:
        st.markdown(f"### 🎯 Top {top_n} Recommendations")
        for i, (_, row) in enumerate(results.iterrows()):
            with st.container():
                col_num, col_title, col_genre, col_score = st.columns([0.5, 3, 3, 1.5])
                with col_num:
                    st.markdown(f"**#{i+1}**")
                with col_title:
                    st.markdown(f"**{row['title']}**")
                with col_genre:
                    genres = row['genres'].replace('|', ' · ')
                    st.caption(genres)
                with col_score:
                    if 'hybrid_score' in row:
                        st.metric("Hybrid", f"{row['hybrid_score']:.2f}")
                    elif 'predicted_rating' in row:
                        st.metric("Predicted", f"{row['predicted_rating']:.2f}")
                    elif 'content_score' in row:
                        st.metric("Similarity", f"{row['content_score']:.2f}")
                    elif 'weighted_score' in row:
                        st.metric("Score", f"{row['weighted_score']:.2f}")
                st.markdown("---")

st.markdown("---")

