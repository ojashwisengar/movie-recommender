import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import os

# Fix path issue — works from any directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ─── Load Data & Models ────────────────────────────────────
@st.cache_resource
def load_models():
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    popular_movies = pd.read_csv(os.path.join(DATA_DIR, 'popular_movies.csv'))

    with open(os.path.join(DATA_DIR, 'svd_model.pkl'), 'rb') as f:
        svd_model = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'cosine_sim.pkl'), 'rb') as f:
        cosine_sim = pickle.load(f)

    indices = pd.read_csv(os.path.join(DATA_DIR, 'movie_indices.csv'), 
                          index_col=0).squeeze()

    return movies, ratings, popular_movies, svd_model, cosine_sim, indices

movies, ratings, popular_movies, svd_model, cosine_sim, indices = load_models()

# ─── Helper Functions ──────────────────────────────────────
def get_content_scores(title, top_n=50):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    result = movies.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
    result['content_score'] = similarity_scores
    return result

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
        rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
        all_movie_ids = movies['movieId'].tolist()
        unseen = [m for m in all_movie_ids if m not in rated_movies]
        preds = []
        for movie_id in unseen:
            pred = svd_model.predict(user_id, movie_id)
            preds.append({'movieId': movie_id, 'predicted_rating': round(pred.est, 3)})
        pred_df = pd.DataFrame(preds).sort_values('predicted_rating', ascending=False)
        result = pred_df.merge(movies, on='movieId')
        return result[['title', 'genres', 'predicted_rating']].head(top_n)

    # Case 1: Both → Full Hybrid
    content_recs = get_content_scores(title, top_n=50)
    if content_recs is None:
        return None

    collab_scores = []
    for _, row in content_recs.iterrows():
        pred = svd_model.predict(user_id, row['movieId'])
        collab_scores.append(round(pred.est, 3))

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

# Get recommendations button
if st.button("🚀 Get Recommendations", type="primary"):

    # Parse inputs
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

    # Show which mode we're in
    if user_id and title:
        st.success(f"🔥 Full Hybrid Mode — User {user_id} + {title}")
    elif title:
        st.info(f"🎯 Content-Based Mode — Similar to {title}")
    elif user_id:
        st.info(f"👤 Collaborative Mode — Personalized for User {user_id}")
    else:
        st.info("⭐ Showing Most Popular Movies")

    # Get recommendations
    with st.spinner("Finding recommendations..."):
        results = hybrid_recommendations(user_id=user_id, title=title, top_n=top_n)

    if results is None:
        st.error("Movie not found. Please try another title.")
    else:
        st.markdown(f"### 🎯 Top {top_n} Recommendations")

        # Display as cards
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
                    # Show whichever score column exists
                    if 'hybrid_score' in row:
                        st.metric("Hybrid", f"{row['hybrid_score']:.2f}")
                    elif 'predicted_rating' in row:
                        st.metric("Predicted", f"{row['predicted_rating']:.2f}")
                    elif 'content_score' in row:
                        st.metric("Similarity", f"{row['content_score']:.2f}")
                    elif 'weighted_score' in row:
                        st.metric("Score", f"{row['weighted_score']:.2f}")

                st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using MovieLens dataset | "
    "Models: SVD Collaborative Filtering + TF-IDF Content-Based"
)