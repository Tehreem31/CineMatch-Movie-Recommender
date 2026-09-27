import os
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Page Configuration
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Realistic CSS Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #14181c;
        color: #9ab;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .nav-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0px 20px 0px;
        border-bottom: 1px solid #2c3440;
        margin-bottom: 25px;
    }
    .logo {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .logo span {
        color: #00e054;
    }

    .movie-card-container {
        background: #1d232a;
        border: 1px solid #2c3440;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
    }
    .poster-img {
        width: 100%;
        height: 320px;
        object-fit: cover;
        border-bottom: 1px solid #2c3440;
    }
    .card-body {
        padding: 14px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    .movie-title {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 4px;
        line-height: 1.3;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-meta {
        font-size: 0.8rem;
        color: #678;
        margin-bottom: 10px;
    }
    .genre-pill {
        display: inline-block;
        background: #242c34;
        color: #9ab;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 3px;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid #2c3440;
    }
    .card-footer-stats {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #242c34;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
    }
    .rating-star {
        color: #00e054;
        font-weight: 600;
    }
    .hybrid-score {
        color: #40bcf4;
        font-weight: 500;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)


# Helper function to fetch live posters safely without exposing raw API keys
@st.cache_data(show_spinner=False)
def fetch_poster(movie_title):
  # Reads from environment variable or Streamlit secrets, falls back cleanly if empty
  api_key = os.getenv("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY", "")

  if not api_key or api_key == "YOUR_TMDB_API_KEY":
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop"

  clean_name = re.sub(r"\s*\(\d{4}\)", "", movie_title).strip()
  url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_name}"
  try:
    response = requests.get(url, timeout=3)
    data = response.json()
    if data.get("results") and len(data["results"]) > 0:
      poster_path = data["results"][0].get("poster_path")
      if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
  except Exception:
    pass

  return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop"


@st.cache_resource
def load_and_prep_data():
  # Relative Paths for GitHub compatibility
  movies_df = pd.read_csv("ml-latest-small/movies.csv")
  ratings_df = pd.read_csv("ml-latest-small/ratings.csv")

  movies_df.dropna(subset=["title", "genres"], inplace=True)
  movies_df.drop_duplicates(subset=["movieId"], inplace=True)

  collab_stats = (
      ratings_df.groupby("movieId")
      .agg(avg_rating=("rating", "mean"), vote_count=("rating", "count"))
      .reset_index()
  )

  movies_df = pd.merge(movies_df, collab_stats, on="movieId", how="left")
  movies_df["avg_rating"] = movies_df["avg_rating"].fillna(0.0)
  movies_df["vote_count"] = movies_df["vote_count"].fillna(0)
  movies_df["year"] = (
      movies_df["title"]
      .str.extract(r"\((\d{4})\)", expand=False)
      .fillna("N/A")
  )

  max_votes = movies_df["vote_count"].max()
  movies_df["norm_collab_score"] = 0.7 * (
      movies_df["avg_rating"] / 5.0
  ) + 0.3 * (movies_df["vote_count"] / max_votes)

  movies_df["genres_processed"] = (
      movies_df["genres"]
      .str.replace("|", " ", regex=False)
      .str.replace("(no genres listed)", "", regex=False)
  )
  movies_df["clean_title"] = movies_df["title"].apply(
      lambda x: re.sub(r"\s*\(\d{4}\)", "", x).strip()
  )

  tfidf = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[\w-]+\b")
  tfidf_matrix = tfidf.fit_transform(movies_df["genres_processed"])
  cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

  movies_df = movies_df.reset_index(drop=True)
  indices = pd.Series(
      movies_df.index, index=movies_df["clean_title"].str.lower()
  )
  indices = indices[~indices.index.duplicated(keep="first")]

  return movies_df, cosine_sim, indices


movies_df, cosine_sim, indices = load_and_prep_data()

# Navigation Header
st.markdown(
    """
    <div class="nav-header">
        <div class="logo">Cine<span>Match</span></div>
    </div>
""",
    unsafe_allow_html=True,
)

# Search & Controls
col_search, col_count = st.columns([3, 1])
movie_list = movies_df["clean_title"].sort_values().unique().tolist()

with col_search:
  selected_movie = st.selectbox("Search for a film you like:", movie_list)

with col_count:
  top_k = st.selectbox("Recommendations:", [3, 6, 9, 12], index=1)

# Preset Selection (Option 1)
st.write("**Recommendation Mode:**")
mode = st.radio(
    label="Algorithm Profile",
    options=["Balanced (Hybrid)", "Genre Focused", "Popular & Top Rated"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Genre Focused":
  content_weight = 0.85
elif mode == "Popular & Top Rated":
  content_weight = 0.15
else:
  content_weight = 0.50

collab_weight = round(1.0 - content_weight, 2)

st.markdown("---")

# Render Recommendations Grid
if selected_movie:
  key = selected_movie.strip().lower()
  if key in indices:
    idx = indices[key]
    if isinstance(idx, pd.Series):
      idx = idx.iloc[0]
    idx = int(idx)

    content_sim = np.array(cosine_sim[idx]).flatten()
    collab_score = movies_df["norm_collab_score"].values
    hybrid_scores = (content_weight * content_sim) + (
        collab_weight * collab_score
    )

    ranked_indices = sorted(
        list(enumerate(hybrid_scores)), key=lambda x: x[1], reverse=True
    )[1 : top_k + 1]

    rec_indices = [i[0] for i in ranked_indices]
    results = movies_df.iloc[rec_indices].copy()
    results["score"] = [round(float(i[1]), 2) for i in ranked_indices]

    st.markdown(f"### If you liked **{selected_movie}**, you might enjoy:")
    st.write("")

    cols = st.columns(3)
    for idx_pos, (_, row) in enumerate(results.iterrows()):
      col = cols[idx_pos % 3]
      poster_url = fetch_poster(row["clean_title"])
      genres_tags = "".join([
          f"<span class='genre-pill'>{g}</span>"
          for g in row["genres"].split("|")[:3]
      ])

      with col:
        st.markdown(
            f"""
                    <div class="movie-card-container">
                        <img src="{poster_url}" class="poster-img" alt="{row['clean_title']}">
                        <div class="card-body">
                            <div class="movie-title" title="{row['clean_title']}">{row['clean_title']}</div>
                            <div class="movie-meta">{row['year']}</div>
                            <div>{genres_tags}</div>
                            <div class="card-footer-stats">
                                <span class="rating-star">★ {row['avg_rating']:.1f} <span style="color:#678; font-size:0.75rem;">({int(row['vote_count'])})</span></span>
                                <span class="hybrid-score">Score: {row['score']}</span>
                            </div>
                        </div>
                    </div>
                """,
            unsafe_allow_html=True,
        )
