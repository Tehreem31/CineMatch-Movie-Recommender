# CineMatch: Hybrid Movie Recommendation System

CineMatch is an interactive web application that provides personalized movie recommendations by blending content-based filtering (TF-IDF vectorization on movie genres) with collaborative filtering signals (normalized user ratings and vote counts).

---

##  Key Features

- **Hybrid Engine Logic:** Combines content similarity matrix calculations with historical popularity and user ratings.
- **Dynamic Algorithm Presets:** Toggle instantly between three tailored modes:
  - **Balanced (Hybrid):** 50% Content / 50% Collaborative Weighting.
  - **Genre Focused:** 85% Content / 15% Collaborative Weighting.
  - **Popular & Top Rated:** 15% Content / 85% Collaborative Weighting.
- **Dynamic UI & Posters:** Built with Streamlit in a responsive dark grid theme, featuring dynamic poster retrieval via TMDB API integration and graceful fallback handling.

---

## 📸 Demonstration Interface

![CineMatch App Interface](app_demo.png)

---

