import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from recommender import recommend_by_movie, recommend_by_mood, recommend_hybrid

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# --------------------------------------------------
# Load resources ONCE
# --------------------------------------------------
@st.cache_resource
def load_resources():
    df = pd.read_csv("data/cleaned_movies_30k.csv")
    embeddings = np.load("data/movie_embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, model

df, embeddings, model = load_resources()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.write(
    "A content-based movie recommender using **sentence embeddings**, "
    "**cosine similarity**, and **IMDb-weighted ranking**."
)

tab1, tab2, tab3 = st.tabs(
    ["🎥 Movie-Based Recommendation", "😊 Mood-Based Recommendation", "🔀 Hybrid Recommendation"]
)

# --------------------------------------------------
# Movie-based recommendation
# --------------------------------------------------
with tab1:
    st.subheader("Find Similar Movies")

    selected_movie = st.selectbox(
        "Choose a movie",
        sorted(df["title"].unique())
    )

    top_n = st.slider("Number of recommendations", 5, 20, 5)
    alpha = st.slider(
        "Similarity vs IMDb weight (α)",
        0.0, 1.0, 0.7,
        help="Higher α → more similarity, lower α → higher IMDb influence"
    )

    if st.button("Recommend Movies"):
        results = recommend_by_movie(
            selected_movie,
            df,
            embeddings,
            top_n=top_n,
            alpha=alpha
        )

        st.success("Recommended Movies")
        st.dataframe(results, use_container_width=True)

# --------------------------------------------------
# Mood-based recommendation
# --------------------------------------------------
with tab2:
    st.subheader("Get Movies Based on Your Mood")

    mood = st.selectbox(
        "Select your mood",
        ["happy", "romantic", "sad", "thriller", "action"]
    )

    top_n = st.slider("Number of recommendations", 5, 20, 5, key="mood_top_n")
    alpha = st.slider(
        "Similarity vs IMDb weight (α)",
        0.0, 1.0, 0.7,
        key="mood_alpha"
    )

    if st.button("Recommend by Mood"):
        results = recommend_by_mood(
            mood,
            df,
            embeddings,
            model,
            top_n=top_n,
            alpha=alpha
        )

        st.success("Movies matching your mood")
        st.dataframe(results, use_container_width=True)

# --------------------------------------------------
# Hybrid Recommendation
# --------------------------------------------------
with tab3:
    st.subheader("Hybrid Movie Recommendation")
    st.write(
        "Combine **movie similarity**, **mood preference**, and **IMDb rating** "
        "to get balanced recommendations."
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_movie = st.selectbox(
            "Select a reference movie (optional)",
            ["None"] + sorted(df["title"].unique())
        )

        mood = st.selectbox(
            "Select your mood (optional)",
            ["None", "happy", "romantic", "sad", "thriller", "action"]
        )

        top_n = st.slider(
            "Number of recommendations",
            5, 20, 10,
            key="hybrid_top_n"
        )

    with col2:
        st.markdown(" Weight Configuration")

        w_movie = st.slider(
            "Movie Similarity Weight",
            0.0, 1.0, 0.4
        )

        w_mood = st.slider(
            "Mood Similarity Weight",
            0.0, 1.0, 0.4
        )

        w_imdb = st.slider(
            "IMDb Rating Weight",
            0.0, 1.0, 0.2
        )

        total = w_movie + w_mood + w_imdb
        if total != 1.0:
            st.warning(
                f"⚠️ Weights sum to {total:.2f}. "
                "They ideally should sum to 1."
            )

    if st.button("Get Hybrid Recommendations"):
        results = recommend_hybrid(
            df=df,
            embeddings=embeddings,
            model=model,
            movie=None if selected_movie == "None" else selected_movie,
            mood=None if mood == "None" else mood,
            top_n=top_n,
            w_movie=w_movie,
            w_mood=w_mood,
            w_imdb=w_imdb
        )

        st.success("Hybrid Recommendations")
        st.dataframe(results, use_container_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>
            👨‍💻 Built by <b>Your Name</b><br>
            🔗 
            <a href="https://github.com/YOUR_GITHUB_USERNAME" target="_blank">
                GitHub
            </a>
            &nbsp; | &nbsp;
            <a href="https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME" target="_blank">
                LinkedIn
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption(
    "Built with Python • NLP Embeddings • Machine Learning • Streamlit"
)
