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
    alpha_percent = st.slider(
        "Similarity vs IMDb weight (α)",
        0, 100, 70, step=5,
        help="Higher α → more similarity, lower α → higher IMDb influence"
    )
    alpha = alpha_percent / 100.0
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
        ["Happy", "Romantic", "Sad", "Thriller", "Action"]
    )

    top_n = st.slider("Number of recommendations", 5, 20, 5, key="mood_top_n")
    alpha_percent = st.slider(
        "Similarity vs IMDb weight (α)",
        0, 100, 70, step=5,
        key="mood_alpha",
        help="Higher α → more mood-friendly, lower α → higher IMDb influence"
    )
    alpha = alpha_percent / 100.0

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
            0, 100, 40,
            step=5,
            help="Influence of the selected reference movie"
        )

        w_mood = st.slider(
            "Mood Similarity Weight",
            0, 100, 40,
            step=5,
            help="How strongly your mood affects recommendations"
        )

        w_imdb = st.slider(
            "IMDb Rating Weight",
            0, 100, 20,
            step=5,
            help="Preference given to higher IMDb-rated movies"
        )

        total = w_movie + w_mood + w_imdb
        if total != 100:
            st.warning(
                f"⚠️ Weights sum to {total:}. "
                "They ideally should sum to 100."
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
    <div style="text-align: right;">
        <p>
            👨‍💻 Built by <b>Satvick Agarwal</b><br>
            🔗 
            <a href="https://www.linkedin.com/in/satvick-agarwal" target="_blank">
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
