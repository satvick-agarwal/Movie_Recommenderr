# recommender.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Mood mapping (used in notebook)
# ---------------------------
mood_map = {
    "happy": "uplifting feel-good light hearted comedy joyful",
    "romantic": "romantic love emotional relationships heartfelt",
    "sad": "emotional sad touching tearjerker",
    "thriller": "thrilling suspense mystery intense",
    "action": "action packed adventure fast paced exciting"
}

# ---------------------------
# Movie-based Recommendation
# ---------------------------
def recommend_by_movie(
    title,
    df,
    embeddings,
    top_n=5,
    alpha=0.7
):
    title = title.lower().strip()

    matches = df[df["title"].str.lower().str.contains(title)]

    if matches.empty:
        return pd.DataFrame({"Error": ["Movie not found"]})

    idx = matches.index[0]

    movie_vec = embeddings[idx].reshape(1, -1)
    similarity_scores = cosine_similarity(movie_vec, embeddings)[0]

    # Normalize IMDb rating
    imdb_norm = df["imdb_rating"] / 10

    # Final ranking score
    final_score = alpha * similarity_scores + (1 - alpha) * imdb_norm

    result_df = df.copy()
    result_df["final_score"] = final_score

    recommendations = (
        result_df
        .sort_values("final_score", ascending=False)
        .iloc[1: top_n + 1]
        .reset_index(drop=True)
    )

    recommendations.insert(0, "Rank", range(1, top_n + 1))

    return recommendations[
        ["Rank", "title", "genres", "imdb_rating"]
    ]

# ---------------------------
# Mood-based Recommendation
# ---------------------------
def recommend_by_mood(
    mood,
    df,
    embeddings,
    model,
    top_n=5,
    alpha=0.7
):
    mood = mood.lower().strip()

    if mood not in mood_map:
        return pd.DataFrame({
            "Error": [f"Available moods: {list(mood_map.keys())}"]
        })

    mood_text = mood_map[mood]

    # Convert mood text to embedding
    mood_embedding = model.encode(mood_text).reshape(1, -1)

    similarity_scores = cosine_similarity(
        mood_embedding,
        embeddings
    )[0]

    imdb_norm = df["imdb_rating"] / 10
    final_score = alpha * similarity_scores + (1 - alpha) * imdb_norm

    result_df = df.copy()
    result_df["final_score"] = final_score

    recommendations = (
        result_df
        .sort_values("final_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    recommendations.insert(0, "Rank", range(1, top_n + 1))

    return recommendations[
        ["Rank", "title", "genres", "imdb_rating"]
    ]
#----------------------
#Hybrid Recommendation
#----------------------
def recommend_hybrid(
    df,
    embeddings,
    model,
    movie=None,
    mood=None,
    top_n=5,
    w_movie=0.4,
    w_mood=0.4,
    w_imdb=0.2
):
    scores = np.zeros(len(df))
    drop_idx = None

    # ---------------- Movie similarity ----------------
    if movie:
        movie = movie.lower().strip()
        matches = df[df["title"].str.lower().str.contains(movie)]

        if not matches.empty:
            idx = matches.index[0]
            drop_idx = idx

            movie_vec = embeddings[idx].reshape(1, -1)
            movie_sim = cosine_similarity(
                movie_vec, embeddings
            )[0]

            scores += w_movie * movie_sim

    # ---------------- Mood similarity ----------------
    if mood:
        mood = mood.lower().strip()

        if mood in mood_map:
            mood_text = mood_map[mood]
            mood_embedding = model.encode(
                mood_text
            ).reshape(1, -1)

            mood_sim = cosine_similarity(
                mood_embedding, embeddings
            )[0]

            scores += w_mood * mood_sim

    # ---------------- IMDb weight ----------------
    imdb_norm = df["imdb_rating"] / 10
    scores += w_imdb * imdb_norm.values

    # ---------------- Final ranking ----------------
    temp_df = df.copy()
    temp_df["final_score"] = scores

    if drop_idx is not None:
        temp_df = temp_df.drop(index=drop_idx)

    results = (
        temp_df.sort_values("final_score", ascending=False)
               .head(top_n)
               .reset_index(drop=True)
    )

    results.insert(0, "Rank", range(1, len(results) + 1))

    return results[
        ["Rank", "title", "genres", "imdb_rating"]
    ]
