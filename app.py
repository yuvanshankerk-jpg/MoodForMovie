import os
import pickle
import urllib.parse
import requests
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="MoodForMovie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b0c10 0%, #1f2833 100%);
        color: #c5c6c7;
        font-family: 'Inter', sans-serif;
    }
    .movie-card {
        background-color: rgba(31, 40, 51, 0.9);
        border: 1px solid #45a29e;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .movie-title {
        color: #66fcf1;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .rating-badge {
        background-color: #e5a00d;
        color: #0b0c10;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .review-box {
        background: rgba(0, 0, 0, 0.35);
        border-left: 3px solid #66fcf1;
        padding: 8px 12px;
        font-style: italic;
        font-size: 0.85rem;
        margin-top: 10px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "movies_processed.pkl"), "rb") as f:
        movies = pickle.load(f)
    with open(os.path.join(base_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(base_dir, "tfidf_matrix.pkl"), "rb") as f:
        matrix = pickle.load(f)
    return movies, tfidf, matrix

df, tfidf, tfidf_matrix = load_data()

@st.cache_data(show_spinner=False)
def fetch_poster_wiki(title):
    headers = {"User-Agent": "MoodForMovieApp/1.0 (student project)"}
    for q in [f"{title} (film)", title]:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(q)}"
            res = requests.get(url, headers=headers, timeout=2.5)
            if res.status_code == 200:
                data = res.json()
                if "thumbnail" in data and "source" in data["thumbnail"]:
                    return data["thumbnail"]["source"]
        except:
            continue
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=400&q=80"

st.title("🎬 MoodForMovie")
st.caption("AI-powered contextual film discovery based on your mood and emotional preferences.")

st.sidebar.header("🎯 Filters & Controls")
all_genres = sorted(list(set(" ".join(df["genres_clean"].dropna().tolist()).split())))
selected_genres = st.sidebar.multiselect("Filter by Genres", options=all_genres, default=[])

languages = sorted(df["original_language"].dropna().unique().tolist())
selected_lang = st.sidebar.selectbox("Original Language", options=["All"] + languages, index=0)

min_rating = st.sidebar.slider("Minimum Rating", min_value=1.0, max_value=10.0, value=6.0, step=0.1)

st.markdown("### 1. What are you in the mood for?")
mood_presets = {
    "Select a quick mood...": "",
    "🧠 Mind-Bending & Psychological": "complex psychological mystery twist mind-bending suspense thriller puzzle",
    "😂 Lighthearted & Fun": "hilarious fun feel-good upbeat comedy laugh cheerful",
    "💔 Melancholic & Deep": "emotional heartbreak deep sorrow tearjerker bittersweet drama sentimental",
    "⚡ Adrenaline & Action": "action packed intense race heist combat explosive fast paced",
    "🌌 Sci-Fi & Atmospheric": "space exploration philosophical universe quiet breathtaking sci-fi journey"
}

chosen_preset = st.selectbox("Quick Mood Presets:", list(mood_presets.keys()))

user_query = st.text_input(
    "Or type your custom mood prompt:",
    value=mood_presets[chosen_preset] if chosen_preset != "Select a quick mood..." else "",
    placeholder="e.g., A rainy evening noir with a clever detective and jazz vibes"
)

num_movies = st.slider("Number of recommendations:", 3, 12, 6)

def recommend(query, top_n=6):
    if not query.strip():
        temp = df[df["vote_average"] >= min_rating]
        if selected_lang != "All":
            temp = temp[temp["original_language"] == selected_lang]
        return temp.sort_values(by="vote_average", ascending=False).head(top_n)

    query_vec = tfidf.transform([query])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    score_df = df.copy()
    score_df["similarity"] = sim

    if selected_lang != "All":
        score_df = score_df[score_df["original_language"] == selected_lang]

    score_df = score_df[score_df["vote_average"] >= min_rating]

    if selected_genres:
        for g in selected_genres:
            score_df = score_df[score_df["genres_clean"].str.contains(g, case=False, na=False)]

    score_df["score"] = (score_df["similarity"] * 0.8) + ((score_df["vote_average"] / 10.0) * 0.2)
    return score_df.sort_values(by="score", ascending=False).head(top_n)

if st.button("✨ Match My Mood", type="primary"):
    results = recommend(user_query, top_n=num_movies)

    if results.empty:
        st.warning("No matches found. Try lowering the minimum rating or clearing genre filters.")
    else:
        st.markdown(f"#### 🍿 Recommended Movies ({len(results)} matches)")
        cols = st.columns(3)

        sample_reviews = [
            "“Captures the mood completely with gripping storytelling.”",
            "“Atmospheric, smartly written, and visually arresting from start to finish.”",
            "“A resonant watch with strong pacing and emotional weight.”"
        ]

        for idx, (_, row) in enumerate(results.iterrows()):
            col = cols[idx % 3]
            poster_url = fetch_poster_wiki(row["title"])
            review = sample_reviews[idx % len(sample_reviews)]

            with col:
                st.markdown(f"""
                <div class="movie-card">
                    <img src="{poster_url}" style="width:100%; border-radius:8px; height:320px; object-fit:cover; margin-bottom:12px;">
                    <div class="movie-title">{row['title']}</div>
                    <p style="margin: 4px 0 8px 0;">
                        <span class="rating-badge">★ {row['vote_average']}/10</span>
                        <span style="font-size:0.8rem; color:#888; margin-left:8px;">({int(row['vote_count']):,} votes)</span>
                    </p>
                    <p style="font-size: 0.8rem; color: #a0aec0; height: 60px; overflow: hidden; text-overflow: ellipsis;">
                        {row['overview']}
                    </p>
                    <div class="review-box">
                        💬 <b>Viewer Take:</b> {review}
                    </div>
                </div>
                """, unsafe_allow_html=True)
