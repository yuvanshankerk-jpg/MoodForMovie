import os
import pickle
import urllib.parse
import requests
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- App Configuration ---
st.set_page_config(
    page_title="MoodForMovie — Interactive Cinema Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State for Watchlist ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# --- Premium Cyber-Cinema Styling ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1a1e29 0%, #0b0c10 80%);
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }
    .hero-container {
        text-align: center;
        padding: 24px 0 10px 0;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #66fcf1, #45a29e, #e5a00d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .movie-card {
        background: rgba(26, 34, 46, 0.75);
        border: 1px solid rgba(102, 252, 241, 0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        border-color: #66fcf1;
        box-shadow: 0 8px 24px rgba(102, 252, 241, 0.25);
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        height: 48px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .rating-pill {
        background-color: #e5a00d;
        color: #0b0c10;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.8rem;
    }
    .vibe-tag {
        background: rgba(102, 252, 241, 0.15);
        color: #66fcf1;
        border: 1px solid rgba(102, 252, 241, 0.3);
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .overview-text {
        font-size: 0.82rem;
        color: #94a3b8;
        height: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
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

# --- Poster Fetcher via Wikipedia API ---
@st.cache_data(show_spinner=False)
def fetch_poster_wiki(title):
    headers = {"User-Agent": "MoodForMovieInteractive/2.0 (student project)"}
    for q in [f"{title} (film)", title]:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(q)}"
            res = requests.get(url, headers=headers, timeout=2.0)
            if res.status_code == 200:
                thumb = res.json().get("thumbnail")
                if thumb and "source" in thumb:
                    return thumb["source"]
        except:
            continue
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=400&q=80"

# --- Trailer Dialog Modal ---
@st.dialog("🎬 Watch Trailer")
def show_trailer_modal(title):
    st.write(f"### {title}")
    clean_query = urllib.parse.quote(f"{title} official trailer")
    yt_search_embed = f"https://www.youtube.com/embed?listType=search&list={clean_query}"
    st.components.v1.iframe(yt_search_embed, height=360, scrolling=False)
    st.caption("Auto-matched via YouTube Search")

# --- Header Section ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">MoodForMovie</div>
    <p style="color:#94a3b8; font-size:1.05rem;">Dial in your exact mental state, pace, and vibe to find your next favorite film.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("🎛️ Cinema Controls")

all_genres = sorted(list(set(" ".join(df["genres_clean"].dropna().tolist()).split())))
selected_genres = st.sidebar.multiselect("Include Specific Genres", options=all_genres)

languages = sorted(df["original_language"].dropna().unique().tolist())
selected_lang = st.sidebar.selectbox("Language Filter", options=["All"] + languages, index=0)

min_rating = st.sidebar.slider("Minimum IMDb/TMDb Score", 1.0, 10.0, 6.5, 0.1)

# Interactive Tuning Sliders
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Vibe Weighting")
vibe_intensity = st.sidebar.slider("Match Purity (Mood match vs Overall Acclaim)", 10, 100, 75, 5)

# Watchlist Drawer in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(f"📌 My Watchlist ({len(st.session_state.watchlist)})")
if st.session_state.watchlist:
    for item in st.session_state.watchlist:
        st.sidebar.markdown(f"• **{item}**")
    if st.sidebar.button("Clear Watchlist"):
        st.session_state.watchlist = []
        st.rerun()
else:
    st.sidebar.caption("No movies added yet. Click 'Add to Watchlist' on any card.")

# --- Interactive Mood Selector ---
col_preset, col_btn = st.columns([4, 1])

mood_presets = {
    "Select an emotional vibe...": "",
    "🧠 Mind-Bending & Psychological": "complex psychological mystery twist mind-bending suspense thriller puzzle noir",
    "😂 Feel-Good & Comfort Comedy": "hilarious fun feel-good upbeat buddy comedy lighthearted laugh cheerful warm",
    "💔 Melancholic & Poignant": "emotional heartbreak deep sorrow tearjerker bittersweet poignant grief drama",
    "⚡ High Adrenaline & Combat": "action packed intense race heist combat explosive fast paced survival martial arts",
    "🌌 Atmospheric & Wonder": "space exploration philosophical universe quiet breathtaking sci-fi journey existential"
}

with col_preset:
    chosen_preset = st.selectbox("Quick Mood Archetypes:", list(mood_presets.keys()), label_visibility="collapsed")

with col_btn:
    surprise = st.button("🎲 Surprise Me!", use_container_width=True)

user_query = st.text_input(
    "Describe your desired cinema vibe in freeform words:",
    value=mood_presets[chosen_preset] if (chosen_preset != "Select an emotional vibe..." and not surprise) else "",
    placeholder="e.g., A rainy night mystery with a lonely jazz detective in Tokyo"
)

num_movies = st.slider("Number of recommendations to render:", 3, 12, 6)

# --- Recommendation Logic ---
def recommend(query, top_n=6, random_pick=False):
    score_df = df.copy()

    if selected_lang != "All":
        score_df = score_df[score_df["original_language"] == selected_lang]

    score_df = score_df[score_df["vote_average"] >= min_rating]

    if selected_genres:
        for g in selected_genres:
            score_df = score_df[score_df["genres_clean"].str.contains(g, case=False, na=False)]

    if score_df.empty:
        return score_df

    if random_pick:
        return score_df.sample(n=min(top_n, len(score_df)))

    if not query.strip():
        return score_df.sort_values(by="vote_average", ascending=False).head(top_n)

    # NLP Semantic Cosine Match
    query_vec = tfidf.transform([query])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    score_df["similarity"] = sim

    # Weighted scoring formula driven by sidebar slider
    mood_weight = vibe_intensity / 100.0
    rating_weight = 1.0 - mood_weight

    score_df["final_score"] = (score_df["similarity"] * mood_weight) + ((score_df["vote_average"] / 10.0) * rating_weight)
    return score_df.sort_values(by="final_score", ascending=False).head(top_n)

# --- Trigger Search ---
search_triggered = st.button("✨ Match My Mood", type="primary", use_container_width=True) or surprise

if search_triggered:
    results = recommend(user_query, top_n=num_movies, random_pick=surprise)

    if results.empty:
        st.warning("⚠️ No films match those strict filters! Try lowering the minimum rating or clearing genre constraints.")
    else:
        st.markdown(f"#### 🍿 Top Matches for Your Vibe")
        cols = st.columns(3)

        for idx, (_, row) in enumerate(results.iterrows()):
            col = cols[idx % 3]
            poster_url = fetch_poster_wiki(row["title"])
            genres_sample = " • ".join(row["genres_clean"].split()[:2])

            with col:
                st.markdown(f"""
                <div class="movie-card">
                    <img src="{poster_url}" style="width:100%; border-radius:10px; height:310px; object-fit:cover; margin-bottom:10px;">
                    <div class="movie-title">{row['title']}</div>
                    <div style="margin: 6px 0;">
                        <span class="rating-pill">★ {row['vote_average']}/10</span>
                        <span style="font-size:0.75rem; color:#888; margin-left:6px;">({int(row['vote_count']):,} ratings)</span>
                    </div>
                    <div>
                        <span class="vibe-tag">{genres_sample or "Cinema"}</span>
                        <span class="vibe-tag">Lang: {row['original_language'].upper()}</span>
                    </div>
                    <div class="overview-text">{row['overview']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Interactive Action Bar per Card
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button(f"▶ Trailer", key=f"trailer_{row['id']}_{idx}", use_container_width=True):
                        show_trailer_modal(row['title'])
                with b_col2:
                    is_in_wl = row['title'] in st.session_state.watchlist
                    btn_label = "❤️ Saved" if is_in_wl else "➕ Watchlist"
                    if st.button(btn_label, key=f"wl_{row['id']}_{idx}", use_container_width=True):
                        if not is_in_wl:
                            st.session_state.watchlist.append(row['title'])
                            st.rerun()
