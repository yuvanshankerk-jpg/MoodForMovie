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
    page_title="MoodForMovie — AI Cinema Matcher",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session States ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# Persistent filter keys for clean resetting
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "All Languages"
if "min_rating" not in st.session_state:
    st.session_state.min_rating = 6.0
if "vibe_intensity" not in st.session_state:
    st.session_state.vibe_intensity = 75
if "custom_query" not in st.session_state:
    st.session_state.custom_query = ""
if "mood_preset" not in st.session_state:
    st.session_state.mood_preset = "Select an emotional vibe..."

def clear_all_selections():
    st.session_state.selected_genres = []
    st.session_state.selected_lang = "All Languages"
    st.session_state.min_rating = 6.0
    st.session_state.vibe_intensity = 75
    st.session_state.custom_query = ""
    st.session_state.mood_preset = "Select an emotional vibe..."

# --- ISO Language Code to Full English Names Mapping ---
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Mandarin)",
    "cn": "Cantonese",
    "ru": "Russian",
    "pt": "Portuguese",
    "sv": "Swedish",
    "nl": "Dutch",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "te": "Telugu",
    "ta": "Tamil",
    "th": "Thai",
    "id": "Indonesian",
    "el": "Greek",
    "cs": "Czech",
    "hu": "Hungarian",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese"
}

# --- Vivid Neon-Cinema Theme Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% -10%, #151928 0%, #080a10 70%, #030407 100%);
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Vivid Glowing Brand Banner */
    .brand-container {
        text-align: center;
        padding: 30px 10px 15px 10px;
    }
    .brand-glow {
        font-size: 3.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 35%, #f093fb 70%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 35px rgba(79, 172, 254, 0.45);
        display: inline-block;
        margin-bottom: 6px;
    }
    .brand-tagline {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* Glassmorphic Vivid Cards */
    .movie-card {
        background: rgba(18, 24, 38, 0.85);
        border: 1px solid rgba(79, 172, 254, 0.25);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .movie-card:hover {
        transform: translateY(-8px) scale(1.01);
        border-color: #00f2fe;
        box-shadow: 0 16px 36px rgba(0, 242, 254, 0.3);
    }
    .poster-img {
        width: 100%;
        border-radius: 12px;
        height: 330px;
        object-fit: cover;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 700;
        height: 48px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-bottom: 6px;
    }
    .score-badge {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #000;
        font-weight: 800;
        font-size: 0.82rem;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .lang-badge {
        background: rgba(240, 147, 251, 0.15);
        color: #f093fb;
        border: 1px solid rgba(240, 147, 251, 0.35);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 5px;
        margin-left: 6px;
    }
    .genre-badge {
        background: rgba(0, 242, 254, 0.12);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.3);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 5px;
        margin-right: 4px;
        display: inline-block;
        margin-top: 6px;
    }
    .synopsis-text {
        font-size: 0.82rem;
        line-height: 1.45;
        color: #94a3b8;
        height: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Preprocessed Artifacts ---
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

# Add Full Language Name Column
df["language_full"] = df["original_language"].map(lambda x: LANG_MAP.get(str(x).lower(), str(x).upper()))

# --- Free Poster Fetcher via Wikipedia API ---
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

# --- Trailer Modal Dialog ---
@st.dialog("🎬 Watch Trailer")
def show_trailer_modal(title):
    st.write(f"### {title}")
    clean_query = urllib.parse.quote(f"{title} official trailer")
    yt_embed = f"https://www.youtube.com/embed?listType=search&list={clean_query}"
    st.components.v1.iframe(yt_embed, height=360, scrolling=False)
    st.caption("Auto-matched via YouTube Search")

# --- Vivid Header Banner ---
st.markdown("""
<div class="brand-container">
    <div class="brand-glow">MoodForMovie</div>
    <div class="brand-tagline">Adaptive Semantic Cinema Discovery • Powered by Natural Language Processing</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Filters & Clear Selection ---
st.sidebar.markdown("### 🎛️ Filter Controls")

# Clear Selection Action Button
if st.sidebar.button("🔄 Clear All Selections", use_container_width=True):
    clear_all_selections()
    st.rerun()

all_genres = sorted(list(set(" ".join(df["genres_clean"].dropna().tolist()).split())))
selected_genres = st.sidebar.multiselect(
    "Genres", 
    options=all_genres, 
    key="selected_genres"
)

# Full language options
unique_langs = sorted(df["language_full"].dropna().unique().tolist())
selected_lang = st.sidebar.selectbox(
    "Language", 
    options=["All Languages"] + unique_langs, 
    key="selected_lang"
)

min_rating = st.sidebar.slider(
    "Minimum Rating (★)", 
    1.0, 10.0, 
    key="min_rating", 
    step=0.1
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Vibe Precision")
vibe_intensity = st.sidebar.slider(
    "Vibe Dominance vs Global Rating", 
    10, 100, 
    key="vibe_intensity", 
    step=5
)

# Watchlist in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(f"📌 My Watchlist ({len(st.session_state.watchlist)})")
if st.session_state.watchlist:
    for item in st.session_state.watchlist:
        st.sidebar.markdown(f"• **{item}**")
    if st.sidebar.button("Empty Watchlist", use_container_width=True):
        st.session_state.watchlist = []
        st.rerun()
else:
    st.sidebar.caption("Click 'Watchlist' on any card to pin movies here.")

# --- Mood Archetypes & Query Input ---
col_preset, col_dice = st.columns([4, 1])

mood_presets = {
    "Select an emotional vibe...": "",
    "🧠 Mind-Bending & Psychological": "complex psychological mystery twist mind-bending suspense thriller puzzle noir",
    "😂 Feel-Good & Comfort Comedy": "hilarious fun feel-good upbeat buddy comedy lighthearted laugh cheerful warm",
    "💔 Melancholic & Deep Emotion": "emotional heartbreak deep sorrow tearjerker bittersweet poignant grief drama",
    "⚡ High Adrenaline & Combat": "action packed intense race heist combat explosive fast paced survival martial arts",
    "🌌 Atmospheric Cosmic Journey": "space exploration philosophical universe quiet breathtaking sci-fi journey existential"
}

def on_preset_change():
    preset_choice = st.session_state.mood_preset
    if preset_choice in mood_presets and preset_choice != "Select an emotional vibe...":
        st.session_state.custom_query = mood_presets[preset_choice]

with col_preset:
    chosen_preset = st.selectbox(
        "Explore Curated Emotional Archetypes:", 
        list(mood_presets.keys()), 
        key="mood_preset",
        on_change=on_preset_change
    )

with col_dice:
    st.write("") # Alignment spacing
    surprise = st.button("🎲 Surprise Me!", use_container_width=True)

user_query = st.text_input(
    "Or describe your exact vibe in natural language:",
    key="custom_query",
    placeholder="e.g., A rainy evening noir with a clever detective and jazz vibes in Tokyo"
)

num_movies = st.slider("Number of recommendations:", 3, 15, 6)

# --- Recommendation Engine ---
def recommend(query, top_n=6, random_pick=False):
    score_df = df.copy()

    # 1. Compute similarity across the entire array
    if query.strip():
        query_vec = tfidf.transform([query])
        sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        score_df["similarity"] = sim
    else:
        score_df["similarity"] = 0.0

    # 2. Weighted Score
    mood_weight = vibe_intensity / 100.0
    rating_weight = 1.0 - mood_weight
    score_df["final_score"] = (score_df["similarity"] * mood_weight) + ((score_df["vote_average"] / 10.0) * rating_weight)

    # 3. Apply Filters
    if selected_lang != "All Languages":
        score_df = score_df[score_df["language_full"] == selected_lang]

    score_df = score_df[score_df["vote_average"] >= min_rating]

    if selected_genres:
        for g in selected_genres:
            score_df = score_df[score_df["genres_clean"].str.contains(g, case=False, na=False)]

    if score_df.empty:
        return score_df

    if random_pick:
        return score_df.sample(n=min(top_n, len(score_df)))

    return score_df.sort_values(by="final_score", ascending=False).head(top_n)

# --- Results Presentation ---
search_triggered = st.button("✨ Match My Mood", type="primary", use_container_width=True) or surprise

if search_triggered:
    results = recommend(user_query, top_n=num_movies, random_pick=surprise)

    if results.empty:
        st.warning("⚠️ No films found matching those constraints. Try clicking 'Clear All Selections' or lowering the minimum rating.")
    else:
        st.markdown(f"#### 🍿 Matching Films ({len(results)} found)")
        cols = st.columns(3)

        for idx, (_, row) in enumerate(results.iterrows()):
            col = cols[idx % 3]
            poster_url = fetch_poster_wiki(row["title"])
            genres_list = row["genres_clean"].split()[:2]

            with col:
                genre_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genres_list])
                
                st.markdown(f"""
                <div class="movie-card">
                    <img class="poster-img" src="{poster_url}">
                    <div class="movie-title">{row['title']}</div>
                    <div style="margin: 4px 0 6px 0;">
                        <span class="score-badge">★ {row['vote_average']}/10</span>
                        <span class="lang-badge">{row['language_full']}</span>
                        <span style="font-size:0.75rem; color:#888; margin-left:6px;">({int(row['vote_count']):,} votes)</span>
                    </div>
                    <div>{genre_html}</div>
                    <div class="synopsis-text">{row['overview']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Card Action Buttons
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    if st.button("▶ Trailer", key=f"trailer_{row['id']}_{idx}", use_container_width=True):
                        show_trailer_modal(row['title'])
                with c_btn2:
                    is_in_wl = row['title'] in st.session_state.watchlist
                    btn_label = "❤️ Saved" if is_in_wl else "➕ Watchlist"
                    if st.button(btn_label, key=f"wl_{row['id']}_{idx}", use_container_width=True):
                        if not is_in_wl:
                            st.session_state.watchlist.append(row['title'])
                            st.rerun()
