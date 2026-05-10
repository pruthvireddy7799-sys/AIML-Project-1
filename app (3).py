import streamlit as st
import pandas as pd
import requests
import re
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="PICKFLIX", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #07070f !important;
    color: #f0f0f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 1380px !important;
}

/* ── HERO ── */
.hero {
    position: relative; overflow: hidden;
    background:
        linear-gradient(160deg, rgba(7,7,15,0.96) 45%, rgba(30,0,0,0.88) 100%),
        url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1974&auto=format&fit=crop");
    background-size: cover; background-position: center;
    padding: 110px 90px 90px;
    border-radius: 0 0 60px 60px;
    margin-bottom: 60px;
    border-bottom: 1px solid rgba(229,9,20,0.2);
    box-shadow: 0 50px 130px rgba(229,9,20,0.1);
    animation: heroFadeIn 1s ease both;
}
@keyframes heroFadeIn {
    from { opacity:0; transform:translateY(-18px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero-glow-1 {
    position:absolute; top:-180px; left:-180px;
    width:550px; height:550px;
    background:radial-gradient(circle, rgba(229,9,20,0.2) 0%, transparent 70%);
    pointer-events:none;
    animation: glowPulse 4s ease-in-out infinite alternate;
}
.hero-glow-2 {
    position:absolute; bottom:-200px; right:-120px;
    width:600px; height:600px;
    background:radial-gradient(circle, rgba(80,0,160,0.15) 0%, transparent 70%);
    pointer-events:none;
    animation: glowPulse 5s ease-in-out infinite alternate-reverse;
}
@keyframes glowPulse {
    from { opacity:0.6; transform:scale(0.95); }
    to   { opacity:1;   transform:scale(1.08); }
}
.hero-eyebrow {
    font-size:12px; font-weight:600; letter-spacing:5px; text-transform:uppercase;
    color:#E50914; margin-bottom:18px;
    animation: slideUp 0.7s 0.2s ease both;
}
.hero-title {
    font-family:'Bebas Neue',sans-serif; font-size:130px; line-height:0.88;
    letter-spacing:3px; margin-bottom:28px;
    background:linear-gradient(135deg,#ffffff 0%,#aaaaaa 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation: slideUp 0.7s 0.35s ease both;
}
.hero-sub {
    font-size:20px; color:rgba(255,255,255,0.5); line-height:1.75;
    max-width:560px; font-weight:300;
    animation: slideUp 0.7s 0.5s ease both;
}
.hero-pill {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(229,9,20,0.12); border:1px solid rgba(229,9,20,0.35);
    border-radius:100px; padding:8px 22px; font-size:12px; font-weight:600;
    color:#ff7070; margin-top:40px; letter-spacing:2px; text-transform:uppercase;
    animation: slideUp 0.7s 0.65s ease both;
    transition: background 0.3s, box-shadow 0.3s, transform 0.3s;
}
.hero-pill:hover {
    background:rgba(229,9,20,0.22);
    box-shadow:0 0 24px rgba(229,9,20,0.3);
    transform:scale(1.04);
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(22px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── GENRE SECTION HEADER ── */
.genre-section-header {
    display: flex; align-items: baseline; gap: 16px; margin-bottom: 28px;
}
.genre-section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 48px; letter-spacing: 2px; color: #fff; line-height: 1;
}
.genre-section-sub {
    font-size: 13px; color: rgba(255,255,255,0.3); letter-spacing: 0.5px;
}

/* Active indicator */
.genre-active-bar {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(229,9,20,0.6), transparent);
    border-radius: 2px; margin: 6px 0 16px;
}
.genre-active-dot {
    display: inline-block; width: 5px; height: 5px; border-radius: 50%;
    background: #E50914; margin-right: 7px;
    box-shadow: 0 0 6px rgba(229,9,20,0.8);
    animation: dotPulse 1.8s ease-in-out infinite;
    vertical-align: middle; position: relative; top: -1px;
}
@keyframes dotPulse {
    0%,100% { box-shadow: 0 0 4px rgba(229,9,20,0.6); }
    50%      { box-shadow: 0 0 12px rgba(229,9,20,1); }
}
.genre-active-label {
    font-size: 12px; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; color: rgba(229,9,20,0.7); margin-bottom: 24px;
}

/* ── SELECTBOX ── */
.stSelectbox label { display: none !important; }
div[data-baseweb="select"] > div {
    background: #1a1a2e !important; border-radius: 16px !important;
    border: 1.5px solid rgba(255,255,255,0.12) !important;
    padding: 6px 18px !important; min-height: 58px !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(229,9,20,0.55) !important;
    box-shadow: 0 0 0 3px rgba(229,9,20,0.1) !important;
}
div[data-baseweb="select"] span, input, p { color: #ffffff !important; }
[data-baseweb="popover"] > div {
    background: #0e0e1c !important; border: 1px solid rgba(255,255,255,0.08) !important;
}
[role="option"]:hover { background: rgba(229,9,20,0.14) !important; color: #ffffff !important; }
[aria-selected="true"] { background: rgba(229,9,20,0.2) !important; color: #ffffff !important; }

/* ── MAIN BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #E50914 0%, #a50710 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 14px !important;
    padding: 17px 48px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-4px) scale(1.03) !important;
    box-shadow: 0 20px 55px rgba(229,9,20,0.55) !important;
}

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(229,9,20,0.35), rgba(120,0,200,0.2), transparent);
    margin: 55px 0 45px; border: none;
}

/* ── SECTION HEADERS ── */
.section-header { margin-bottom: 36px; }
.section-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 68px; font-weight: 400;
    letter-spacing: 2px; color: #ffffff; line-height: 1; margin: 0 0 6px;
}
.section-sub { font-size: 14px; color: rgba(255,255,255,0.35); letter-spacing: 0.5px; }

/* ── SEARCH PANEL ── */
.search-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 28px; padding: 44px 52px 20px; margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.search-panel-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 32px;
    letter-spacing: 2px; color: rgba(255,255,255,0.85); margin-bottom: 6px;
}

/* ── MOVIE CARDS ── */
.movie-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.055);
    border-radius: 22px; padding: 12px 12px 18px;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
    height: 100%; position: relative; overflow: hidden;
}
.movie-card:hover {
    transform: translateY(-14px); background: rgba(255,255,255,0.055);
    border-color: rgba(229,9,20,0.28);
    box-shadow: 0 35px 90px rgba(0,0,0,0.65), 0 0 0 1px rgba(229,9,20,0.12);
}
.poster-wrap { position: relative; border-radius: 14px; overflow: hidden; cursor: pointer; }
.poster-wrap img { width: 100%; border-radius: 14px !important; transition: transform 0.5s !important; }
.poster-wrap:hover img { transform: scale(1.06) !important; }
.poster-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(0deg, rgba(7,7,15,0.97) 0%, rgba(7,7,15,0.7) 45%, rgba(7,7,15,0.1) 75%);
    opacity: 0; transition: opacity 0.35s ease;
    display: flex; flex-direction: column; justify-content: flex-end;
    padding: 18px; border-radius: 14px;
}
.poster-wrap:hover .poster-overlay { opacity: 1; }
.overlay-overview {
    font-size: 11.5px; line-height: 1.65; color: rgba(255,255,255,0.88);
    display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical; overflow: hidden;
}
.overlay-tag { font-size: 10px; font-weight: 700; color: #E50914; margin-bottom: 8px; text-transform: uppercase; }
.movie-title { font-size: 14px; font-weight: 600; text-align: center; margin-top: 16px; min-height: 58px; color: rgba(255,255,255,0.88); }
.rating { text-align: center; font-size: 15px; font-weight: 700; color: #FFD54F; margin: 10px 0 14px; }

/* ── YOUTUBE STYLE SKELETON LOADER ── */
@keyframes yt-shimmer {
    0% { background-position: -800px 0; }
    100% { background-position: 800px 0; }
}
.yt-skeleton-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 22px; padding: 12px;
    height: 100%; min-height: 420px;
    display: flex; flex-direction: column; gap: 12px;
}
.yt-skeleton-img {
    width: 100%; aspect-ratio: 2/3; border-radius: 14px;
    background: #1a1a24;
    background-image: linear-gradient(90deg, rgba(255,255,255,0.02) 25%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.02) 75%);
    background-size: 800px 100%;
    animation: yt-shimmer 2s infinite linear;
}
.yt-skeleton-text {
    height: 14px; border-radius: 6px; background: #1a1a24;
    background-image: linear-gradient(90deg, rgba(255,255,255,0.02) 25%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.02) 75%);
    background-size: 800px 100%;
    animation: yt-shimmer 2s infinite linear;
}
.yt-skeleton-text.short { width: 50%; align-self: center; margin-bottom: 20px;}

/* ── TRENDING BADGE ── */
.trending-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(229,9,20,0.2), rgba(229,9,20,0.08));
    border: 1px solid rgba(229,9,20,0.3); border-radius: 8px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
    color: #ff6b6b; margin-bottom: 10px; width: fit-content;
}
.rank-number {
    position: absolute; top: 12px; left: 16px;
    font-family: 'Bebas Neue', sans-serif; font-size: 52px; line-height: 1;
    color: rgba(255,255,255,0.13); pointer-events: none; z-index: 2;
    text-shadow: 0 2px 8px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="hero">
    <div class="hero-glow-1"></div>
    <div class="hero-glow-2"></div>
    <div class="hero-eyebrow">🎬 &nbsp; Your Personal Cinema</div>
    <div class="hero-title">PICKFLIX</div>
    <div class="hero-sub">Discover movies you love.</div>
    <div class="hero-pill">✦ &nbsp; Curated just for you</div>
</div>
""", unsafe_allow_html=True)

# TODO: MUST UPDATE THIS API KEY IF BLANK POSTERS PERSIST
API_KEY = "ee382e8c26eb177d62b556ae2fcea027" 

# ── TRAILER MODAL FUNCTION ──
@st.dialog("▶ Watch Trailer")
def show_trailer_modal(video_url):
    st.video(video_url)

# ── LOAD DATA ──
@st.cache_data
def load_data():
    # Make sure movies.csv and ratings.csv exist in your directory
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()
movies_raw = movies.copy()

# ── CONTENT-BASED SIMILARITY ──
movies_cb = movies.copy()
movies_cb['genres_clean'] = movies_cb['genres'].fillna('').str.replace('|', ' ', regex=False)
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies_cb['genres_clean'])
content_sim = cosine_similarity(genre_matrix)
title_to_idx = {title: i for i, title in enumerate(movies_cb['title'])}

# ── CONCURRENT API FETCHING & UTILS ──
def render_skeleton(placeholder):
    placeholder.markdown('''
    <div class="yt-skeleton-card">
        <div class="yt-skeleton-img"></div>
        <div class="yt-skeleton-text"></div>
        <div class="yt-skeleton-text short"></div>
    </div>
    ''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie):
    try:
        movie_clean = movie.split('(')[0].strip()
        year_match = re.search(r"\((\d{4})\)", movie)
        year = year_match.group(1) if year_match else ""
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_clean}&year={year}"
        
        res = requests.get(url, timeout=5)
        data = res.json()
        
        if 'success' in data and data['success'] is False:
            return None 
            
        if not data.get('results'):
            return None
            
        d = data['results'][0]
        poster = f"https://image.tmdb.org/t/p/w500{d['poster_path']}" if d.get('poster_path') else None
        rating = round(d.get('vote_average', 0), 1)
        overview = d.get('overview', '')
        trailer = None
        mid = d.get('id')
        
        if mid:
            vids = requests.get(
                f"https://api.themoviedb.org/3/movie/{mid}/videos?api_key={API_KEY}", timeout=5
            ).json()
            for v in vids.get('results', []):
                if v['site'] == 'YouTube':
                    trailer = f"https://www.youtube.com/watch?v={v['key']}"
                    break
        return {"poster": poster, "rating": rating, "trailer": trailer, "overview": overview}
    except Exception:
        return None

# FASTER: Fetch multiple movies concurrently
def fetch_details_concurrent(movies_list):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(fetch_movie_details, movies_list))

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trending_basic():
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={API_KEY}"
        res = requests.get(url, timeout=5).json()
        if 'success' in res and res['success'] is False:
            st.error(f"TMDB API Error: {res['status_message']}")
            return []
        return res.get('results', [])[:5]
    except Exception:
        return []

def fetch_trailer_only(mid):
    if not mid: return None
    try:
        vids = requests.get(f"https://api.themoviedb.org/3/movie/{mid}/videos?api_key={API_KEY}", timeout=5).json()
        for v in vids.get('results', []):
            if v['site'] == 'YouTube':
                return f"https://www.youtube.com/watch?v={v['key']}"
    except:
        pass
    return None

# ── RECOMMEND ──
def recommend(movie_name, top_n=5):
    if movie_name not in title_to_idx:
        return []
    idx = title_to_idx[movie_name]
    scores = sorted(enumerate(content_sim[idx]), key=lambda x: x[1], reverse=True)
    return [movies_cb.iloc[i[0]].title for i in scores[1:top_n+1]]

# ── RENDER MOVIE CARD ──
def render_movie_card(placeholder, movie_title, details, rank=None):
    with placeholder.container():
        poster_url = (
            details['poster'] if details and details['poster']
            else "https://placehold.co/300x450/111122/333?text=No+Poster"
        )
        overview_safe = ""
        if details and details.get('overview'):
            overview_safe = details['overview'].replace('"', '&quot;').replace("'", "&#39;")[:400]
            
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        
        # Inject rank overlay if trending
        rank_html = f'<div class="rank-number">{rank}</div>' if rank else ''
        
        st.markdown(f"""
        <div class="poster-wrap">
            <img src="{poster_url}" style="width:100%;" />
            <div class="poster-overlay">
                <div class="overlay-tag">Overview</div>
                <div class="overlay-overview">{overview_safe or 'No description available.'}</div>
            </div>
            {rank_html}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div class='movie-title'>{movie_title}</div>", unsafe_allow_html=True)
        
        if details:
            filled = int(details['rating'] / 2)
            stars = "★" * filled + "☆" * (5 - filled)
            st.markdown(f"<div class='rating'>{stars} &nbsp;{details['rating']}</div>", unsafe_allow_html=True)
            
            if details.get('trailer'):
                if st.button("▶ Watch Trailer", key=f"trailer_btn_{movie_title}_{rank}", use_container_width=True):
                    show_trailer_modal(details['trailer'])
                    
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# GENRE FILTER
# =========================================================

genres_set = set()
for g_str in movies_raw['genres'].dropna():
    for g in g_str.split('|'):
        g = g.strip()
        if g and g != '(no genres listed)':
            genres_set.add(g)
all_genres = sorted(genres_set)
genre_options = ["All"] + all_genres

st.markdown("""
<div class="genre-section-header">
    <span class="genre-section-title">Browse by Genre</span>
    <span class="genre-section-sub">Pick a mood</span>
</div>
""", unsafe_allow_html=True)

try:
    selected_genre = st.pills("Select Genre", genre_options, default="All", label_visibility="collapsed")
    if selected_genre is None: 
        selected_genre = "All"
except AttributeError:
    selected_genre = st.radio("Select Genre", genre_options, horizontal=True, label_visibility="collapsed")

if selected_genre != "All":
    st.markdown(f"""
    <div style="margin-top:16px;">
        <div class="genre-active-bar"></div>
        <div class="genre-active-label">
            <span class="genre-active-dot"></span>Showing: {selected_genre}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

if selected_genre == "All":
    filtered_movies = movies_raw.copy()
else:
    mask = movies_raw['genres'].str.contains(
        r'(?:^|\|)' + re.escape(selected_genre) + r'(?:\||$)',
        na=False, regex=True
    )
    filtered_movies = movies_raw[mask].copy()

# ── SHOW GENRE MOVIES (With Skeletons & Concurrency) ──
if selected_genre != "All":
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">{selected_genre} Films</div>
        <div class="section-sub">Top picks from our library</div>
    </div>
    """, unsafe_allow_html=True)

    genre_titles = list(filtered_movies['title'].dropna().unique()[:10])

    if genre_titles:
        row1 = genre_titles[:5]
        row2 = genre_titles[5:10]

        # Skeletons for Row 1
        g_cols1 = st.columns(5, gap="medium")
        ph1 = [col.empty() for col in g_cols1]
        for ph in ph1: render_skeleton(ph)

        # Skeletons for Row 2
        ph2 = []
        if row2:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            g_cols2 = st.columns(5, gap="medium")
            ph2 = [col.empty() for col in g_cols2]
            for ph in ph2: render_skeleton(ph)

        # Concurrent Fetch
        all_genre_details = fetch_details_concurrent(genre_titles)

        # Overwrite Row 1
        for i, (m, d) in enumerate(zip(row1, all_genre_details[:5])):
            ph1[i].empty()
            render_movie_card(ph1[i], m, d)
            
        # Overwrite Row 2
        if row2:
            for i, (m, d) in enumerate(zip(row2, all_genre_details[5:10])):
                ph2[i].empty()
                render_movie_card(ph2[i], m, d)
    else:
        st.markdown("<p style='color:rgba(255,255,255,0.3);'>No movies found for this genre.</p>", unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# =========================================================
# TRENDING (With Skeletons & Concurrency)
# =========================================================

st.markdown("""
<div class="section-header">
    <div class="trending-badge">🔥 &nbsp; This Week</div>
    <div class="section-title">Trending Now</div>
    <div class="section-sub">What everyone is watching right now</div>
</div>
""", unsafe_allow_html=True)

# Setup Skeletons
t_cols = st.columns(5, gap="medium")
t_phs = [col.empty() for col in t_cols]
for ph in t_phs: render_skeleton(ph)

trending_raw = fetch_trending_basic()

if not trending_raw:
    for ph in t_phs: ph.empty()
    st.markdown("<p style='color:rgba(255,255,255,0.3);font-size:14px;'>Could not load trending movies. Make sure your API key is correct.</p>", unsafe_allow_html=True)
else:
    # Concurrent Trailer Fetch
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        trailers = list(executor.map(fetch_trailer_only, [d.get('id') for d in trending_raw]))

    for rank, (ph, d, trailer) in enumerate(zip(t_phs, trending_raw, trailers), 1):
        movie_title = d.get('title', '')
        details = {
            "rating": round(d.get('vote_average', 0), 1),
            "overview": d.get('overview', ''),
            "poster": f"https://image.tmdb.org/t/p/w500{d['poster_path']}" if d.get('poster_path') else None,
            "trailer": trailer
        }
        ph.empty()
        render_movie_card(ph, movie_title, details, rank=rank)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# =========================================================
# SEARCH PANEL & RECOMMENDATIONS (Fixed Disappearing Bug)
# =========================================================

# Initialize Session State
if 'recommend_target' not in st.session_state:
    st.session_state.recommend_target = None

st.markdown("""
<div class="search-panel">
    <div class="search-panel-title">Find Your Next Watch</div>
    <div class="search-panel-hint">Type or scroll to pick a title from our library</div>
</div>
""", unsafe_allow_html=True)

movie_list = sorted(filtered_movies['title'].dropna().unique())

if not movie_list:
    st.markdown(
        f"<p style='color:rgba(255,255,255,0.4);font-size:14px;padding:12px 0;'>"
        f"No movies found for <strong style='color:#ff7070'>{selected_genre}</strong>.</p>",
        unsafe_allow_html=True
    )
    st.stop()

selected_movie = st.selectbox("Pick a movie", movie_list, label_visibility="collapsed")
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# Generate Button updates session state
if st.button("▶  Find Movies"):
    st.session_state.recommend_target = selected_movie

# Render Recommendations persistently if State exists
if st.session_state.recommend_target:
    recommendations = recommend(st.session_state.recommend_target)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    title_clean = st.session_state.recommend_target.split('(')[0].strip()
    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">You Might Love</div>
        <div class="section-sub">Because you picked &ldquo;{title_clean}&rdquo;</div>
    </div>
    """, unsafe_allow_html=True)

    # Set up Skeletons
    result_cols = st.columns(5, gap="medium")
    result_phs = [col.empty() for col in result_cols]
    for ph in result_phs: render_skeleton(ph)

    # Concurrent Fetch for Recommendations
    details_list = fetch_details_concurrent(recommendations)

    for i, (movie, details) in enumerate(zip(recommendations, details_list)):
        result_phs[i].empty()
        render_movie_card(result_phs[i], movie, details)