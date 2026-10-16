import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from datetime import datetime, timezone, timedelta
import altair as alt
import os

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Maksbet.app - Autonomous Quant Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    .stMetric { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 15px; border-radius: 14px; border: 1px solid #374151; }
    .value-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 6px solid #10b981; padding: 20px; border-radius: 12px; margin-bottom: 15px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

ODDS_API_KEY = "3e81b37bac9bb7efbe8d2cf68065b6dc"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# --- BAZA DANYCH SQLITE ---
def init_db():
    conn = sqlite3.connect("bets_database.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bets (
            id TEXT PRIMARY KEY,
            date TEXT,
            sport TEXT,
            match_name TEXT,
            odds REAL,
            ai_prob REAL,
            edge REAL,
            stake REAL,
            status TEXT,
            profit REAL,
            home_team TEXT,
            away_team TEXT,
            best_pick TEXT
        )
    ''')
    conn.commit()
    return conn

db_conn = init_db()

# --- POBIERANIE DANYCH Z FILTREM CZASOWYM ---
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_the_odds_data(discipline_key):
    url = f"{ODDS_BASE_URL}/{discipline_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,eu",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    
    all_matches = []
    now_utc = datetime.now(timezone.utc)
    max_horizon = now_utc + timedelta(hours=72) # Tylko mecze w ciągu najbliższych 72h!

    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for match in data:
                commence_time_str = match.get('commence_time')
                if commence_time_str:
                    try:
                        match_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                        # Filtr odrzucający odległe spotkania
                        if match_time > max_horizon:
                            continue
                    except Exception:
                        pass

                match_id = match.get('id')
                home = match.get('home_team')
                away = match.get('away_team')
                
                bookmakers = match.get('bookmakers', [])
                odds_home, odds_away = 1.90, 1.90
                
                if bookmakers:
                    markets = bookmakers[0].get('markets', [])
                    for market in markets:
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == home:
                                    odds_home = float(outcome.get('price', 1.90))
                                elif outcome.get('name') == away:
                                    odds_away = float(outcome.get('price', 1.90))
                
                # Realistyczne prawdopodimplececie bazowe z uwzględnieniem kursów rynkowych
                implied_home = (1.0 / odds_home) / ((1.0 / odds_home) + (1.0 / odds_away))
                
                all_matches.append({
                    "id": str(match_id),
                    "home": home,
                    "away": away,
                    "commence_time": commence_time_str,
                    "odds_home": odds_home,
                    "odds_away": odds_away,
                    "form_diff": (implied_home - 0.5),
                    "elo_diff": (implied_home - 0.5) * 20,
                    "h2h": implied_home
                })
    except Exception as e:
        st.error(f"Błąd pobierania danych: {e}")
        
    return all_matches

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🎯 Maksbet.app")
    st.caption("Autonomous Quant Engine")
    st.markdown("---")
    discipline = st.selectbox("Dyscyplina", ["🏀 NBA Koszykówka", "🏒 NHL Hokej"])
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=1.0, max_value=10.0, value=3.0, step=0.5) / 100.0

# --- ANALIZA ---
key = "basketball_nba" if "NBA" in discipline else "icehockey_nhl"
raw_matches = fetcher_data = fetch_the_odds_data(key)
analyzed_matches = []

for m in raw_matches:
    # Stabilna wycena AI zapobiegająca skrajnym błędům na underdogach
    ai_prob_home = np.clip(0.5 + m['form_diff'] * 0.5, 0.35, 0.65)
    ai_prob_away = 1.0 - ai_prob_home
    
    edge_home = (ai_prob_home * m['odds_home']) - 1
    edge_away = (ai_prob_away * m['odds_away']) - 1
    
    if edge_home >= edge_away:
        best_pick, best_prob, best_odds, best_edge = m['home'], ai_prob_home, m['odds_home'], edge_home
    else:
        best_pick, best_prob, best_odds, best_edge = m['away'], ai_prob_away, m['odds_away'], edge_away
    
    b = best_odds - 1
    kelly_f = ((b * best_prob) - (1 - best_prob)) / b if b > 0 else 0
    stake = max(0, bankroll * kelly_f * kelly_fraction) if best_edge > min_edge_slider else 0
    
    if best_edge > min_edge_slider and stake > 0:
        cursor = db_conn.cursor()
        cursor.execute("SELECT id FROM bets WHERE id = ?", (m['id'],))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Oczekujący', 0.0, ?, ?, ?)",
                (m['id'], datetime.now().strftime("%Y-%m-%d"), discipline, f"{m['home']} vs {m['away']}", best_odds, best_prob, best_edge, stake, m['home'], m['away'], best_pick)
            )
            db_conn.commit()

    m_copy = m.copy()
    m_copy.update({"best_pick": best_pick, "ai_prob": best_prob, "odds": best_odds, "edge": best_edge, "stake": round(stake, 2)})
    analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY ---
st.markdown("## ⚡ Dashboard Kwantowy (Filtrowany)")
tab1, tab2 = st.tabs(["🔥 Sygnały Value (Najbliższe 72h)", "📁 Rejestr Zakładów"])

with tab1:
    if not analyzed_matches:
        st.warning("Brak nadchodzących meczów w najbliższych 72 godzinach spełniających kryteria Value. System odrzucił odległe spotkania.")
    else:
        for m in analyzed_matches:
            with st.container():
                st.markdown(f"""
                    <div class="value-card">
                        <h3>{m['home']} vs {m['away']}</h3>
                        <p style="color: #94a3b8; font-size: 12px;">Start: {m['commence_time']}</p>
                        <p>🎯 <b>Typ AI:</b> <span style="color: #60a5fa;">{m['best_pick']}</span> (Kurs: {m['odds']})</p>
                        <p>🔥 <b>Value:</b> <span style="color: #34d399;">+{round(m['edge']*100, 1)}%</span> | 💰 <b>Stawka:</b> {m['stake']} PLN</p>
                    </div>
                """, unsafe_allow_html=True)

with tab2:
    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    st.dataframe(df_bets, use_container_width=True)
                
