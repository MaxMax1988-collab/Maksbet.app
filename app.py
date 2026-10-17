import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timezone, timedelta
import altair as alt
import io
import os
import re

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Maksbet.app - Autonomous Quant Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NOWOCZESNY DESIGN CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    .stMetric { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 15px; border-radius: 14px; border: 1px solid #374151; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2); }
    .value-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 6px solid #10b981; padding: 20px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); border: 1px solid #334155; }
    .stButton>button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; border-radius: 8px; border: none; padding: 8px 16px; font-weight: 600; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3); }
    .stButton>button:hover { background: linear-gradient(135deg, #2563eb 100%, #1e40af 100%); border: none; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- KONFIGURACJA THE ODDS API ---
ODDS_API_KEY = "3e81b37bac9bb7efbe8d2cf68065b6dc"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# --- HELPER: NORMALIZACJA NAZW DRUŻYN ---
def normalize_team_name(name):
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    return " ".join(name.split())

def find_best_stats_match(team_name, stats_dict):
    norm_target = normalize_team_name(team_name)
    if not norm_target or not stats_dict:
        return None
    
    for key, val in stats_dict.items():
        if normalize_team_name(key) == norm_target:
            return val
            
    target_words = [w for w in norm_target.split() if len(w) > 3]
    for key, val in stats_dict.items():
        norm_key = normalize_team_name(key)
        for tw in target_words:
            if tw in norm_key:
                return val
    return None

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
    for col_def in [
        ("profit", "REAL DEFAULT 0.0"),
        ("home_team", "TEXT"),
        ("away_team", "TEXT"),
        ("best_pick", "TEXT")
    ]:
        try:
            cursor.execute(f"ALTER TABLE bets ADD COLUMN {col_def[0]} {col_def[1]}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn

db_conn = init_db()

# --- MODUŁ POBIERANIA STATYSTYK NHL ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nhl_live_stats():
    try:
        url = "https://api-web.nhle.com/v1/standings/now"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            stats_dict = {}
            for team in data.get("standings", []):
                name_en = team.get("teamName", {}).get("default", "")
                place_en = team.get("placeName", {}).get("default", "")
                full_name = f"{place_en} {name_en}".strip()
                
                wins = team.get("wins", 0)
                losses = team.get("losses", 0)
                ot_losses = team.get("otLosses", 0)
                total_games = wins + losses + ot_losses
                win_pct = (wins / total_games) if total_games > 0 else 0.5
                
                goals_for = team.get("goalFor", 0)
                goals_against = team.get("goalAgainst", 0)
                gf_g = (goals_for / total_games) if total_games > 0 else 3.0
                ga_g = (goals_against / total_games) if total_games > 0 else 3.0
                
                team_data = {"win_pct": win_pct, "gf_g": gf_g, "ga_g": ga_g}
                stats_dict[full_name] = team_data
                stats_dict[name_en] = team_data
            return stats_dict
    except Exception:
        pass
    return {}

nhl_stats_cache = fetch_nhl_live_stats()

# --- MODUŁ POBIERANIA STATYSTYK AHL (HockeyTech) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ahl_live_stats():
    try:
        url = "https://lscluster.hockeytech.com/feed/index.php"
        params = {
            "feed": "modulekit",
            "view": "statviewtype",
            "type": "standings",
            "client_code": "ahl",
            "key": "446521baf8c38984",
            "fmt": "json"
        }
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            stats_dict = {}
            teams_list = []
            def extract_teams(obj):
                if isinstance(obj, dict):
                    if "team_name" in obj or "name" in obj:
                        teams_list.append(obj)
                    for k, v in obj.items():
                        extract_teams(v)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_teams(item)
            extract_teams(data)
                
            for t in teams_list:
                name = t.get("team_name", t.get("name", ""))
                if name:
                    wins = float(t.get("wins", t.get("w", 20)))
                    losses = float(t.get("losses", t.get("l", 20)))
                    ot_losses = float(t.get("ot_losses", t.get("ot", 5)))
                    tot_g = wins + losses + ot_losses
                    win_pct = (wins / tot_g) if tot_g > 0 else 0.5
                    
                    gf = float(t.get("goals_for", t.get("gf", 100)))
                    ga = float(t.get("goals_against", t.get("ga", 100)))
                    gf_g = (gf / tot_g) if tot_g > 0 else 3.0
                    ga_g = (ga / tot_g) if tot_g > 0 else 3.0
                    
                    stats_dict[name] = {"win_pct": win_pct, "gf_g": gf_g, "ga_g": ga_g}
            return stats_dict
    except Exception:
        pass
    return {}

ahl_stats_cache = fetch_ahl_live_stats()

# --- MODUŁ POBIERANIA STATYSTYK NBA ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nba_advanced_stats():
    try:
        from nba_api.stats.endpoints import leaguestandings
        standings = leaguestandings.LeagueStandings().get_data_frames()[0]
        stats_dict = {}
        for _, row in standings.iterrows():
            city = str(row.get('TeamCity', ''))
            name = str(row.get('TeamName', ''))
            full_name = f"{city} {name}".strip()
            win_pct = float(row.get('WinPCT', 0.5))
            stats_dict[full_name] = {"win_pct": win_pct}
            stats_dict[name] = {"win_pct": win_pct}
        return stats_dict
    except Exception:
        return {}

nba_stats_cache = fetch_nba_advanced_stats()

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🎯 Maksbet.app")
    st.caption("Autonomous Quant Engine")
    st.markdown("---")
    
    discipline = st.selectbox("Dyscyplina", ["🏀 NBA Koszykówka", "🏒 NHL Hokej", "🏒 AHL Hokej"])
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=0.5, max_value=15.0, value=1.5, step=0.5) / 100.0
    
    st.markdown("---")
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bets")
    total_tracked = cursor.fetchone()[0]
    st.metric("Monitorowane zakłady", total_tracked)
    
    # DIAGNOSTYKA ŹRÓDŁA DANYCH
    if "AHL" in discipline:
        st.info(f"Pobrano drużyn AHL w cache: {len(ahl_stats_cache)}")

# --- POBIERANIE DANYCH Z THE ODDS API ---
if "NBA" in discipline:
    key = "basketball_nba"
elif "NHL" in discipline:
    key = "icehockey_nhl"
else:
    key = "icehockey_ahl"

raw_matches = fetch_the_odds_data(key) if 'fetch_the_odds_data' in globals() else []

# Jeśli funkcja pobierania nie była zdefiniowana w locie, wklejam pełny handler odpytywania
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_odds(d_key):
    url = f"{ODDS_BASE_URL}/{d_key}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "us,eu", "markets": "h2h", "oddsFormat": "decimal"}
    all_m = []
    now_utc = datetime.now(timezone.utc)
    max_h = now_utc + timedelta(hours=72)
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            for match in res.json():
                commence = match.get('commence_time')
                if commence:
                    try:
                        m_time = datetime.fromisoformat(commence.replace('Z', '+00:00'))
                        if m_time > max_h: continue
                    except: pass
                match_id = match.get('id')
                home, away = match.get('home_team'), match.get('away_team')
                oh, oa = 1.90, 1.90
                books = match.get('bookmakers', [])
                if books:
                    for mkt in books[0].get('markets', []):
                        if mkt.get('key') == 'h2h':
                            for out in mkt.get('outcomes', []):
                                if out.get('name') == home: oh = float(out.get('price', 1.90))
                                elif out.get('name') == away: oa = float(out.get('price', 1.90))
                all_m.append({"id": str(match_id), "home": home, "away": away, "commence_time": commence, "odds_home": oh, "odds_away": oa})
    except: pass
    return all_m

raw_matches = fetch_odds(key)
analyzed_matches = []

for m in raw_matches:
    match_id = m['id']
    match_name = f"{m['home']} vs {m['away']}"
    odds_home, odds_away = m['odds_home'], m['odds_away']
    
    # Baza No-Vig
    prob_raw_h = 1.0 / odds_home if odds_home > 0 else 0.5
    prob_raw_a = 1.0 / odds_away if odds_away > 0 else 0.5
    tot_v = prob_raw_h + prob_raw_a
    implied_h = prob_raw_h / tot_v if tot_v > 0 else 0.5
    
    # Delta statystyczna z twardym zakresem bezpieczeństwa (Max ±2%)
    stat_delta = 0.0
    if "NBA" in discipline:
        h_s = find_best_stats_match(m['home'], nba_stats_cache)
        a_s = find_best_stats_match(m['away'], nba_stats_cache)
        if h_s and a_s: stat_delta = np.clip((h_s["win_pct"] - a_s["win_pct"]) * 0.02, -0.02, 0.02)
    elif "NHL" in discipline:
        h_s = find_best_stats_match(m['home'], nhl_stats_cache)
        a_s = find_best_stats_match(m['away'], nhl_stats_cache)
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            stat_delta = np.clip(wd * 0.015, -0.02, 0.02)
    else: # AHL - Silne zacieśnienie do bazy rynkowej
        h_s = find_best_stats_match(m['home'], ahl_stats_cache)
        a_s = find_best_stats_match(m['away'], ahl_stats_cache)
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            stat_delta = np.clip(wd * 0.01, -0.015, 0.015)
            
    ai_prob_home = np.clip(implied_h + stat_delta, 0.15, 0.85)
    ai_prob_away = 1.0 - ai_prob_home
    
    edge_home = (ai_prob_home * odds_home) - 1.0
    edge_away = (ai_prob_away * odds_away) - 1.0
    
    if edge_home >= edge_away:
        best_pick, best_prob, best_odds, best_edge = m['home'], ai_prob_home, odds_home, edge_home
    else:
        best_pick, best_prob, best_odds, best_edge = m['away'], ai_prob_away, odds_away, edge_away
    
    b = best_odds - 1.0
    kelly_f = (b * best_prob - (1.0 - best_prob)) / b if b > 0 else 0
    stake = max(0, bankroll * kelly_f * kelly_fraction) if best_edge > min_edge_slider else 0
    
    if best_edge > min_edge_slider and stake > 0:
        cursor.execute("SELECT id FROM bets WHERE id = ?", (match_id,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Oczekujący', 0.0, ?, ?, ?)",
                (match_id, datetime.now().strftime("%Y-%m-%d"), discipline, match_name, best_odds, best_prob, best_edge, stake, m['home'], m['away'], best_pick)
            )
            db_conn.commit()

    m_copy = m.copy()
    m_copy.update({"best_pick": best_pick, "ai_prob": best_prob, "odds": best_odds, "edge": best_edge, "stake": round(stake, 2)})
    analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY ---
st.markdown("## ⚡ Dashboard Kwantowy")
tab1, tab2, tab3 = st.tabs(["🔥 Sygnały Value", "📁 Rejestr, ROI & Heatmapa", "📈 Krzywa Kapitału"])

with tab1:
    st.subheader(f"Terminal Analityczny: {discipline}")
    st.caption("Źródło: Oficjalne API + Rynek No-Vig Strict Anchoring")
    
    if not analyzed_matches:
        st.warning("Brak nadchodzących spotkań spełniających kryteria rynkowe.")
    else:
        for m in analyzed_matches:
            edge_pct = round(m['edge'] * 100, 1)
            prob_pct = round(m['ai_prob'] * 100, 1)
            with st.container():
                st.markdown(f"""
                    <div class="value-card">
                        <h3>{m['home']} vs {m['away']}</h3>
                        <p style="color: #94a3b8; font-size: 12px;">Start: {m.get('commence_time', 'Najbliższe godziny')}</p>
                        <p>🎯 <b>Typ AI:</b> <span style="color: #60a5fa;">{m['best_pick']}</span> (Kurs: {m['odds']})</p>
                        <p>🔥 <b>Value:</b> <span style="color: #34d399;">+{edge_pct}%</span> | 💰 <b>Stawka:</b> {m['stake']} PLN</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(m['ai_prob'], text=f"Rozkład szans AI: {prob_pct}%")
                st.divider()

with tab2:
    st.subheader("📁 Rejestr Zakładów & ROI")
    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    if df_bets.empty: st.info("Brak zapisanych zakładów.")
    else: st.dataframe(df_bets, use_container_width=True)

with tab3:
    st.subheader("📈 Krzywa Kapitału")
    df_bets = pd.read_sql("SELECT date, profit FROM bets WHERE status != 'Oczekujący'", db_conn)
    if df_bets.empty: st.info("Wykres uaktywni się po rozliczeniu.")
    else:
        df_bets['cumulative_profit'] = df_bets['profit'].cumsum() + bankroll
        chart = alt.Chart(df_bets).mark_line(strokeWidth=3, color='#34d399').encode(x='date:T', y='cumulative_profit:Q').properties(height=350)
        st.altair_chart(chart, use_container_width=True)
                
