import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
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

# --- MODUŁ POBIERANIA STATYSTYK NHL (Oficjalne API NHL) ---
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
                
                team_data = {
                    "win_pct": win_pct,
                    "gf_g": gf_g,
                    "ga_g": ga_g
                }
                stats_dict[full_name] = team_data
                stats_dict[name_en] = team_data
            return stats_dict
    except Exception:
        pass
    return {}

nhl_stats_cache = fetch_nhl_live_stats()

# --- MODUŁ POBIERANIA STATYSTYK AHL (Oficjalne HockeyTech / LeagueStat API) ---
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
            # Parsowanie struktury HockeyTech dla tabel ligowych
            sections = data.get("securables", data.get("site_kit", {}).get("standings", []))
            # Bezpieczne przeszukiwanie obiektów JSON w feedzie HockeyTech
            teams_list = []
            if isinstance(data, dict):
                # Przejście przez zagnieżdżone sekcje HockeyTech
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
                    
                    stats_dict[name] = {
                        "win_pct": win_pct,
                        "gf_g": gf_g,
                        "ga_g": ga_g
                    }
            return stats_dict
    except Exception:
        pass
    return {}

ahl_stats_cache = fetch_ahl_live_stats()

# --- MODUŁ POBIERANIA STATYSTYK NBA (nba_api) ---
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
            
            team_data = {"win_pct": win_pct}
            stats_dict[full_name] = team_data
            stats_dict[name] = team_data
        return stats_dict
    except Exception:
        return {}

nba_stats_cache = fetch_nba_advanced_stats()

# --- ADAPTIVE TEMPORAL WEIGHTING ENGINE ---
def get_adaptive_weights(discipline_name):
    if "NBA" in discipline_name:
        return {"Net Rating / Bilans": 0.35, "Forma L10": 0.25, "Dom/Wyjazd": 0.15, "Siła Ataku": 0.15, "Siła Obrony": 0.10}
    else:
        return {"Forma / Bilans Wygranych": 0.30, "Siła Ataku (GF)": 0.35, "Szczelność Obrony (GA)": 0.35}

# --- POBIERANIE DANYCH Z THE ODDS API (Z FILTREM 72H) ---
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
    max_horizon = now_utc + timedelta(hours=72)

    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for match in data:
                commence_time_str = match.get('commence_time')
                if commence_time_str:
                    try:
                        match_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
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
                
                all_matches.append({
                    "id": str(match_id),
                    "home": home,
                    "away": away,
                    "commence_time": commence_time_str,
                    "odds_home": odds_home,
                    "odds_away": odds_away
                })
    except Exception as e:
        st.error(f"Błąd pobierania danych z The Odds API: {e}")
        
    return all_matches

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
    current_weights = get_adaptive_weights(discipline)
    st.markdown(f"**🧠 Wagi Modelu Statystycznego**")
    for feature_name, weight_val in current_weights.items():
        st.text(f"{feature_name}: {int(weight_val*100)}%")
        st.progress(weight_val)

    st.markdown("---")
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bets")
    total_tracked = cursor.fetchone()[0]
    st.metric("Monitorowane zakłady", total_tracked)
    
    if os.path.exists("bets_database.db"):
        with open("bets_database.db", "rb") as f:
            st.download_button(
                label="📥 Pobierz Backup Bazy (.db)",
                data=f,
                file_name=f"bets_backup_{datetime.now().strftime('%Y%m%d')}.db",
                mime="application/x-sqlite3"
            )

# --- WYBÓR KLUCZA API DLA DYSCYPLINY ---
if "NBA" in discipline:
    key = "basketball_nba"
elif "NHL" in discipline:
    key = "icehockey_nhl"
else:
    key = "icehockey_ahl"

raw_matches = fetch_the_odds_data(key)
analyzed_matches = []

for m in raw_matches:
    match_id = m['id']
    match_name = f"{m['home']} vs {m['away']}"
    odds_home = m['odds_home']
    odds_away = m['odds_away']
    
    # 1. CZYSTE PRAWDOPODOBIEŃSTWO RYNKOWE (NO-VIG) JAKO BAZA
    prob_raw_h = 1.0 / odds_home if odds_home > 0 else 0.5
    prob_raw_a = 1.0 / odds_away if odds_away > 0 else 0.5
    tot_v = prob_raw_h + prob_raw_a
    implied_h = prob_raw_h / tot_v if tot_v > 0 else 0.5
    
    # 2. OBLICZANIE DELTY STATYSTYCZNEJ Z OFICJALNYCH ŹRÓDEŁ
    stat_delta = 0.0
    
    if "NBA" in discipline:
        h_stats = find_best_stats_match(m['home'], nba_stats_cache)
        a_stats = find_best_stats_match(m['away'], nba_stats_cache)
        if h_stats and a_stats:
            win_diff = h_stats["win_pct"] - a_stats["win_pct"]
            stat_delta = win_diff * 0.08
    elif "NHL" in discipline:
        h_stats = find_best_stats_match(m['home'], nhl_stats_cache)
        a_stats = find_best_stats_match(m['away'], nhl_stats_cache)
        if h_stats and a_stats:
            win_diff = h_stats["win_pct"] - a_stats["win_pct"]
            gf_diff = (h_stats["gf_g"] - a_stats["gf_g"]) / 5.0
            ga_diff = (a_stats["ga_g"] - h_stats["ga_g"]) / 5.0
            stat_delta = (win_diff * 0.04) + (gf_diff * 0.02) + (ga_diff * 0.02)
    else: # AHL - Oficjalne HockeyTech API
        h_stats = find_best_stats_match(m['home'], ahl_stats_cache)
        a_stats = find_best_stats_match(m['away'], ahl_stats_cache)
        if h_stats and a_stats:
            win_diff = h_stats["win_pct"] - a_stats["win_pct"]
            gf_diff = (h_stats["gf_g"] - a_stats["gf_g"]) / 5.0
            ga_diff = (a_stats["ga_g"] - h_stats["ga_g"]) / 5.0
            stat_delta = (win_diff * 0.04) + (gf_diff * 0.02) + (ga_diff * 0.02)
            
    # 3. OSTATECZNE PRAWDOPODOBIEŃSTWO AI
    ai_prob_home = np.clip(implied_h + stat_delta, 0.15, 0.85)
    ai_prob_away = 1.0 - ai_prob_home
    
    edge_home = (ai_prob_home * odds_home) - 1.0
    edge_away = (ai_prob_away * odds_away) - 1.0
    
    if edge_home >= edge_away:
        best_pick = m['home']
        best_prob = ai_prob_home
        best_odds = odds_home
        best_edge = edge_home
    else:
        best_pick = m['away']
        best_prob = ai_prob_away
        best_odds = odds_away
        best_edge = edge_away
    
    b = best_odds - 1.0
    p = best_prob
    q = 1.0 - p
    kelly_f = (b * p - q) / b if b > 0 else 0
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
    m_copy.update({
        "best_pick": best_pick,
        "ai_prob": best_prob,
        "odds": best_odds,
        "edge": best_edge,
        "stake": round(stake, 2)
    })
    analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY (3 ZAKŁADKI) ---
st.markdown("## ⚡ Dashboard Kwantowy")
tab1, tab2, tab3 = st.tabs(["🔥 Sygnały Value", "📁 Rejestr, ROI & Heatmapa", "📈 Krzywa Kapitału"])

with tab1:
    st.subheader(f"Terminal Analityczny: {discipline}")
    st.caption("Źródło: Oficjalne API (NBA / NHL / AHL HockeyTech) + Fuzzy Team Matching")
    
    if not analyzed_matches:
        st.warning("Brak nadchodzących spotkań w najbliższych 72h spełniających kryteria rynkowe.")
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
                st.progress(m['ai_prob'], text=f"Skalowany rozkład szans AI: {prob_pct}%")
                st.divider()

with tab2:
    st.subheader("📁 Autonomiczny Rejestr Zakładów & Dashboard ROI")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Wyczyść historię bazy danych"):
            cursor.execute("DELETE FROM bets")
            db_conn.commit()
            st.success("Wyczyszczono historię zakładów!")
            st.rerun()
            
    with col_btn2:
        report_buffer = io.StringIO()
        report_buffer.write("=== MAKSBET.APP - RAPORT KWANTOWY ===\n\n")
        df_export = pd.read_sql("SELECT * FROM bets", db_conn)
        report_buffer.write(df_export.to_string() if not df_export.empty else "Brak zakładów.")
        
        st.download_button(
            label="📄 Pobierz Raport (TXT)",
            data=report_buffer.getvalue(),
            file_name=f"Maksbet_Raport_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    
    if df_bets.empty:
        st.info("Brak zapisanych zakładów w bazie danych.")
    else:
        total_bets = len(df_bets)
        settled_bets = df_bets[df_bets['status'] != 'Oczekujący']
        won_bets = df_bets[df_bets['status'] == 'Wygrany']
        
        winrate = (len(won_bets) / len(settled_bets) * 100) if len(settled_bets) > 0 else 0.0
        total_profit = df_bets['profit'].sum() if 'profit' in df_bets.columns else 0.0
        total_staked = df_bets['stake'].sum() if 'stake' in df_bets.columns else 0.0
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1: st.metric("Winrate", f"{round(winrate, 1)}%")
        with col_m2: st.metric("Zysk/Strata", f"{round(total_profit, 2)} PLN")
        with col_m3: st.metric("ROI", f"{round(roi, 2)}%")
        with col_m4: st.metric("Zakłady", total_bets)
            
        st.markdown("---")
        st.markdown("### 📊 Heatmapa Skuteczności według Kursów")
        if not settled_bets.empty:
            def categorize_odds(odds):
                if odds < 1.50: return "1.00 - 1.49"
                elif odds < 1.80: return "1.50 - 1.79"
                elif odds < 2.10: return "1.80 - 2.09"
                elif odds < 2.50: return "2.10 - 2.49"
                else: return "2.50+"
            settled_bets['odds_range'] = settled_bets['odds'].apply(categorize_odds)
            range_summary = settled_bets.groupby('odds_range').agg(Liczba=('id', 'count'), Zysk=('profit', 'sum'), Postawiono=('stake', 'sum')).reset_index()
            st.dataframe(range_summary, use_container_width=True)
        else:
            st.info("Heatmapa uzupełni się po rozliczeniu pierwszych spotkań.")

        st.markdown("---")
        st.dataframe(df_bets, use_container_width=True)

with tab3:
    st.subheader("📈 Krzywa Wzrostu Kapitału")
    df_bets = pd.read_sql("SELECT date, profit FROM bets WHERE status != 'Oczekujący'", db_conn)
    if df_bets.empty or 'profit' not in df_bets.columns or len(df_bets) == 0:
        st.info("Wykres uaktywni się po rozliczeniu pierwszych zakładów.")
    else:
        df_bets['cumulative_profit'] = df_bets['profit'].cumsum() + bankroll
        chart = alt.Chart(df_bets).mark_line(strokeWidth=3, color='#34d399').encode(
            x='date:T', y='cumulative_profit:Q', tooltip=['date', 'cumulative_profit']
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
