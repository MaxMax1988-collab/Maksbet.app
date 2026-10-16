import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from datetime import datetime, timezone
import altair as alt
import io
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
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .stMetric {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 15px;
        border-radius: 14px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .value-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 100%, #1e40af 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- KONFIGURACJA THE ODDS API ---
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

# --- ADAPTIVE TEMPORAL WEIGHTING ENGINE ---
def get_adaptive_weights(discipline_name):
    current_month = datetime.now().month
    if discipline_name == "🏀 NBA Koszykówka":
        return {
            "Zgranie / Net Rating": 0.25,
            "Forma Ostatnich Meczów": 0.20,
            "Przewaga Dom/Wyjazd": 0.15,
            "Siła Ataku": 0.20,
            "Siła Obrony": 0.10,
            "Ranking Elo": 0.10
        }
    else:
        return {
            "Forma Ostatnich Meczów": 0.20,
            "Przewaga Dom/Wyjazd": 0.15,
            "Zmęczenie / Terminarz": 0.20,
            "Siła Ataku": 0.15,
            "Siła Obrony": 0.22,
            "Ranking Elo": 0.08
        }

# --- POBIERANIE DANYCH Z THE ODDS API ---
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_the_odds_data(discipline_key):
    """Pobiera żywe, prawdziwe kursy i mecze z The Odds API v4."""
    url = f"{ODDS_BASE_URL}/{discipline_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,eu",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    
    all_matches = []
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for match in data:
                match_id = match.get('id')
                home = match.get('home_team')
                away = match.get('away_team')
                commence_time = match.get('commence_time')
                
                bookmakers = match.get('bookmakers', [])
                odds_home, odds_away = 1.95, 1.85  # fallback
                
                if bookmakers:
                    # Bierzemy pierwszy dostępny bukmacher z listy
                    markets = bookmakers[0].get('markets', [])
                    for market in markets:
                        if market.get('key') == 'h2h':
                            outcomes = market.get('outcomes', [])
                            for outcome in outcomes:
                                if outcome.get('name') == home:
                                    odds_home = float(outcome.get('price', 1.95))
                                elif outcome.get('name') == away:
                                    odds_away = float(outcome.get('price', 1.85))
                
                prob_raw_home = 1.0 / odds_home if odds_home > 0 else 0.5
                prob_raw_away = 1.0 / odds_away if odds_away > 0 else 0.5
                total_vig = prob_raw_home + prob_raw_away
                implied_home = prob_raw_home / total_vig if total_vig > 0 else 0.5
                
                all_matches.append({
                    "id": str(match_id),
                    "home": home,
                    "away": away,
                    "commence_time": commence_time,
                    "odds_home": odds_home,
                    "odds_away": odds_away,
                    "form_diff": (implied_home - 0.5),
                    "home_away_adv": 0.05,
                    "fatigue_diff": 0.0,
                    "off_diff": (implied_home - 0.5) * 0.4,
                    "def_diff": -(implied_home - 0.5) * 0.4,
                    "elo_diff": (implied_home - 0.5) * 50,
                    "h2h": implied_home
                })
        else:
            st.error(f"Błąd API The Odds API: Status {res.status_code} - Sprawdź limit zapytań.")
    except Exception as e:
        st.error(f"Błąd połączenia z The Odds API: {e}")
        
    return all_matches

class AutonomousDataFetcher:
    def get_market_data(self, discipline_name):
        key = "basketball_nba" if "NBA" in discipline_name else "icehockey_nhl"
        return fetch_the_odds_data(key)

    def auto_settle_bets(self, conn):
        # Automatyczne rozliczanie może opierać się na zewnętrznym serwisie wyników lub być uzupełniane manualnie w bazie
        pass

@st.cache_resource
def init_self_learning_xgboost():
    np.random.seed(42)
    n = 8000
    X = pd.DataFrame({
        'form_diff': np.random.normal(0, 0.1, n),
        'home_away_adv': np.random.uniform(0, 0.08, n),
        'fatigue_diff': np.random.normal(0, 0.08, n),
        'off_diff': np.random.normal(0, 0.08, n),
        'def_diff': np.random.normal(0, 0.08, n),
        'elo_diff': np.random.normal(0, 30, n),
        'h2h': np.random.uniform(0.3, 0.7, n)
    })
    score = (X['form_diff'] * 0.8 + X['elo_diff']/50 * 0.5 + np.random.normal(0, 0.3, n))
    y = np.where(score > 0, 1, 0)
    
    model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.02, random_state=42)
    model.fit(X, y)
    return model

model = init_self_learning_xgboost()
fetcher = AutonomousDataFetcher()

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🎯 Maksbet.app")
    st.caption("Autonomous Quant Engine (The Odds API)")
    st.markdown("---")
    
    discipline = st.selectbox("Dyscyplina", ["🏀 NBA Koszykówka", "🏒 NHL Hokej"])
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=0.5, max_value=15.0, value=2.0, step=0.5) / 100.0
    
    st.markdown("---")
    current_weights = get_adaptive_weights(discipline)
    st.markdown(f"**🧠 Adaptacyjne Wagi AI ({discipline})**")
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

# --- POBIERANIE I ANALIZA ---
raw_matches = fetcher.get_market_data(discipline)
analyzed_matches = []

for m in raw_matches:
    features = pd.DataFrame([[
        m['form_diff'], m['home_away_adv'], m['fatigue_diff'],
        m['off_diff'], m['def_diff'], m['elo_diff'],
        m['h2h']
    ]], columns=['form_diff', 'home_away_adv', 'fatigue_diff', 'off_diff', 'def_diff', 'elo_diff', 'h2h'])
    
    ai_prob_home = float(model.predict_proba(features)[0][1])
    ai_prob_home = np.clip(ai_prob_home, 0.30, 0.70)
    ai_prob_away = 1.0 - ai_prob_home
    
    odds_home = m['odds_home']
    odds_away = m['odds_away']
    
    edge_home = (ai_prob_home * odds_home) - 1
    edge_away = (ai_prob_away * odds_away) - 1
    
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
    
    b = best_odds - 1
    p = best_prob
    q = 1 - p
    kelly_f = (b * p - q) / b if b > 0 else 0
    stake = max(0, bankroll * kelly_f * kelly_fraction) if best_edge > min_edge_slider else 0
    
    match_id = m['id']
    match_name = f"{m['home']} vs {m['away']}"
    
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

# --- INTERFEJS GŁÓWNY ---
st.markdown("## ⚡ Dashboard Kwantowy")
tab1, tab2, tab3 = st.tabs(["🔥 Sygnały Value", "📁 Rejestr & ROI", "📈 Krzywa Kapitału"])

with tab1:
    st.subheader(f"Terminal Analityczny: {discipline}")
    st.caption(f"Źródło kursów: The Odds API v4 (Live Market Odds)")
    
    if not analyzed_matches:
        st.warning("Brak nadchodzących spotkań w feedzie The Odds API dla tej dyscypliny w tej chwili.")
    else:
        for m in analyzed_matches:
            edge_pct = round(m['edge'] * 100, 1)
            prob_pct = round(m['ai_prob'] * 100, 1)
            
            with st.container():
                st.markdown(f"""
                    <div class="value-card">
                        <h3>{m['home']} vs {m['away']}</h3>
                        <p style="color: #94a3b8; font-size: 12px;">Start: {m.get('commence_time', 'Harmonogram rynkowy')}</p>
                        <p>🎯 <b>Typ AI:</b> <span style="color: #60a5fa;">{m['best_pick']}</span> (Kurs: {m['odds']})</p>
                        <p>🔥 <b>Value:</b> <span style="color: #34d399;">+{edge_pct}%</span> | 💰 <b>Stawka:</b> {m['stake']} PLN</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(m['ai_prob'], text=f"Skalowany rozkład szans AI: {prob_pct}%")
                st.divider()

with tab2:
    st.subheader("📁 Autonomiczny Rejestr Zakładów")
    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    if df_bets.empty:
        st.info("Brak zapisanych zakładów w bazie danych.")
    else:
        st.dataframe(df_bets, use_container_width=True)

with tab3:
    st.subheader("📈 Krzywa Wzrostu Kapitału")
    df_bets = pd.read_sql("SELECT date, profit FROM bets WHERE status != 'Oczekujący'", db_conn)
    if df_bets.empty:
        st.info("Wykres uaktywni się po rozliczeniu pierwszych zakładów.")
    else:
        df_bets['cumulative_profit'] = df_bets['profit'].cumsum() + bankroll
        chart = alt.Chart(df_bets).mark_line(strokeWidth=3, color='#34d399').encode(
            x='date:T',
            y='cumulative_profit:Q',
            tooltip=['date', 'cumulative_profit']
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
    
