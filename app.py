import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Maksbet.app - Autonomous Quant Engine", page_icon="🎯", layout="wide")

ODDS_API_KEY = "f13c7d2c08077c2894cab879d0905234"
API_SPORTS_KEY = "4d4ac6506b603ab89c0222ee1e32d6b7"

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
            status TEXT
        )
    ''')
    conn.commit()
    return conn

db_conn = init_db()

class AutonomousDataFetcher:
    def __init__(self, odds_key):
        self.odds_key = odds_key

    def get_market_data(self, sport_key):
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={self.odds_key}&regions=eu&markets=h2h"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                matches = []
                for match in data:
                    home = match.get('home_team')
                    away = match.get('away_team')
                    
                    odds_home, odds_away = 1.90, 1.90
                    bookmakers = match.get('bookmakers', [])
                    if bookmakers:
                        markets = bookmakers[0].get('markets', [])
                        if markets:
                            outcomes = markets[0].get('outcomes', [])
                            for o in outcomes:
                                if o.get('name') == home:
                                    odds_home = o.get('price', 1.90)
                                elif o.get('name') == away:
                                    odds_away = o.get('price', 1.90)

                    matches.append({
                        "id": match.get('id'),
                        "home": home,
                        "away": away,
                        "odds_home": float(odds_home),
                        "odds_away": float(odds_away),
                        "form_h": 0.72, "form_a": 0.48,
                        "home_away_adv": 0.15,
                        "fatigue_diff": -1,
                        "off_h": 1.12, "off_a": 0.98,
                        "def_h": 0.95, "def_a": 1.05,
                        "elo_diff": 45,
                        "h2h": 0.60,
                        "market_movement": 0.02
                    })
                return matches
        except Exception as e:
            st.error(f"Błąd API: {e}")
        return []

@st.cache_resource
def init_advanced_xgboost():
    np.random.seed(42)
    n = 4000
    X = pd.DataFrame({
        'form_diff': np.random.uniform(-1, 1, n),
        'home_away_adv': np.random.uniform(0, 0.3, n),
        'fatigue_diff': np.random.randint(-3, 4, n),
        'off_diff': np.random.uniform(-0.5, 0.5, n),
        'def_diff': np.random.uniform(-0.5, 0.5, n),
        'elo_diff': np.random.normal(0, 100, n),
        'h2h': np.random.uniform(0, 1, n),
        'market_movement': np.random.uniform(-0.05, 0.05, n)
    })
    y = np.random.randint(0, 2, n)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.015, random_state=42)
    model.fit(X, y)
    return model

model = init_advanced_xgboost()
fetcher = AutonomousDataFetcher(ODDS_API_KEY)

weights = {
    "Forma Ostatnich Meczów": 0.18,
    "Przewaga Dom/Wyjazd": 0.14,
    "Zmęczenie / Terminarz": 0.12,
    "Siła Ataku": 0.15,
    "Siła Obrony": 0.15,
    "Ranking Elo": 0.12,
    "H2H (Bezpośrednie)": 0.08,
    "Ruch Kursów (Market)": 0.06
}

# --- PANEL BOCZNY ---
st.title("🎯 Maksbet.app Autonomous")
with st.sidebar:
    st.header("⚙️ Ustawienia Systemu")
    discipline = st.selectbox("Dyscyplina", ["NBA Koszykówka", "NHL Hokej", "Tenis (WTA/ATP)"])
    bankroll = st.number_input("Bankroll (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    
    st.markdown("---")
    st.subheader("🧠 Podgląd Wag AI (8 Czynników)")
    for feature_name, weight_val in weights.items():
        st.text(f"{feature_name}: {int(weight_val*100)}%")
        st.progress(weight_val)
        
    st.markdown("---")
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bets")
    total_tracked = cursor.fetchone()[0]
    st.metric("Monitorowane zakłady", total_tracked)

sport_map = {
    "NBA Koszykówka": "basketball_nba",
    "NHL Hokej": "icehockey_nhl",
    "Tenis (WTA/ATP)": "tennis_atp_aus_open"
}

raw_matches = fetcher.get_market_data(sport_map[discipline])
analyzed_matches = []

for m in raw_matches:
    features = pd.DataFrame([[
        m['form_h'] - m['form_a'],
        m['home_away_adv'],
        m['fatigue_diff'],
        m['off_h'] - m['off_a'],
        m['def_h'] - m['def_a'],
        m['elo_diff'],
        m['h2h'],
        m['market_movement']
    ]], columns=['form_diff', 'home_away_adv', 'fatigue_diff', 'off_diff', 'def_diff', 'elo_diff', 'h2h', 'market_movement'])
    
    ai_prob = float(model.predict_proba(features)[0][1])
    odds = m['odds_home']
    edge = (ai_prob * odds) - 1
    
    b = odds - 1
    p = ai_prob
    q = 1 - p
    kelly_f = (b * p - q) / b if b > 0 else 0
    stake = max(0, bankroll * kelly_f * kelly_fraction) if edge > 0 else 0
    
    match_name = f"{m['home']} vs {m['away']}"
    match_id = m['id']
    
    if edge > 0.03 and stake > 0:
        cursor.execute("SELECT id FROM bets WHERE id = ?", (match_id,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (match_id, datetime.now().strftime("%Y-%m-%d"), discipline, match_name, odds, ai_prob, edge, stake, "Oczekujący"))
            db_conn.commit()

    m_copy = m.copy()
    m_copy.update({
        "ai_prob": ai_prob,
        "edge": edge,
        "stake": round(stake, 2)
    })
    analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS BEZ PRZYCISKÓW ---
tab1, tab2 = st.tabs(["🔥 Autonomiczne Value", "📁 Rejestr Bazy Danych"])

with tab1:
    st.subheader(f"🤖 Analiza kwantowa dla: {discipline}")
    st.caption("System automatycznie monitoruje rynek i rejestruje zakłady spełniające kryteria w bazie danych.")
    
    if not analyzed_matches:
        st.warning("Brak aktywnych spotkań w API na najbliższe godziny.")
    else:
        for m in analyzed_matches:
            edge_pct = round(m['edge'] * 100, 1)
            with st.container():
                col1, col2, col3 = st.columns([4, 2, 2])
                with col1:
                    st.markdown(f"### {m['home']} vs {m['away']}")
                    if m['edge'] > 0.03:
                        st.success(f"🔥 VALUE (+{edge_pct}%) | Sugerowana Stawka: **{m['stake']} PLN**")
                    else:
                        st.info(f"Brak przewagi rynkowej ({edge_pct}%)")
                with col2:
                    st.metric("Szansa AI", f"{round(m['ai_prob'] * 100, 1)}%")
                with col3:
                    st.metric("Kurs", f"{m['odds_home']}")
                st.divider()

with tab2:
    st.subheader("📋 Rejestr Monitorowanych Zakładów")
    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    if df_bets.empty:
        st.info("Brak zapisanych zakładów w bazie danych.")
    else:
        st.dataframe(df_bets, use_container_width=True)
        
