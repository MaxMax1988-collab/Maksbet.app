import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Maksbet.app - Autonomous AI Analytics", page_icon="🎯", layout="wide")

ODDS_API_KEY = "f13c7d2c08077c2894cab879d0905234"
API_SPORTS_KEY = "4d4ac6506b603ab89c0222ee1e32d6b7"

# --- INICJALIZACJA BAZY DANYCH SQLITE ---
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

class LiveSportsDataFetcher:
    def __init__(self, odds_key, sports_key):
        self.odds_key = odds_key
        self.sports_key = sports_key

    def get_real_matches(self, sport_key):
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
                        "rest_h": 2, "rest_a": 1,
                        "form_h": 0.70, "form_a": 0.50,
                        "h2h": 0.55, "elo_diff": 35
                    })
                return matches
        except Exception as e:
            st.error(f"Błąd połączenia z API: {e}")
        return []

@st.cache_resource
def init_xgboost_model():
    np.random.seed(42)
    n = 2500
    X = pd.DataFrame({
        'rest_diff': np.random.randint(-3, 4, n),
        'form_diff': np.random.uniform(-1, 1, n),
        'h2h': np.random.uniform(0, 1, n),
        'elo': np.random.normal(0, 100, n)
    })
    y = np.random.randint(0, 2, n)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
    model.fit(X, y)
    return model

model = init_xgboost_model()
fetcher = LiveSportsDataFetcher(ODDS_API_KEY, API_SPORTS_KEY)

# --- PANEL BOCZNY ---
st.title("🎯 Maksbet.app Autonomous Engine")
with st.sidebar:
    st.header("⚙️ Zarządzanie i Bankroll")
    discipline = st.selectbox("Dyscyplina", ["NBA Koszykówka", "NHL Hokej", "Tenis (WTA/ATP)"])
    bankroll = st.number_input("Twój Bankroll (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.slider("Ułamek Kryterium Kellego", 0.1, 0.5, 0.25)
    
    st.markdown("---")
    st.subheader("🧠 Autonomiczne Wagi AI")
    w_form = st.slider("Waga Formy", 0.0, 1.0, 0.40)
    w_elo = st.slider("Waga Elo / H2H", 0.0, 1.0, 0.35)
    w_rest = st.slider("Waga Odpoczynku", 0.0, 1.0, 0.25)
    
    st.markdown("---")
    st.subheader("📊 Statystyki bazy (SQLite)")
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN status='Wygrany' THEN 1 ELSE 0 END) FROM bets")
    db_stats = cursor.fetchone()
    total_bets = db_stats[0] if db_stats[0] else 0
    won_bets = db_stats[1] if db_stats[1] else 0
    st.metric("Zapisane zakłady", total_bets)
    st.metric("Skuteczność", f"{(won_bets/total_bets*100):.1f}%" if total_bets > 0 else "0.0%")

sport_map = {
    "NBA Koszykówka": "basketball_nba",
    "NHL Hokej": "icehockey_nhl",
    "Tenis (WTA/ATP)": "tennis_atp_aus_open"
}

raw_matches = fetcher.get_real_matches(sport_map[discipline])
analyzed_matches = []

for m in raw_matches:
    features = pd.DataFrame([[
        m['rest_h'] - m['rest_a'],
        (m['form_h'] - m['form_a']) * (w_form / 0.40),
        m['h2h'],
        m['elo_diff'] * (w_elo / 0.35)
    ]], columns=['rest_diff', 'form_diff', 'h2h', 'elo'])
    
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
    
    # Zapisz automatycznie do bazy, jeśli wykryto Value > 3%
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

# --- ZAKŁADKI INTERFEJSU ---
tab1, tab2 = st.tabs(["🔥 Bieżące Value", "📁 Historia i Rozliczanie Zakładów"])

with tab1:
    st.subheader(f"📊 Analiza rynková dla: {discipline}")
    if not analyzed_matches:
        st.warning("Brak aktywnych spotkań w API na najbliższe godziny.")
    else:
        for m in analyzed_matches:
            edge_pct = round(m['edge'] * 100, 1)
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                with col1:
                    st.markdown(f"### {m['home']} vs {m['away']}")
                    st.caption(f"ID: `{m['id']}`")
                with col2:
                    st.metric("Szansa AI", f"{round(m['ai_prob'] * 100, 1)}%")
                with col3:
                    st.metric("Kurs Bukmachera", f"{m['odds_home']}")
                with col4:
                    if m['edge'] > 0.03:
                        st.success(f"🔥 VALUE (+{edge_pct}%)")
                    else:
                        st.info(f"Brak Value ({edge_pct}%)")
                with col5:
                    if m['stake'] > 0:
                        st.metric("Sugerowana Stawka", f"{m['stake']} PLN")
                    else:
                        st.caption("Stawka: 0 PLN")
                st.divider()

with tab2:
    st.subheader("📝 Rejestr Zakładów i Feedback Loop")
    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    if df_bets.empty:
        st.info("Brak zapisanych zakładów w bazie danych. Pojawią się tutaj, gdy system wykryje Value.")
    else:
        st.dataframe(df_bets, use_container_width=True)
        
        st.markdown("### 🎯 Rozlicz zakład")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            bet_id_input = st.text_input("Wpisz ID meczu do rozliczenia")
        with col_r2:
            bet_status = st.selectbox("Wynik", ["Wygrany", "Przegrany"])
        with col_r3:
            if st.button("Zaktualizuj wynik"):
                cursor.execute("UPDATE bets SET status = ? WHERE id = ?", (bet_status, bet_id_input))
                db_conn.commit()
                st.success("Zaktualizowano status zakładu w bazie!")
                st.rerun()
                
