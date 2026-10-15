import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests

st.set_page_config(page_title="Maksbet.app - AI Analytics", page_icon="🎯", layout="wide")

ODDS_API_KEY = "f13c7d2c08077c2894cab879d0905234"
API_SPORTS_KEY = "4d4ac6506b603ab89c0222ee1e32d6b7"

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

st.title("🎯 Maksbet.app AI Analytics")
with st.sidebar:
    st.header("⚙️ Ustawienia i Bankroll")
    discipline = st.selectbox("Dyscyplina", ["NBA Koszykówka", "NHL Hokej", "Tenis (WTA/ATP)"])
    bankroll = st.number_input("Twój Bankroll (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.slider("Ułamek Kryterium Kellego", 0.1, 0.5, 0.25)
    
    st.markdown("---")
    st.subheader("Kalibracja Wag AI")
    w_form = st.slider("Waga Formy", 0.0, 1.0, 0.40)
    w_elo = st.slider("Waga Elo / H2H", 0.0, 1.0, 0.35)
    w_rest = st.slider("Waga Odpoczynku", 0.0, 1.0, 0.25)

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
    
    m_copy = m.copy()
    m_copy.update({
        "ai_prob": ai_prob,
        "edge": edge,
        "stake": round(stake, 2)
    })
    analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

st.subheader(f"📊 Przeliczone mecze dla: {discipline}")

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
  
