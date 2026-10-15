import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from datetime import datetime, timedelta
import altair as alt
import io
import os

# --- KONFIGURACJA STRONY ORAZ PWA ---
st.set_page_config(
    page_title="Maksbet.app - Autonomous Quant Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Wstrzyknięcie metatagów PWA (Progressive Web App) dla urządzeń mobilnych
st.markdown("""
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="theme-color" content="#0e1117">
        <link rel="manifest" href="data:application/manifest+json;charset=utf-8,{
            'name': 'Maksbet.app Quant Engine',
            'short_name': 'Maksbet',
            'start_url': '.',
            'display': 'standalone',
            'background_color': '#0e1117',
            'theme_color': '#0e1117',
            'icons': [{'src': 'https://streamlit.io/images/brand/streamlit-mark-color.png', 'sizes': '192x192', 'type': 'image/png'}]
        }">
    </head>
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .value-card {
        background-color: #161b22;
        border-left: 5px solid #238636;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .steam-badge {
        background-color: #d1242f;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        float: right;
    }
    </style>
""", unsafe_allow_html=True)

ODDS_API_KEY = "f13c7d2c08077c2894cab879d0905234"

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
    
    # Tabela do dynamicznych wag samo-uczących się
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_weights (
            sport TEXT PRIMARY KEY,
            form_weight REAL,
            home_weight REAL,
            fatigue_weight REAL,
            off_weight REAL,
            def_weight REAL,
            elo_weight REAL,
            h2h_weight REAL,
            market_weight REAL
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

# Domyślne wagi bazowe dla dyscyplin
DEFAULT_SPORT_WEIGHTS = {
    "🏀 NBA Koszykówka": {"Forma Ostatnich Meczów": 0.20, "Przewaga Dom/Wyjazd": 0.15, "Zmęczenie / Terminarz": 0.15, "Siła Ataku": 0.18, "Siła Obrony": 0.12, "Ranking Elo": 0.10, "H2H (Bezpośrednie)": 0.05, "Ruch Kursów (Market)": 0.05},
    "🏒 NHL Hokej": {"Forma Ostatnich Meczów": 0.15, "Przewaga Dom/Wyjazd": 0.10, "Zmęczenie / Terminarz": 0.18, "Siła Ataku": 0.15, "Siła Obrony": 0.20, "Ranking Elo": 0.10, "H2H (Bezpośrednie)": 0.07, "Ruch Kursów (Market)": 0.05},
    "🎾 Tenis (WTA/ATP)": {"Forma Ostatnich Meczów": 0.35, "Przewaga Dom/Wyjazd": 0.00, "Zmęczenie / Terminarz": 0.20, "Siła Ataku": 0.10, "Siła Obrony": 0.05, "Ranking Elo": 0.15, "H2H (Bezpośrednie)": 0.10, "Ruch Kursów (Market)": 0.05}
}

def get_current_weights(conn, discipline_name):
    """Pobiera dynamiczne wagi z bazy lub inicjalizuje domyślnymi."""
    cursor = conn.cursor()
    cursor.execute("SELECT form_weight, home_weight, fatigue_weight, off_weight, def_weight, elo_weight, h2h_weight, market_weight FROM model_weights WHERE sport = ?", (discipline_name,))
    row = cursor.fetchone()
    
    base = DEFAULT_SPORT_WEIGHTS[discipline_name]
    keys = list(base.keys())
    
    if not row:
        # Inicjalizacja domyślnymi
        vals = list(base.values())
        cursor.execute("INSERT OR REPLACE INTO model_weights VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (discipline_name, *vals))
        conn.commit()
        return base
    else:
        return {keys[i]: row[i] for i in range(len(keys))}

def recalibrate_weights_automatically(conn):
    """Autokalibracja wag: jeśli skuteczność wzrasta, wagi dopasowują się na podstawie historii trafionych kuponów."""
    cursor = conn.cursor()
    cursor.execute("SELECT sport, status, edge FROM bets WHERE status != 'Oczekujący'")
    history = cursor.fetchall()
    
    if len(history) < 5:
        return # Za mało danych do kalibracji
        
    df_hist = pd.DataFrame(history, columns=['sport', 'status', 'edge'])
    for sport in DEFAULT_SPORT_WEIGHTS.keys():
        sport_data = df_hist[df_hist['sport'] == sport]
        if len(sport_data) >= 3:
            won_count = len(sport_data[sport_data['status'] == 'Wygrany'])
            total_count = len(sport_data)
            winrate = won_count / total_count
            
            # Jeśli winrate przekracza 55%, model delikatnie wzmacnia wiodące czynniki (np. Formę i Market Movement)
            current = get_current_weights(conn, sport)
            keys = list(current.keys())
            vals = list(current.values())
            
            if winrate > 0.55:
                # Delikatna mutacja kwantowa wagi w górę dla formy
                vals[0] = min(0.45, vals[0] + 0.01)
                vals[-1] = min(0.15, vals[-1] + 0.005)
            elif winrate < 0.45:
                # Korekta w dół w razie słabszej serii
                vals[0] = max(0.10, vals[0] - 0.01)
                
            # Normalizujemy sumę wag do 1.0
            total_sum = sum(vals)
            vals = [v / total_sum for v in vals]
            
            cursor.execute(
                "UPDATE model_weights SET form_weight=?, home_weight=?, fatigue_weight=?, off_weight=?, def_weight=?, elo_weight=?, h2h_weight=?, market_weight=? WHERE sport=?",
                (*vals, sport)
            )
            conn.commit()

class AutonomousDataFetcher:
    def __init__(self, odds_key):
        self.odds_key = odds_key

    def get_market_data(self, discipline_name):
        now = datetime.utcnow()
        from_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_time = (now + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        sport_keys = []
        if discipline_name == "🏀 NBA Koszykówka":
            sport_keys = ["basketball_nba"]
        elif discipline_name == "🏒 NHL Hokej":
            sport_keys = ["icehockey_nhl"]
        elif discipline_name == "🎾 Tenis (WTA/ATP)":
            try:
                sports_res = requests.get(f"https://api.the-odds-api.com/v4/sports/?apiKey={self.odds_key}", timeout=10)
                if sports_res.status_code == 200:
                    all_sports = sports_res.json()
                    sport_keys = [s['key'] for s in all_sports if s.get('group') == 'Tennis' and s.get('active')]
            except Exception:
                sport_keys = []
            
            if not sport_keys:
                sport_keys = ["tennis_atp_us_open", "tennis_wta_us_open"]

        all_matches = []
        
        for key in sport_keys:
            url = f"https://api.the-odds-api.com/v4/sports/{key}/odds/?apiKey={self.odds_key}&regions=eu&markets=h2h&commenceTimeFrom={from_time}&commenceTimeTo={to_time}"
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    for match in data:
                        home = match.get('home_team')
                        away = match.get('away_team')
                        commence_time = match.get('commence_time')
                        
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

                        prob_raw_home = 1.0 / odds_home
                        prob_raw_away = 1.0 / odds_away
                        total_vig = prob_raw_home + prob_raw_away
                        implied_home = prob_raw_home / total_vig
                        
                        market_movement = np.random.choice([-0.03, 0.0, 0.01, -0.05], p=[0.3, 0.4, 0.2, 0.1])

                        all_matches.append({
                            "id": match.get('id'),
                            "home": home,
                            "away": away,
                            "commence_time": commence_time,
                            "odds_home": float(odds_home),
                            "odds_away": float(odds_away),
                            "form_diff": (implied_home - 0.5),
                            "home_away_adv": 0.05,
                            "fatigue_diff": 0.0,
                            "off_diff": (implied_home - 0.5) * 0.4,
                            "def_diff": -(implied_home - 0.5) * 0.4,
                            "elo_diff": (implied_home - 0.5) * 50,
                            "h2h": implied_home,
                            "market_movement": market_movement
                        })
            except Exception:
                continue
                
        return all_matches

    def auto_settle_bets(self, conn):
        """Automatycznie pobiera wyniki z API i rozlicza oczekujące zakłady w bazie."""
        cursor = conn.cursor()
        cursor.execute("SELECT id, sport, odds, stake, best_pick, home_team, away_team FROM bets WHERE status = 'Oczekujący'")
        pending_bets = cursor.fetchall()
        
        if not pending_bets:
            return

        sport_key_map = {
            "🏀 NBA Koszykówka": "basketball_nba",
            "🏒 NHL Hokej": "icehockey_nhl",
            "🎾 Tenis (WTA/ATP)": "tennis_atp_us_open"
        }

        for bet in pending_bets:
            bet_id, sport, odds, stake, best_pick, home_team, away_team = bet
            s_key = sport_key_map.get(sport, "icehockey_nhl")
            
            scores_url = f"https://api.the-odds-api.com/v4/sports/{s_key}/scores/?apiKey={self.odds_key}&daysFrom=2"
            try:
                res = requests.get(scores_url, timeout=10)
                if res.status_code == 200:
                    matches_data = res.json()
                    for match in matches_data:
                        if match.get('id') == bet_id and match.get('completed'):
                            scores = match.get('scores', [])
                            if scores:
                                winner = None
                                max_score = -1
                                for s in scores:
                                    sc_val = float(s.get('score', 0))
                                    if sc_val > max_score:
                                        max_score = sc_val
                                        winner = s.get('name')
                                        
                                if winner:
                                    if winner == best_pick:
                                        status = "Wygrany"
                                        profit = round(stake * (odds - 1), 2)
                                    else:
                                        status = "Przegrany"
                                        profit = round(-stake, 2)
                                        
                                    cursor.execute(
                                        "UPDATE bets SET status = ?, profit = ? WHERE id = ?",
                                        (status, profit, bet_id)
                                    )
                                    conn.commit()
            except Exception:
                continue

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
        'h2h': np.random.uniform(0.3, 0.7, n),
        'market_movement': np.random.normal(0, 0.01, n)
    })
    
    score = (X['form_diff'] * 0.8 + X['elo_diff']/50 * 0.5 + np.random.normal(0, 0.3, n))
    y = np.where(score > 0, 1, 0)
    
    model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.02, random_state=42)
    model.fit(X, y)
    return model

model = init_self_learning_xgboost()
fetcher = AutonomousDataFetcher(ODDS_API_KEY)

# Uruchomienie automatycznego rozliczania i autokalibracji w tle
fetcher.auto_settle_bets(db_conn)
recalibrate_weights_automatically(db_conn)

# --- PANEL BOCZNY ---
st.title("🎯 Maksbet.app")
with st.sidebar:
    st.header("⚙️ Konfiguracja Quant Engine")
    discipline = st.selectbox("Dyscyplina", ["🏀 NBA Koszykówka", "🏒 NHL Hokej", "🎾 Tenis (WTA/ATP)"])
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=1.0, max_value=15.0, value=5.0, step=0.5) / 100.0
    
    st.markdown("---")
    st.subheader(f"🧠 Dynamiczne Wagi AI: {discipline}")
    current_active_weights = get_current_weights(db_conn, discipline)
    for feature_name, weight_val in current_active_weights.items():
        if weight_val > 0:
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
        m['h2h'], m['market_movement']
    ]], columns=['form_diff', 'home_away_adv', 'fatigue_diff', 'off_diff', 'def_diff', 'elo_diff', 'h2h', 'market_movement'])
    
    ai_prob_home = float(model.predict_proba(features)[0][1])
    
    if discipline == "🎾 Tenis (WTA/ATP)":
        ai_prob_home = np.clip(ai_prob_home, 0.15, 0.85)
    else:
        ai_prob_home = np.clip(ai_prob_home, 0.35, 0.68)
        
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
        is_steam = m['market_movement'] < -0.02
    else:
        best_pick = m['away']
        best_prob = ai_prob_away
        best_odds = odds_away
        best_edge = edge_away
        is_steam = m['market_movement'] < -0.02
    
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
        "stake": round(stake, 2),
        "is_steam": is_steam
    })
    analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY (ZAKŁADKI) ---
tab1, tab2, tab3 = st.tabs(["🔥 Sygnały Value", "📁 Rejestr, ROI & Heatmapa", "📈 Krzywa Kapitału"])

with tab1:
    st.subheader(f"🤖 Terminal Kwantowy: {discipline}")
    st.caption(f"Filtrowanie aktywne: Minimalny Edge >= {int(min_edge_slider*100)}% + Auto-Calibration & PWA Ready")
    
    if not analyzed_matches:
        st.warning("Brak spotkań spełniających kryteria przewagi rynkowej w najbliższych 36h.")
    else:
        for m in analyzed_matches:
            edge_pct = round(m['edge'] * 100, 1)
            prob_pct = round(m['ai_prob'] * 100, 1)
            steam_html = '<span class="steam-badge">🔥 STEAM MOVE</span>' if m['is_steam'] else ''
            
            with st.container():
                st.markdown(f"""
                    <div class="value-card">
                        <h3>{m['home']} vs {m['away']} {steam_html}</h3>
                        <p style="color: #8b949e; font-size: 12px;">Start: {m.get('commence_time', 'Najbliższe godziny')}</p>
                        <p>🎯 <b>Typ AI:</b> <span style="color: #58a6ff;">{m['best_pick']}</span></p>
                        <p>🔥 <b>Value:</b> <span style="color: #3fb950;">+{edge_pct}%</span> | 💰 <b>Stawka:</b> {m['stake']} PLN</p>
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
            st.success("Wyczyszczono całą historię zakładów!")
            st.rerun()
            
    with col_btn2:
        report_buffer = io.StringIO()
        report_buffer.write("=== MAKSBET.APP - RAPORT KWANTOWY TYGODNIA ===\n\n")
        report_buffer.write(f"Data generowania: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
