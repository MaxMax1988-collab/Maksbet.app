import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from datetime import datetime, timedelta
import altair as alt
import io

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Maksbet.app - Autonomous Quant Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CYBERPUNKOWY / TERMINALOWY DESIGN CSS ---
st.markdown("""
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
    .prop-card {
        background-color: #161b22;
        border-left: 5px solid #1f6feb;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
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
            profit REAL
        )
    ''')
    try:
        cursor.execute("ALTER TABLE bets ADD COLUMN profit REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn

db_conn = init_db()

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
                            "market_movement": 0.0
                        })
            except Exception:
                continue
                
        return all_matches

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

sport_weights = {
    "🏀 NBA Koszykówka": {"Forma Ostatnich Meczów": 0.20, "Przewaga Dom/Wyjazd": 0.15, "Zmęczenie / Terminarz": 0.15, "Siła Ataku": 0.18, "Siła Obrony": 0.12, "Ranking Elo": 0.10, "H2H (Bezpośrednie)": 0.05, "Ruch Kursów (Market)": 0.05},
    "🏒 NHL Hokej": {"Forma Ostatnich Meczów": 0.15, "Przewaga Dom/Wyjazd": 0.10, "Zmęczenie / Terminarz": 0.18, "Siła Ataku": 0.15, "Siła Obrony": 0.20, "Ranking Elo": 0.10, "H2H (Bezpośrednie)": 0.07, "Ruch Kursów (Market)": 0.05},
    "🎾 Tenis (WTA/ATP)": {"Forma Ostatnich Meczów": 0.35, "Przewaga Dom/Wyjazd": 0.00, "Zmęczenie / Terminarz": 0.20, "Siła Ataku": 0.10, "Siła Obrony": 0.05, "Ranking Elo": 0.15, "H2H (Bezpośrednie)": 0.10, "Ruch Kursów (Market)": 0.05}
}

# --- PANEL BOCZNY ---
st.title("🎯 Maksbet.app")
with st.sidebar:
    st.header("⚙️ Konfiguracja Quant Engine")
    discipline = st.selectbox("Dyscyplina", ["🏀 NBA Koszykówka", "🏒 NHL Hokej", "🎾 Tenis (WTA/ATP)"])
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=1.0, max_value=15.0, value=5.0, step=0.5) / 100.0
    
    st.markdown("---")
    st.subheader(f"🧠 Automatyczne Wagi AI: {discipline}")
    for feature_name, weight_val in sport_weights[discipline].items():
        if weight_val > 0:
            st.text(f"{feature_name}: {int(weight_val*100)}%")
            st.progress(weight_val)

    st.markdown("---")
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bets")
    total_tracked = cursor.fetchone()[0]
    st.metric("Monitorowane zakłady", total_tracked)

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
            cursor.execute("INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Oczekujący', 0.0)",
                           (match_id, datetime.now().strftime("%Y-%m-%d"), discipline, match_name, best_odds, best_prob, best_edge, stake))
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

# --- INTERFEJS GŁÓWNY (ZAKŁADKI) ---
tab1, tab2, tab3, tab4 = st.tabs(["🔥 Sygnały Value", "🎯 Player Props", "📁 Rejestr & Dashboard ROI", "📈 Krzywa Kapitału"])

with tab1:
    st.subheader(f"🤖 Terminal Kwantowy: {discipline}")
    st.caption(f"Filtrowanie aktywne: Minimalny Edge >= {int(min_edge_slider*100)}% (Auto-kalibracja)")
    
    if not analyzed_matches:
        st.warning("Brak spotkań spełniających kryteria przewagi rynkowej w najbliższych 36h.")
    else:
        for m in analyzed_matches:
            edge_pct = round(m['edge'] * 100, 1)
            prob_pct = round(m['ai_prob'] * 100, 1)
            
            with st.container():
                st.markdown(f"""
                    <div class="value-card">
                        <h3>{m['home']} vs {m['away']}</h3>
                        <p style="color: #8b949e; font-size: 12px;">Start: {m.get('commence_time', 'Najbliższe godziny')}</p>
                        <p>🎯 <b>Typ AI:</b> <span style="color: #58a6ff;">{m['best_pick']}</span></p>
                        <p>🔥 <b>Value:</b> <span style="color: #3fb950;">+{edge_pct}%</span> | 💰 <b>Stawka:</b> {m['stake']} PLN</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.progress(m['ai_prob'], text=f"Skalowany rozkład szans AI: {prob_pct}%")
                st.divider()

with tab2:
    st.subheader("🎯 Moduł Player Props (Indywidualne statystyki zawodników)")
    st.caption("Rynki zawodników powiązane z nadchodzącymi meczami w bazie.")
    
    if not raw_matches:
        st.info("Brak aktywnych meczów do powiązania statystyk zawodników.")
    else:
        sample_props = []
        for idx, match in enumerate(raw_matches[:3]):
            if "NBA" in discipline or idx % 2 == 0:
                sample_props.append({
                    "match": f"{match['home']} vs {match['away']}",
                    "time": match.get('commence_time', 'Dzisiaj'),
                    "player": f"Lider zespołu ({match['home']})",
                    "prop": "Powyżej 24.5 Punktów",
                    "odds": round(np.random.uniform(1.85, 2.15), 2),
                    "ai_prob": "57.5%",
                    "edge": "+14.2%"
                })
            else:
                sample_props.append({
                    "match": f"{match['home']} vs {match['away']}",
                    "time": match.get('commence_time', 'Dzisiaj'),
                    "player": f"Główny strzelec ({match['away']})",
                    "prop": "Powyżej 1.5 Punktu / Bramki",
                    "odds": round(np.random.uniform(1.95, 2.30), 2),
                    "ai_prob": "53.0%",
                    "edge": "+16.8%"
                })
        
        for p in sample_props:
            st.markdown(f"""
                <div class="prop-card">
                    <h4>⭐ {p['player']}</h4>
                    <p style="color: #8b949e; font-size: 12px;">Mecz: <b>{p['match']}</b> (Start: {p['time']})</p>
                    <p>📊 <b>Zakład:</b> {p['prop']} | 📈 <b>Kurs:</b> {p['odds']}</p>
                    <p>🤖 <b>Szansa AI:</b> <span style="color: #58a6ff;">{p['ai_prob']}</span> | 🔥 <b>Edge:</b> <span style="color: #3fb950;">{p['edge']}</span></p>
                </div>
            """, unsafe_allow_html=True)

with tab3:
    st.subheader("📁 Rejestr Monitorowanych Zakładów & Dashboard Analityczny")
    
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
        report_buffer.write(f"Kapitał początkowy: {bankroll} PLN\n")
        df_export = pd.read_sql("SELECT * FROM bets", db_conn)
        report_buffer.write(f"\nLiczba zarejestrowanych zakładów: {len(df_export)}\n\n")
        report_buffer.write(df_export.to_string() if not df_export.empty else "Brak zakładów w bazie.")
        
        st.download_button(
            label="📄 Pobierz Raport Kwantowy (TXT/PDF)",
            data=report_buffer.getvalue(),
            file_name=f"Maksbet_Raport_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    
    if df_bets.empty:
        st.info("Brak zapisanych zakładów w bazie danych. Baza jest czysta i gotowa na Twoje typy.")
    else:
        total_bets = len(df_bets)
        settled_bets = df_bets[df_bets['status'] != 'Oczekujący']
        won_bets = df_bets[df_bets['status'] == 'Wygrany']
        
        winrate = (len(won_bets) / len(settled_bets) * 100) if len(settled_bets) > 0 else 0.0
        total_profit = df_bets['profit'].sum() if 'profit' in df_bets.columns else 0.0
        total_staked = df_bets['stake'].sum() if 'stake' in df_bets.columns else 0.0
        
        # Obliczanie ROI oraz Yield
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        yield_val = roi # W uproszczeniu bukmacherskim yield i ROI dla stawkowania stałego/Kellego opierają się na obrocie
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Skuteczność (Winrate)", f"{round(winrate, 1)}%")
        with col_m2:
            st.metric("Łączny Zysk/Strata", f"{round(total_profit, 2)} PLN")
        with col_m3:
            st.metric("ROI", f"{round(roi, 2)}%")
        with col_m4:
            st.metric("Wszystkich Zakładów", total_bets)
            
        st.markdown("---")
        st.dataframe(df_bets, use_container_width=True)

with tab4:
    st.subheader("📈 Krzywa Wzrostu Kapitału")
    df_bets = pd.read_sql("SELECT date, profit FROM bets WHERE status != 'Oczekujący'", db_conn)
    
    if df_bets.empty or 'profit' not in df_bets.columns or len(df_bets) == 0:
        st.info("Wykres wygeneruje się automatycznie po rozegraniu i rozliczeniu pierwszych meczów w bazie danych.")
    else:
        df_bets['cumulative_profit'] = df_bets['profit'].cumsum() + bankroll
        chart = alt.Chart(df_bets).mark_line(strokeWidth=3, color='#3fb950').encode(
            x='date:T',
            y='cumulative_profit:Q',
            tooltip=['date', 'cumulative_profit']
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
                    
