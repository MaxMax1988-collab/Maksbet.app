import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from datetime import datetime, timedelta, timezone
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

# --- NOWOCZESNY DESIGN CSS ---
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
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
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
        border: none;
        color: white;
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

# --- POMOCNIK DO KONWERSJI CZASU NA POLSKI (CET/CEST) ---
def format_match_time(commence_time_str):
    try:
        dt_utc = datetime.strptime(commence_time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        dt_pl = dt_utc + timedelta(hours=2)
        return dt_pl.strftime("%d.%m, %H:%M")
    except Exception:
        return commence_time_str

# --- ADAPTIVE TEMPORAL WEIGHTING ENGINE ---
def get_adaptive_weights(discipline_name):
    current_month = datetime.now().month
    if discipline_name in ["🏀 NBA Koszykówka", "🏀 Euroliga Koszykówka"]:
        if current_month in [10, 11, 12]:
            return {
                "Zgranie / Net Rating": 0.25,
                "Forma Ostatnich Meczów": 0.15,
                "Przewaga Dom/Wyjazd": 0.15,
                "Siła Ataku": 0.20,
                "Siła Obrony": 0.15,
                "Ranking Elo": 0.10
            }
        elif current_month in [1, 2]:
            return {
                "Zmęczenie / Terminarz (b2b)": 0.30,
                "Forma Ostatnich Meczów": 0.20,
                "Siła Obrony": 0.20,
                "Przewaga Dom/Wyjazd": 0.12,
                "Ranking Elo": 0.10,
                "H2H": 0.08
            }
        else:
            return {
                "Clutch / Efficiency": 0.30,
                "Motywacja / Seeding": 0.25,
                "Zmęczenie / Terminarz": 0.20,
                "Forma Ostatnich Meczów": 0.15,
                "Ranking Elo": 0.10
            }
    elif discipline_name in ["🏒 NHL Hokej", "🏒 AHL Hokej"]:
        return {
            "Forma Ostatnich Meczów": 0.18,
            "Przewaga Dom/Wyjazd": 0.12,
            "Zmęczenie / Terminarz": 0.20,
            "Siła Ataku": 0.16,
            "Siła Obrony": 0.22,
            "Ranking Elo": 0.08,
            "H2H (Bezpośrednie)": 0.04
        }
    else:
        return {
            "Forma Ostatnich Meczów": 0.40,
            "Przewaga Dom/Wyjazd": 0.00,
            "Zmęczenie / Terminarz": 0.25,
            "Siła Ataku": 0.10,
            "Siła Obrony": 0.05,
            "Ranking Elo": 0.10,
            "H2H (Bezpośrednie)": 0.10
        }

# --- CACHOWANE ZAPYTANIA API (NAPRAWIONE OKNO CZASOWE) ---
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_odds_cached(discipline_name, odds_key):
    now_utc = datetime.now(timezone.utc)
    from_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Zakres na najbliższe 7 dni gwarantuje pobranie pełnej oferty bez pustych odpowiedzi
    to_time = (now_utc + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    sport_keys = []
    markets_param = "h2h"
    
    if discipline_name == "🏀 NBA Koszykówka":
        sport_keys = ["basketball_nba"]
        markets_param = "h2h,spreads"
    elif discipline_name == "🏀 Euroliga Koszykówka":
        sport_keys = ["basketball_euroleague"]
        markets_param = "h2h,spreads"
    elif discipline_name == "🏒 NHL Hokej":
        sport_keys = ["icehockey_nhl"]
    elif discipline_name == "🏒 AHL Hokej":
        sport_keys = ["icehockey_ahl"]
    elif discipline_name == "🎾 Tenis (WTA/ATP)":
        try:
            sports_res = requests.get(f"https://api.the-odds-api.com/v4/sports/?apiKey={odds_key}", timeout=10)
            if sports_res.status_code == 200:
                all_sports = sports_res.json()
                sport_keys = [s['key'] for s in all_sports if s.get('group') == 'Tennis' and s.get('active')]
        except Exception:
            sport_keys = []
        
        if not sport_keys:
            sport_keys = ["tennis_atp_us_open", "tennis_wta_us_open"]

    all_matches = []
    
    for key in sport_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{key}/odds/?apiKey={odds_key}&regions=eu&markets={markets_param}&commenceTimeFrom={from_time}&commenceTimeTo={to_time}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                for match in data:
                    home = match.get('home_team')
                    away = match.get('away_team')
                    commence_time = match.get('commence_time')
                    
                    odds_home, odds_away = 1.90, 1.90
                    spread_home, spread_away = None, None
                    
                    bookmakers = match.get('bookmakers', [])
                    if bookmakers:
                        for m_type in bookmakers[0].get('markets', []):
                            if m_type.get('key') == 'h2h':
                                for o in m_type.get('outcomes', []):
                                    if o.get('name') == home:
                                        odds_home = o.get('price', 1.90)
                                    elif o.get('name') == away:
                                        odds_away = o.get('price', 1.90)
                            elif m_type.get('key') == 'spreads':
                                for o in m_type.get('outcomes', []):
                                    if o.get('name') == home:
                                        spread_home = o.get('point', 0.0)
                                    elif o.get('name') == away:
                                        spread_away = o.get('point', 0.0)

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
                        "spread_home": spread_home,
                        "spread_away": spread_away,
                        "form_diff": (implied_home - 0.5),
                        "home_away_adv": 0.05,
                        "fatigue_diff": 0.0,
                        "off_diff": (implied_home - 0.5) * 0.4,
                        "def_diff": -(implied_home - 0.5) * 0.4,
                        "elo_diff": (implied_home - 0.5) * 50,
                        "h2h": implied_home
                    })
        except Exception:
            continue
            
    return all_matches

class AutonomousDataFetcher:
    def __init__(self, odds_key):
        self.odds_key = odds_key

    def get_market_data(self, discipline_name):
        return fetch_odds_cached(discipline_name, self.odds_key)

    def auto_settle_bets(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT id, sport, odds, stake, best_pick, home_team, away_team FROM bets WHERE status = 'Oczekujący'")
        pending_bets = cursor.fetchall()
        
        if not pending_bets:
            return

        sport_key_map = {
            "🏀 NBA Koszykówka": "basketball_nba",
            "🏀 Euroliga Koszykówka": "basketball_euroleague",
            "🏒 NHL Hokej": "icehockey_nhl",
            "🏒 AHL Hokej": "icehockey_ahl",
            "🎾 Tenis (WTA/ATP)": "tennis_atp_us_open"
        }

        for bet in pending_bets:
            bet_id, sport, odds, stake, best_pick, home_team, away_team = bet
            s_key = sport_key_map.get(sport, "icehockey_nhl")
            
            scores_url = f"https://api.the-odds-api.com/v4/sports/{s_key}/scores/?apiKey={self.odds_key}&daysFrom=3"
            try:
                res = requests.get(scores_url, timeout=10)
                if res.status_code == 200:
                    matches_data = res.json()
                    for match in matches_data:
                        if match.get('id') == bet_id and match.get('completed'):
                            scores = match.get('scores', [])
                            if scores:
                                winner = None
                                max_score = -999
                                for s in scores:
                                    try:
                                        sc_val = float(s.get('score', 0))
                                        if sc_val > max_score:
                                            max_score = sc_val
                                            winner = s.get('name')
                                    except Exception:
                                        continue
                                        
                                if winner:
                                    is_won = (best_pick.lower() in winner.lower()) or (winner.lower() in best_pick.lower())
                                    status = "Wygrany" if is_won else "Przegrany"
                                    profit = round(stake * (odds - 1), 2) if is_won else round(-stake, 2)
                                        
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
        'h2h': np.random.uniform(0.3, 0.7, n)
    })
    
    score = (X['form_diff'] * 0.8 + X['elo_diff']/50 * 0.5 + np.random.normal(0, 0.3, n))
    y = np.where(score > 0, 1, 0)
    
    model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.02, random_state=42)
    model.fit(X, y)
    return model

model = init_self_learning_xgboost()
fetcher = AutonomousDataFetcher(ODDS_API_KEY)
fetcher.auto_settle_bets(db_conn)

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🎯 Maksbet.app")
    st.caption("Autonomous Quant Engine")
    st.markdown("---")
    
    discipline = st.selectbox(
        "Dyscyplina", 
        [
            "🏀 NBA Koszykówka", 
            "🏀 Euroliga Koszykówka", 
            "🏒 NHL Hokej", 
            "🏒 AHL Hokej", 
            "🎾 Tenis (WTA/ATP)"
        ]
    )
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=1.0, max_value=15.0, value=5.0, step=0.5) / 100.0
    
    st.markdown("---")
    if st.button("🔄 Wymusź Auto-Rozliczenie"):
        with st.spinner("Pobieranie wyników i rozliczanie zakładów..."):
            fetcher.auto_settle_bets(db_conn)
        st.success("Zakończono synchronizację!")
        st.rerun()

    if st.button("🧹 Wyczyść Cache API"):
        st.cache_data.clear()
        st.success("Wyczyszczono pamięć podręczną API!")
        st.rerun()

    st.markdown("---")
    current_weights = get_adaptive_weights(discipline)
    st.markdown(f"**🧠 Adaptacyjne Wagi AI**")
    st.caption(f"({discipline})")
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
        spread_val = m['spread_home']
    else:
        best_pick = m['away']
        best_prob = ai_prob_away
        best_odds = odds_away
        best_edge = edge_away
        spread_val = m['spread_away']
    
    b = best_odds - 1
    p = best_prob
    q = 1 - p
    kelly_f = (b * p - q) / b if b > 0 else 0
    stake = max(0, bankroll * kelly_f * kelly_fraction) if best_edge > min_edge_slider else 0
    
    match_id = m['id']
    
    pick_label = best_pick
    if spread_val is not None:
        sign = "+" if spread_val > 0 else ""
        pick_label = f"{best_pick} ({sign}{spread_val})"

    match_name = f"{m['home']} vs {m['away']}"
    
    if best_edge > min_edge_slider and stake > 0:
        cursor.execute("SELECT id FROM bets WHERE id = ?", (match_id,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Oczekujący', 0.0, ?, ?, ?)",
                (match_id, datetime.now().strftime("%Y-%m-%d"), discipline, match_name, best_odds, best_prob, best_edge, stake, m['home'], m['away'], pick_label)
            )
            db_conn.commit()

    m_copy = m.copy()
    m_copy.update({
        "best_pick": pick_label,
        "ai_prob": best_prob,
        "odds": best_odds,
        "edge": best_edge,
        "stake": round(stake, 2),
        "local_time": format_match_time(m.get('commence_time', ''))
    })
    analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY ---
st.markdown("## ⚡ Dashboard Kwantowy")
tab1, tab2, tab3 = st.tabs(["🔥 Sygnały Value", "📁 Rejestr, ROI & Heatmapa", "📈 Krzywa Kapitału"])

with tab1:
    st.subheader(f"Terminal Analityczny: {discipline}")
    st.caption(f"Filtrowanie aktywne: Minimalny Edge >= {int(min_edge_slider*100)}% + Czas polski (CET/CEST)")
    
    if not analyzed_matches:
        st.warning("Brak spotkań w ofercie API dla tej dyscypliny w tej chwili.")
    else:
        for m in analyzed_matches:
            edge_pct = round(m['edge'] * 100, 1)
            prob_pct = round(m['ai_prob'] * 100, 1)
            
            with st.container():
                st.markdown(f"""
                    <div class="value-card">
                        <h3>{m['home']} vs {m['away']}</h3>
                        <p style="color: #94a3b8; font-size: 12px;">Start (czas pl): {m['local_time']}</p>
                        <p>🎯 <b>Typ AI / Handicap:</b> <span style="color: #60a5fa;">{m['best_pick']}</span></p>
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
            label="📄 Pobierz Raport Kwantowy (TXT)",
            data=report_buffer.getvalue(),
            file_name=f"Maksbet_Raport_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    
    if df_bets.empty:
        st.info("Brak zapisanych zakładów w bazie danych. Baza jest czysta i gotowa na autonomiczne typy.")
    else:
        total_bets = len(df_bets)
        settled_bets = df_bets[df_bets['status'] != 'Oczekujący']
        won_bets = df_bets[df_bets['status'] == 'Wygrany']
        
        winrate = (len(won_bets) / len(settled_bets) * 100) if len(settled_bets) > 0 else 0.0
        total_profit = df_bets['profit'].sum() if 'profit' in df_bets.columns else 0.0
        total_staked = df_bets['stake'].sum() if 'stake' in df_bets.columns else 0.0
        
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        
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
        
        st.markdown("### 📊 Heatmapa Skuteczności i Zysku według Przedziałów Kursowych")
        if not settled_bets.empty:
            def categorize_odds(odds):
                if odds < 1.50: return "1.00 - 1.49"
                elif odds < 1.80: return "1.50 - 1.79"
                elif odds < 2.10: return "1.80 - 2.09"
                elif odds < 2.50: return "2.10 - 2.49"
                else: return "2.50+"
                
            settled_bets['odds_range'] = settled_bets['odds'].apply(categorize_odds)
            range_summary = settled_bets.groupby('odds_range').agg(
                Liczba=('id', 'count'),
                Zysk=('profit', 'sum'),
                Postawiono=('stake', 'sum')
            ).reset_index()
            range_summary['ROI (%)'] = np.where(range_summary['Postawiono'] > 0, 
                                                round((range_summary['Zysk'] / range_summary['Postawiono']) * 100, 2), 0.0)
            st.dataframe(range_summary, use_container_width=True)
        else:
            st.info("Heatmapa kursowa uzupełni się automatycznie po zakończeniu i rozliczeniu pierwszych meczów przez API.")

        st.markdown("---")
        st.dataframe(df_bets, use_container_width=True)

with tab3:
    st.subheader("📈 Krzywa Wzrostu Kapitału")
    df_bets = pd.read_sql("SELECT date, profit FROM bets WHERE status != 'Oczekujący'", db_conn)
    
    if df_bets.empty or 'profit' not in df_bets.columns or len(df_bets) == 0:
        st.info("Wykres wygeneruje się automatycznie po rozegraniu i rozliczeniu pierwszych meczów w tle przez system.")
    else:
        df_bets['cumulative_profit'] = df_bets['profit'].cumsum() + bankroll
        chart = alt.Chart(df_bets).mark_line(strokeWidth=3, color='#34d399').encode(
            x='date:T',
            y='cumulative_profit:Q',
            tooltip=['date', 'cumulative_profit']
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
