import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import psycopg2
from datetime import datetime, timezone, timedelta
import altair as alt

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Maksbet.app - Advanced NHL Quant Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    .stMetric { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 15px; border-radius: 14px; border: 1px solid #374151; }
    .value-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 6px solid #10b981; padding: 20px; border-radius: 12px; margin-bottom: 15px; border: 1px solid #334155; }
    .test-box { background: #111827; border: 1px solid #374151; padding: 15px; border-radius: 10px; margin-bottom: 10px; font-family: monospace; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
ODDS_API_KEY = "62e230538ba5e8f6ff0aa80fcdbb481c"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# --- POŁĄCZENIE Z BAZĄ DANYCH W CHMURZE (NEON.TECH) ---
def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        try:
            db_url = st.secrets["DATABASE_URL"]
        except Exception:
            st.error("Brak skonfigurowanego URL bazy danych w Secrets (DATABASE_URL)!")
            st.stop()
            
    if "&channel_binding=require" in db_url:
        db_url = db_url.replace("&channel_binding=require", "")
    elif "?channel_binding=require" in db_url:
        db_url = db_url.replace("?channel_binding=require", "")

    conn = psycopg2.connect(db_url)
    return conn

def init_db():
    conn = get_db_connection()
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
            best_pick TEXT,
            bet_type TEXT DEFAULT 'H2H'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weights_history (
            sport TEXT PRIMARY KEY,
            w_win_pct REAL,
            w_advanced REAL,
            w_form REAL,
            w_home REAL
        )
    ''')
    
    cursor.execute('''
        INSERT INTO weights_history (sport, w_win_pct, w_advanced, w_form, w_home)
        VALUES ('🏒 NHL Hokej', 0.25, 0.40, 0.20, 0.15)
        ON CONFLICT (sport) DO NOTHING
    ''')
        
    conn.commit()
    cursor.close()
    conn.close()

init_db()

def get_sport_weights():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT w_win_pct, w_advanced, w_form, w_home FROM weights_history WHERE sport = '🏒 NHL Hokej'")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        return {"win_pct": row[0], "advanced": row[1], "form": row[2], "home": row[3]}
    return {"win_pct": 0.25, "advanced": 0.40, "form": 0.20, "home": 0.15}

def update_weights_after_settlement(bet_won):
    conn = get_db_connection()
    cursor = conn.cursor()
    w = get_sport_weights()
    lr = 0.01 if bet_won else -0.01
    
    new_adv = max(0.10, w["advanced"] + lr)
    new_form = max(0.10, w["form"] + (lr * 0.5))
    total = w["win_pct"] + new_adv + new_form + w["home"]
    
    cursor.execute('''
        UPDATE weights_history 
        SET w_win_pct = %s, w_advanced = %s, w_form = %s, w_home = %s
        WHERE sport = '🏒 NHL Hokej'
    ''', (w["win_pct"]/total, new_adv/total, new_form/total, w["home"]/total))
    conn.commit()
    cursor.close()
    conn.close()

def auto_settle_from_api():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, sport, odds, stake, best_pick, home_team, away_team FROM bets WHERE status = 'Oczekujący'")
    pending_bets = cursor.fetchall()
    
    if not pending_bets:
        cursor.close()
        conn.close()
        return

    url = f"{ODDS_BASE_URL}/icehockey_nhl/scores"
    params = {"apiKey": ODDS_API_KEY, "daysFrom": 3}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            for match in res.json():
                if match.get("completed", False):
                    match_id = str(match.get("id"))
                    scores = match.get("scores", [])
                    if scores and len(scores) >= 2:
                        s1, s2 = int(scores[0].get("score", 0)), int(scores[1].get("score", 0))
                        t1, t2 = scores[0].get("name"), scores[1].get("name")
                        for bet in pending_bets:
                            bet_id, b_sport, odds, stake, best_pick, home, away = bet
                            if bet_id.split("_")[0] == match_id and b_sport == '🏒 NHL Hokej':
                                winner = t1 if s1 > s2 else t2
                                won = (best_pick.lower() in winner.lower() or winner.lower() in best_pick.lower())
                                status = 'Wygrany' if won else 'Przegrany'
                                profit = round((odds - 1.0) * stake, 2) if won else round(-stake, 2)
                                cursor.execute("UPDATE bets SET status = %s, profit = %s WHERE id = %s", (status, profit, bet_id))
                                conn.commit()
                                update_weights_after_settlement(won)
    except Exception:
        pass
    cursor.close()
    conn.close()

auto_settle_from_api()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nhl_stats():
    stats_dict = {}
    try:
        url = "http://site.api.espn.com/apis/v2/sports/hockey/nhl/standings"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for group in data.get("children", []):
                for team_entry in group.get("standings", {}).get("entries", []):
                    team_info = team_entry.get("team", {})
                    name, nickname = team_info.get("displayName", ""), team_info.get("name", "")
                    stats = team_entry.get("stats", [])
                    wins, losses, ot, gf, ga = 0, 0, 0, 0.0, 0.0
                    for s in stats:
                        if s.get("name") == "wins": wins = s.get("value", 0)
                        if s.get("name") == "losses": losses = s.get("value", 0)
                        if s.get("name") == "otLosses": ot = s.get("value", 0)
                        if s.get("name") in ["pointsFor", "goalsFor"]: gf = s.get("value", 0.0)
                        if s.get("name") in ["pointsAgainst", "goalsAgainst"]: ga = s.get("value", 0.0)
                    tot = wins + losses + ot
                    win_pct = (wins / tot) if tot > 0 else 0.5
                    stats_dict[name.lower()] = {"win_pct": win_pct, "gf_g": (gf/tot if tot > 0 else 3.0), "ga_g": (ga/tot if tot > 0 else 3.0)}
                    stats_dict[nickname.lower()] = {"win_pct": win_pct, "gf_g": (gf/tot if tot > 0 else 3.0), "ga_g": (ga/tot if tot > 0 else 3.0)}
    except Exception:
        pass
    return stats_dict

nhl_stats_cache = fetch_nhl_stats()

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🎯 Maksbet.app")
    st.caption("Advanced NHL Quant Engine")
    st.markdown("---")
    
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=0.0, max_value=10.0, value=0.5, step=0.5) / 100.0
    
    curr_weights = get_sport_weights()
    st.markdown("---")
    st.markdown("<b>🧠 Wyuczone Wagi NHL (Baza Neon):</b>", unsafe_allow_html=True)
    st.caption(f"・ Win %: {round(curr_weights['win_pct']*100, 1)}%")
    st.caption(f"・ Advanced Goals: {round(curr_weights['advanced']*100, 1)}%")
    st.caption(f"・ Forma: {round(curr_weights['form']*100, 1)}%")
    st.caption(f"・ Home Advantage: {round(curr_weights['home']*100, 1)}%")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_odds():
    url = f"{ODDS_BASE_URL}/icehockey_nhl/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            return res.json(), 200
        return [], res.status_code
    except Exception:
        return [], 500

raw_matches, status_code = fetch_odds()
analyzed_h2h = []

db_conn = get_db_connection()
cursor = db_conn.cursor()

def find_team_stats(team_name):
    if not nhl_stats_cache: return None
    t_lower = team_name.lower()
    if t_lower in nhl_stats_cache: return nhl_stats_cache[t_lower]
    for k, v in nhl_stats_cache.items():
        if k in t_lower or t_lower in k: return v
    return None

for m in raw_matches:
    match_id = m.get('id')
    home, away = m.get('home_team'), m.get('away_team')
    commence = m.get('commence_time')
    
    oh, oa = 1.90, 1.90
    books = m.get('bookmakers', [])
    if books:
        for mkt in books[0].get('markets', []):
            if mkt.get('key') == 'h2h':
                for out in mkt.get('outcomes', []):
                    if out.get('name') == home: oh = float(out.get('price', 1.90))
                    elif out.get('name') == away: oa = float(out.get('price', 1.90))
                    
    h_s = find_team_stats(home)
    a_s = find_team_stats(away)
    if not h_s or not a_s: continue

    prob_raw_h = 1.0 / oh if oh > 0 else 0.5
    prob_raw_a = 1.0 / oa if oa > 0 else 0.5
    tot_v = prob_raw_h + prob_raw_a
    implied_h = prob_raw_h / tot_v if tot_v > 0 else 0.5
    
    w = curr_weights
    wd = h_s.get("win_pct", 0.5) - a_s.get("win_pct", 0.5)
    gf_diff = (h_s.get("gf_g", 3.0) - a_s.get("ga_g", 3.0)) - (a_s.get("gf_g", 3.0) - h_s.get("ga_g", 3.0))
    stat_delta = (wd * w["win_pct"] * 0.15) + (gf_diff * w["advanced"] * 0.20) + (w["home"] * 0.05)
    
    ai_prob_home = np.clip(implied_h + stat_delta, 0.05, 0.95)
    ai_prob_away = 1.0 - ai_prob_home
    
    edge_home = (ai_prob_home * oh) - 1.0
    edge_away = (ai_prob_away * oa) - 1.0
    
    if edge_home >= edge_away:
        best_pick, best_prob, best_odds, best_edge = home, ai_prob_home, oh, edge_home
    else:
        best_pick, best_prob, best_odds, best_edge = away, ai_prob_away, oa, edge_away
    
    b = best_odds - 1.0
    kelly_f = (b * best_prob - (1.0 - best_prob)) / b if b > 0 else 0
    stake = min(bankroll * 0.04, max(0, bankroll * kelly_f * kelly_fraction))
    
    if best_edge >= min_edge_slider and stake > 0:
        cursor.execute("SELECT id FROM bets WHERE id = %s", (match_id,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO bets VALUES (%s, %s, '🏒 NHL Hokej', %s, %s, %s, %s, %s, 'Oczekujący', 0.0, %s, %s, %s, 'H2H')",
                (match_id, datetime.now().strftime("%Y-%m-%d"), f"{home} vs {away}", float(best_odds), float(best_prob), float(best_edge), float(stake), home, away, best_pick)
            )
            db_conn.commit()

        analyzed_h2h.append({
            "home": home, "away": away, "commence_time": commence,
            "best_pick": best_pick, "ai_prob": best_prob, "odds": best_odds, 
            "edge": best_edge, "stake": round(stake, 2), "implied_h": round(implied_h * 100, 2)
        })

cursor.close()
db_conn.close()
analyzed_h2h.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY ---
st.markdown("## 🏒 Maksbet.app - Advanced NHL Quant Terminal")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Sygnały Value (NHL)", 
    "🛡️ Terminal Diagnostyczny", 
    "📈 Rejestr & Eksport CSV", 
    "📊 Krzywa Kapitału (Equity)", 
    "🎯 Zaawansowana Analityka"
])

with tab1:
    st.subheader("Aktualne Value Bets w NHL")
    if not analyzed_h2h:
        st.warning("Brak spotkań NHL spełniających kryterium Edge w tym oknie czasowym.")
    else:
        for m in analyzed_h2h:
            st.markdown(f"""
                <div class="value-card">
                    <h3>{m['home']} vs {m['away']}</h3>
                    <p style="color: #94a3b8; font-size: 12px;">Start: {m.get('commence_time', 'Najbliższe godziny')}</p>
                    <p>🎯 <b>Typ AI:</b> <span style="color: #60a5fa;">{m['best_pick']}</span> (Kurs: {m['odds']})</p>
                    <p>🔥 <b>Value:</b> <span style="color: #34d399;">+{round(m['edge']*100, 1)}%</span> | 💰 <b>Stawka Kellego:</b> {m['stake']} PLN</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(m['ai_prob'], text=f"Szanse AI: {round(m['ai_prob']*100, 1)}%")
            st.divider()

with tab2:
    st.subheader("🛡️ Status API oraz Bazy Danych")
    status_icon = "✅ OK" if status_code == 200 else "❌ Błąd"
    st.markdown(f"""
        <div class="test-box">
        <b>The Odds API (NHL):</b> Status HTTP {status_code} ({status_icon})<br>
        <b>Baza Neon.tech:</b> Połączenie aktywne ✅
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.subheader("📈 Rejestr Zakładów & Eksport danych")
    conn_s = get_db_connection()
    df_settled = pd.read_sql("SELECT * FROM bets", conn_s)
    conn_s.close()
    
    if df_settled.empty:
        st.info("Brak wpisów w bazie danych.")
    else:
        st.dataframe(df_settled, use_container_width=True)
        csv_data = df_settled.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pobierz pełny rejestr zakładów (CSV)",
            data=csv_data,
            file_name="maksbet_nhl_bets.csv",
            mime="text/csv",
        )

with tab4:
    st.subheader("📊 Krzywa Kapitału na żywo (Equity Curve)")
    conn_eq = get_db_connection()
    df_eq = pd.read_sql("SELECT date, profit, status FROM bets WHERE status != 'Oczekujący' ORDER BY date ASC", conn_eq)
    conn_eq.close()
    
    if df_eq.empty:
        st.info("Brak rozliczonych zakładów do wygenerowania krzywej kapitału.")
    else:
        df_eq['cumulative_profit'] = df_eq['profit'].cumsum() + bankroll
        df_eq['index'] = range(1, len(df_eq) + 1)
        
        chart = alt.Chart(df_eq).mark_line(point=True, strokeWidth=3, color='#10b981').encode(
            x=alt.X('index:Q', title='Liczba rozliczonych zakładów'),
            y=alt.Y('cumulative_profit:Q', title='Kapitał (PLN)', scale=alt.Scale(zero=False)),
            tooltip=['date', 'cumulative_profit', 'profit', 'status']
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

with tab5:
    st.subheader("🎯 Zaawansowane Statystyki Analityczne")
    conn_an = get_db_connection()
    df_an = pd.read_sql("SELECT * FROM bets WHERE status != 'Oczekujący'", conn_an)
    conn_an.close()
    
    if df_an.empty:
        st.info("Zaawansowana analityka uaktywni się po rozliczeniu pierwszych zakładów.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📌 Podział Gospodarz vs Gość")
            df_an['is_home'] = df_an['best_pick'] == df_an['home_team']
            split_summary = df_an.groupby('is_home').agg(
                Zakłady=('id', 'count'),
                Zysk=('profit', 'sum')
            ).reset()
            split_summary['Typ zakładu'] = split_summary['is_home'].map({True: 'Gospodarz (Home)', False: 'Gość (Away)'})
            st.dataframe(split_summary[['Typ zakładu', 'Zakłady', 'Zysk']], use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Skuteczność wg Przedziałów Kursów")
            bins = [1.0, 1.5, 2.0, 3.0, 10.0]
            labels = ['1.00 - 1.50', '1.51 - 2.00', '2.01 - 3.00', '3.00+']
            df_an['odds_bin'] = pd.cut(df_an['odds'], bins=bins, labels=labels, right=False)
            odds_summary = df_an.groupby('odds_bin', observed=False).agg(
                Zakłady=('id', 'count'),
                Zysk=('profit', 'sum')
            ).reset()
            st.dataframe(odds_summary, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### 📉 Reliability Diagram (Kalibracja Modelu AI)")
        st.caption("Porównanie pewności modelu z rzeczywistymi trafiami w przedziałach prawdopodobieństwa")
        
        prob_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]
        prob_labels = ['< 50%', '50-60%', '60-70%', '70-80%', '80%+']
        df_an['prob_bin'] = pd.cut(df_an['ai_prob'], bins=prob_bins, labels=prob_labels, right=False)
        calib = df_an.groupby('prob_bin', observed=False).agg(
            Wszystkie=('id', 'count'),
            Trafione=('status', lambda x: (x == 'Wygrany').sum())
        ).reset()
        calib['Rzeczywista Skuteczność'] = np.where(calib['Wszystkie'] > 0, (calib['Trafione'] / calib['Wszystkie'] * 100).round(1), 0)
        st.dataframe(calib, use_container_width=True)
    
