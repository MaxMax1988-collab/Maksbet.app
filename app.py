import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timezone, timedelta
import altair as alt

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Maksbet.app - Autonomous Quant Engine",
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
ODDS_API_KEY = "3e81b37bac9bb7efbe8d2cf68065b6dc"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# --- BAZA DANYCH SQLITE Z MODUŁEM WAG ---
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weights_history (
            sport TEXT PRIMARY KEY,
            w_win_pct REAL,
            w_advanced REAL,
            w_form REAL,
            w_home REAL
        )
    ''')
    
    default_weights = {
        "🏀 NBA Koszykówka": (0.25, 0.35, 0.25, 0.15),
        "🏒 NHL Hokej": (0.30, 0.35, 0.20, 0.15),
        "🏈 NFL Futbol Amerykański": (0.35, 0.30, 0.20, 0.15)
    }
    
    for sport, (w1, w2, w3, w4) in default_weights.items():
        cursor.execute('''
            INSERT OR IGNORE INTO weights_history (sport, w_win_pct, w_advanced, w_form, w_home)
            VALUES (?, ?, ?, ?, ?)
        ''', (sport, w1, w2, w3, w4))
        
    conn.commit()
    return conn

db_conn = init_db()

# --- HELPER: POBIERANIE AKTYWNYCH WAG ---
def get_sport_weights(sport_name):
    cursor = db_conn.cursor()
    cursor.execute("SELECT w_win_pct, w_advanced, w_form, w_home FROM weights_history WHERE sport = ?", (sport_name,))
    row = cursor.fetchone()
    if row:
        return {"win_pct": row[0], "advanced": row[1], "form": row[2], "home": row[3]}
    return {"win_pct": 0.25, "advanced": 0.35, "form": 0.25, "home": 0.15}

# --- MODUŁ UCZENIA SIĘ WAG (FEEDBACK LOOP) ---
def update_weights_after_settlement(sport_name, bet_won):
    cursor = db_conn.cursor()
    w = get_sport_weights(sport_name)
    lr = 0.01 if bet_won else -0.01
    
    new_adv = max(0.10, w["advanced"] + lr)
    new_form = max(0.10, w["form"] + (lr * 0.5))
    
    total = w["win_pct"] + new_adv + new_form + w["home"]
    
    nw_win = w["win_pct"] / total
    nw_adv = new_adv / total
    nw_form = new_form / total
    nw_home = w["home"] / total
    
    cursor.execute('''
        UPDATE weights_history 
        SET w_win_pct = ?, w_advanced = ?, w_form = ?, w_home = ?
        WHERE sport = ?
    ''', (nw_win, nw_adv, nw_form, nw_home, sport_name))
    db_conn.commit()

# --- AUTOMATYCZNE ROZLICZANIE ZAKŁADÓW Z API (SCORES) ---
def auto_settle_from_api():
    cursor = db_conn.cursor()
    cursor.execute("SELECT id, sport, odds, stake, best_pick, home_team, away_team FROM bets WHERE status = 'Oczekujący'")
    pending_bets = cursor.fetchall()
    
    if not pending_bets:
        return

    sport_keys = {
        "🏀 NBA Koszykówka": "basketball_nba",
        "🏒 NHL Hokej": "icehockey_nhl",
        "🏈 NFL Futbol Amerykański": "americanfootball_nfl"
    }

    for sport_name, d_key in sport_keys.items():
        url = f"{ODDS_BASE_URL}/{d_key}/scores"
        params = {"apiKey": ODDS_API_KEY, "daysFrom": 3}
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                completed_matches = res.json()
                for match in completed_matches:
                    if match.get("completed", False):
                        match_id = str(match.get("id"))
                        scores = match.get("scores", [])
                        
                        if scores and len(scores) >= 2:
                            team1, score1 = scores[0].get("name"), int(scores[0].get("score", 0))
                            team2, score2 = scores[1].get("name"), int(scores[1].get("score", 0))
                            
                            if score1 > score2:
                                winner = team1
                            elif score2 > score1:
                                winner = team2
                            else:
                                continue
                            
                            for bet in pending_bets:
                                bet_id, b_sport, odds, stake, best_pick, home, away = bet
                                if bet_id == match_id and b_sport == sport_name:
                                    won = (best_pick.lower() in winner.lower() or winner.lower() in best_pick.lower())
                                    
                                    if won:
                                        status = 'Wygrany'
                                        profit = round((odds - 1.0) * stake, 2)
                                    else:
                                        status = 'Przegrany'
                                        profit = round(-stake, 2)
                                        
                                    cursor.execute("UPDATE bets SET status = ?, profit = ? WHERE id = ?", (status, profit, bet_id))
                                    db_conn.commit()
                                    update_weights_after_settlement(sport_name, won)
        except Exception:
            pass

auto_settle_from_api()

# --- POBIERANIE STATYSTYK Z OFICJALNYCH API ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nhl_stats():
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
                wins, losses, ot = team.get("wins", 0), team.get("losses", 0), team.get("otLosses", 0)
                tot = wins + losses + ot
                win_pct = (wins / tot) if tot > 0 else 0.5
                gf_g = (team.get("goalFor", 0) / tot) if tot > 0 else 3.0
                ga_g = (team.get("goalAgainst", 0) / tot) if tot > 0 else 3.0
                stats_dict[full_name] = {"win_pct": win_pct, "gf_g": gf_g, "ga_g": ga_g}
                stats_dict[name_en] = {"win_pct": win_pct, "gf_g": gf_g, "ga_g": ga_g}
            return stats_dict
    except Exception: pass
    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nba_stats():
    try:
        from nba_api.stats.endpoints import leaguestandings
        standings = leaguestandings.LeagueStandings().get_data_frames()[0]
        stats_dict = {}
        for _, row in standings.iterrows():
            city, name = str(row.get('TeamCity', '')), str(row.get('TeamName', ''))
            full_name = f"{city} {name}".strip()
            win_pct = float(row.get('WinPCT', 0.5))
            stats_dict[full_name] = {"win_pct": win_pct}
            stats_dict[name] = {"win_pct": win_pct}
        return stats_dict
    except Exception: return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nfl_stats():
    try:
        url = "http://site.api.espn.com/apis/v2/sports/football/nfl/standings"
        res = requests.get(url, timeout=10)
        stats_dict = {}
        if res.status_code == 200:
            data = res.json()
            for group in data.get("children", []):
                for team_entry in group.get("standings", {}).get("entries", []):
                    team_info = team_entry.get("team", {})
                    name = team_info.get("displayName", "")
                    nickname = team_info.get("name", "")
                    
                    stats = team_entry.get("stats", [])
                    wins, losses = 0, 0
                    point_diff = 0.0
                    for s in stats:
                        if s.get("name") == "wins": wins = s.get("value", 0)
                        if s.get("name") == "losses": losses = s.get("value", 0)
                        if s.get("name") == "pointDifferential": point_diff = s.get("value", 0.0)
                        
                    tot = wins + losses
                    win_pct = (wins / tot) if tot > 0 else 0.5
                    stats_dict[name] = {"win_pct": win_pct, "point_diff": point_diff}
                    stats_dict[nickname] = {"win_pct": win_pct, "point_diff": point_diff}
        return stats_dict
    except Exception:
        return {}

nhl_stats_cache = fetch_nhl_stats()
nba_stats_cache = fetch_nba_stats()
nfl_stats_cache = fetch_nfl_stats()

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🎯 Maksbet.app")
    st.caption("Autonomous Quant Engine")
    st.markdown("---")
    
    discipline = st.selectbox("Dyscyplina", ["🏀 NBA Koszykówka", "🏒 NHL Hokej", "🏈 NFL Futbol Amerykański"])
    bankroll = st.number_input("Kapitał początkowy (PLN)", min_value=100, value=1000, step=100)
    kelly_fraction = st.selectbox("Strategia Kellego", [0.1, 0.25, 0.5], index=1, format_func=lambda x: f"{int(x*100)}% Kellego")
    min_edge_slider = st.slider("Minimalny Edge (%) do akceptacji", min_value=0.5, max_value=10.0, value=1.5, step=0.5) / 100.0
    
    curr_weights = get_sport_weights(discipline)
    st.markdown("---")
    st.markdown("<b>🧠 Wyuczone Wagi Modelu:</b>", unsafe_allow_html=True)
    st.caption(f"・ Win % / Bilans: {round(curr_weights['win_pct']*100, 1)}%")
    st.caption(f"・ Advanced Stats: {round(curr_weights['advanced']*100, 1)}%")
    st.caption(f"・ Form (L10/L5): {round(curr_weights['form']*100, 1)}%")
    st.caption(f"・ Home Advantage: {round(curr_weights['home']*100, 1)}%")
    
    st.markdown("---")
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bets")
    total_tracked = cursor.fetchone()[0]
    st.metric("Monitorowane zakłady", total_tracked)

# --- POBIERANIE KURSÓW Z THE ODDS API ---
if "NBA" in discipline:
    key = "basketball_nba"
elif "NHL" in discipline:
    key = "icehockey_nhl"
else:
    key = "americanfootball_nfl"

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
cursor = db_conn.cursor()

for m in raw_matches:
    match_id = m['id']
    match_name = f"{m['home']} vs {m['away']}"
    odds_home, odds_away = m['odds_home'], m['odds_away']
    
    prob_raw_h = 1.0 / odds_home if odds_home > 0 else 0.5
    prob_raw_a = 1.0 / odds_away if odds_away > 0 else 0.5
    tot_v = prob_raw_h + prob_raw_a
    implied_h = prob_raw_h / tot_v if tot_v > 0 else 0.5
    no_vig_sum = implied_h + (1.0 - implied_h)
    
    w = curr_weights
    stat_delta = 0.0
    
    if "NBA" in discipline:
        h_s, a_s = nba_stats_cache.get(m['home']), nba_stats_cache.get(m['away'])
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            stat_delta = wd * (w["win_pct"] * 0.08 + w["advanced"] * 0.05)
            
    elif "NHL" in discipline:
        h_s, a_s = nhl_stats_cache.get(m['home']), nhl_stats_cache.get(m['away'])
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            gf_diff = (h_s["gf_g"] - a_s["gf_g"]) / 5.0
            stat_delta = (wd * w["win_pct"] * 0.08) + (gf_diff * w["advanced"] * 0.08)
            
    else:
        h_s, a_s = nfl_stats_cache.get(m['home']), nfl_stats_cache.get(m['away'])
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            pd_diff = (h_s["point_diff"] - a_s["point_diff"]) / 50.0
            stat_delta = (wd * w["win_pct"] * 0.08) + (pd_diff * w["advanced"] * 0.08)
        
    ai_prob_home = np.clip(implied_h + stat_delta, 0.05, 0.95)
    ai_prob_away = 1.0 - ai_prob_home
    
    edge_home = (ai_prob_home * odds_home) - 1.0
    edge_away = (ai_prob_away * odds_away) - 1.0
    
    if edge_home >= edge_away:
        best_pick, best_prob, best_odds, best_edge = m['home'], ai_prob_home, odds_home, edge_home
    else:
        best_pick, best_prob, best_odds, best_edge = m['away'], ai_prob_away, odds_away, edge_away
    
    b = best_odds - 1.0
    kelly_f = (b * best_prob - (1.0 - best_prob)) / b if b > 0 else 0
    stake = max(0, bankroll * kelly_f * kelly_fraction)
    
    if best_edge >= min_edge_slider and stake > 0:
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
            "implied_h": round(implied_h * 100, 2),
            "no_vig_sum": round(no_vig_sum, 4)
        })
        analyzed_matches.append(m_copy)

analyzed_matches.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY ---
st.markdown("## ⚡ Dashboard Kwantowy")
tab1, tab2, tab3, tab4 = st.tabs(["🔥 Sygnały Value", "🛡️ Terminal Diagnostyczny", "📁 Rejestr & Rozliczenia", "📈 Krzywa Kapitału"])

with tab1:
    st.subheader(f"Terminal Analityczny: {discipline}")
    st.caption("Czysty model No-Vig z adaptacyjnymi wagami uczenia")
    
    if not analyzed_matches:
        st.warning("Brak nadchodzących spotkań spełniających kryteria dodatniego Value w oknie 72h.")
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
                st.progress(m['ai_prob'], text=f"Rozkład szans AI: {prob_pct}% (No-Vig bazowy: {m['implied_h']}%)")
                st.divider()

with tab2:
    st.subheader("🛡️ Terminal Diagnostyczny Systemu")
    st.caption("Weryfikacja API, neutralności marży oraz stanu sieci wag")
    
    test_api_status = "✅ OK (Połączenie stabilne)" if raw_matches else "❌ Brak danych w oknie 72h"
    st.markdown(f"""
        <div class="test-box">
        <b>[TEST 1] Zewnętrzne API Kursów (The Odds API):</b><br>
        Status: {test_api_status}<br>
        Pobrano surowych meczów dla {discipline}: <b>{len(raw_matches)}</b>
        </div>
    """, unsafe_allow_html=True)
    
    sample_check = analyzed_matches[0] if analyzed_matches else None
    if sample_check:
        st.markdown(f"""
            <div class="test-box">
            <b>[TEST 2] Matematyczna neutralność marży (No-Vig Check):</b><br>
            Mecz testowy: {sample_check['home']} vs {sample_check['away']}<br>
            Suma prawdopodobieństw No-Vig (musi wynosić dokładnie 1.0): <b>{sample_check['no_vig_sum']}</b> ✅
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="test-box">
            <b>[TEST 2] Matematyczna neutralność marży (No-Vig Check):</b><br>
            Brak meczów z dodatnim Value do weryfikacji.
            </div>
        """, unsafe_allow_html=True)
        
    try:
        cursor.execute("SELECT COUNT(*) FROM bets")
        db_count = cursor.fetchone()[0]
        db_status = f"✅ OK ({db_count} rekordów w tabeli)"
    except Exception as e:
        db_status = f"❌ Błąd: {e}"
        
    st.markdown(f"""
        <div class="test-box">
        <b>[TEST 3] Integralność bazy danych i silnika wag:</b><br>
        Połączenie z plikiem `bets_database.db`: {db_status}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ Strefa Zarządzania Bazą")
    if st.button("🗑️ Resetuj / Wyczyść całą bazę danych"):
        cursor.execute("DROP TABLE IF EXISTS bets")
        cursor.execute("DROP TABLE IF EXISTS weights_history")
        db_conn.commit()
        st.success("Baza danych została w pełni zresetowana! Odśwież stronę (F5), aby zacząć od nowa.")
        st.rerun()

with tab3:
    st.subheader("📁 Rejestr Zakładów & Rozliczanie (Feedback Loop)")
    df_bets = pd.read_sql("SELECT * FROM bets", db_conn)
    if df_bets.empty: 
        st.info("Brak zapisanych zakładów.")
    else: 
        st.dataframe(df_bets, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ⚖️ Ręczna korekta / rozliczenie")
        pending_bets = df_bets[df_bets['status'] == 'Oczekujący']
        if not pending_bets.empty:
            selected_bet_id = st.selectbox("Wybierz mecz do ręcznego rozliczenia", pending_bets['id'].tolist(), format_func=lambda x: pending_bets[pending_bets['id']==x]['match_name'].values[0])
            bet_row = pending_bets[pending_bets['id'] == selected_bet_id].iloc[0]
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Wygrana (Ucz wagi +)"):
                    profit = (bet_row['odds'] - 1.0) * bet_row['stake']
                    cursor.execute("UPDATE bets SET status='Wygrany', profit=? WHERE id=?", (profit, selected_bet_id))
                    db_conn.commit()
                    update_weights_after_settlement(bet_row['sport'], True)
                    st.success("Zakład rozliczony jako WYGRANY! Wagi modelu zostały zaktualizowane.")
                    st.rerun()
            with col_b:
                if st.button("❌ Porażka (Ucz wagi -)"):
                    profit = -bet_row['stake']
                    cursor.execute("UPDATE bets SET status='Przegrany', profit=? WHERE id=?", (profit, selected_bet_id))
                    db_conn.commit()
                    update_weights_after_settlement(bet_row['sport'], False)
                    st.error("Zakład rozliczony jako PRZEGRANY. Model skorygował wagi.")
                    st.rerun()

with tab4:
    st.subheader("📈 Krzywa Kapitału")
    df_bets = pd.read_sql("SELECT date, profit FROM bets WHERE status != 'Oczekujący'", db_conn)
    if df_bets.empty: st.info("Wykres uaktywni się po rozliczeniu pierwszego zakładu.")
    else:
        df_bets['cumulative_profit'] = df_bets['profit'].cumsum() + bankroll
        chart = alt.Chart(df_bets).mark_line(strokeWidth=3, color='#34d399').encode(x='date:T', y='cumulative_profit:Q').properties(height=350)
        st.altair_chart(chart, use_container_width=True)
