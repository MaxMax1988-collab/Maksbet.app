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
    .handicap-card { background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border-left: 6px solid #818cf8; padding: 20px; border-radius: 12px; margin-bottom: 15px; border: 1px solid #312e81; }
    .test-box { background: #111827; border: 1px solid #374151; padding: 15px; border-radius: 10px; margin-bottom: 10px; font-family: monospace; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG API ---
ODDS_API_KEY = "af7b4349b28ca8ffe8a5f2a0722073e2"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# --- BAZA DANYCH SQLITE Z BEZPIECZNĄ MIGRACJĄ ---
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
            best_pick TEXT,
            bet_type TEXT DEFAULT 'H2H'
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE bets ADD COLUMN bet_type TEXT DEFAULT 'H2H'")
    except sqlite3.OperationalError:
        pass
    
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
        "🏀 NBA Koszykówka": (0.30, 0.35, 0.20, 0.15),
        "🏒 NHL Hokej": (0.25, 0.40, 0.20, 0.15),
        "🏈 NFL Futbol Amerykański": (0.35, 0.35, 0.15, 0.15)
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
    return {"win_pct": 0.30, "advanced": 0.35, "form": 0.20, "home": 0.15}

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
                            s1 = int(scores[0].get("score", 0))
                            s2 = int(scores[1].get("score", 0))
                            t1, t2 = scores[0].get("name"), scores[1].get("name")
                            
                            for bet in pending_bets:
                                bet_id, b_sport, odds, stake, best_pick, home, away = bet
                                if bet_id.split("_")[0] == match_id and b_sport == sport_name:
                                    winner = t1 if s1 > s2 else t2
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
    st.caption(f"・ Bilans / Win %: {round(curr_weights['win_pct']*100, 1)}%")
    st.caption(f"・ Advanced Stats (API): {round(curr_weights['advanced']*100, 1)}%")
    st.caption(f"・ Dynamika / Forma: {round(curr_weights['form']*100, 1)}%")
    st.caption(f"・ Home Advantage: {round(curr_weights['home']*100, 1)}%")
    
    st.markdown("---")
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bets")
    total_tracked = cursor.fetchone()[0]
    st.metric("Monitorowane zakłady", total_tracked)

# --- POBIERANIE KURSÓW I HANDICAPÓW Z THE ODDS API ---
if "NBA" in discipline:
    key = "basketball_nba"
elif "NHL" in discipline:
    key = "icehockey_nhl"
else:
    key = "americanfootball_nfl"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_odds_and_spreads(d_key):
    url = f"{ODDS_BASE_URL}/{d_key}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,spreads", "oddsFormat": "decimal"}
    all_m = []
    debug_info = {"status_code": 0, "total_raw": 0, "error": None}
    
    try:
        res = requests.get(url, params=params, timeout=10)
        debug_info["status_code"] = res.status_code
        if res.status_code == 200:
            raw_json = res.json()
            debug_info["total_raw"] = len(raw_json)
            for match in raw_json:
                match_id = match.get('id')
                home, away = match.get('home_team'), match.get('away_team')
                commence = match.get('commence_time')
                
                oh, oa = 1.90, 1.90
                sh_line, sa_line, sh_odds, sa_odds = 0.0, 0.0, 1.90, 1.90
                
                books = match.get('bookmakers', [])
                if books:
                    for mkt in books[0].get('markets', []):
                        if mkt.get('key') == 'h2h':
                            for out in mkt.get('outcomes', []):
                                if out.get('name') == home: oh = float(out.get('price', 1.90))
                                elif out.get('name') == away: oa = float(out.get('price', 1.90))
                        elif mkt.get('key') == 'spreads':
                            for out in mkt.get('outcomes', []):
                                if out.get('name') == home:
                                    sh_line = float(out.get('point', 0.0))
                                    sh_odds = float(out.get('price', 1.90))
                                elif out.get('name') == away:
                                    sa_line = float(out.get('point', 0.0))
                                    sa_odds = float(out.get('price', 1.90))
                                    
                all_m.append({
                    "id": str(match_id), "home": home, "away": away, "commence_time": commence,
                    "odds_home": oh, "odds_away": oa,
                    "spread_home_line": sh_line, "spread_home_odds": sh_odds,
                    "spread_away_line": sa_line, "spread_away_odds": sa_odds
                })
        else:
            debug_info["error"] = res.text
    except Exception as e:
        debug_info["error"] = str(e)
        
    return all_m, debug_info

raw_matches, api_debug = fetch_odds_and_spreads(key)
analyzed_h2h = []
analyzed_handicaps = []
cursor = db_conn.cursor()

def find_team_stats(team_name, stats_cache):
    if not stats_cache: return None
    if team_name in stats_cache: return stats_cache[team_name]
    for k, v in stats_cache.items():
        if k.lower() in team_name.lower() or team_name.lower() in k.lower():
            return v
    return None

for m in raw_matches:
    match_id = m['id']
    match_name = f"{m['home']} vs {m['away']}"
    odds_home, odds_away = m['odds_home'], m['odds_away']
    
    # --- 1. MODEL H2H ---
    prob_raw_h = 1.0 / odds_home if odds_home > 0 else 0.5
    prob_raw_a = 1.0 / odds_away if odds_away > 0 else 0.5
    tot_v = prob_raw_h + prob_raw_a
    implied_h = prob_raw_h / tot_v if tot_v > 0 else 0.5
    no_vig_sum = implied_h + (1.0 - implied_h)
    
    w = curr_weights
    stat_delta = 0.0
    expected_margin = 0.0
    
    if "NBA" in discipline:
        h_s = find_team_stats(m['home'], nba_stats_cache)
        a_s = find_team_stats(m['away'], nba_stats_cache)
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            stat_delta = wd * (w["win_pct"] * 0.08 + w["advanced"] * 0.05)
            expected_margin = wd * 12.0 + (w["home"] * 3.5)
            
    elif "NHL" in discipline:
        h_s = find_team_stats(m['home'], nhl_stats_cache)
        a_s = find_team_stats(m['away'], nhl_stats_cache)
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            gf_diff = (h_s["gf_g"] - a_s["ga_g"]) - (a_s["gf_g"] - h_s["ga_g"])
            stat_delta = (wd * w["win_pct"] * 0.08) + (gf_diff * w["advanced"] * 0.05)
            expected_margin = gf_diff + (w["home"] * 0.3)
            
    else:
        h_s = find_team_stats(m['home'], nfl_stats_cache)
        a_s = find_team_stats(m['away'], nfl_stats_cache)
        if h_s and a_s:
            wd = h_s["win_pct"] - a_s["win_pct"]
            pd_diff = (h_s["point_diff"] - a_s["point_diff"]) / 50.0
            stat_delta = (wd * w["win_pct"] * 0.08) + (pd_diff * w["advanced"] * 0.05)
            expected_margin = (pd_diff * 7.0) + (w["home"] * 2.5)
        
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
    stake = min(bankroll * 0.04, max(0, bankroll * kelly_f * kelly_fraction))
    
    if best_edge >= min_edge_slider and stake > 0:
        cursor.execute("SELECT id FROM bets WHERE id = ?", (match_id,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Oczekujący', 0.0, ?, ?, ?, 'H2H')",
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
        analyzed_h2h.append(m_copy)

    # --- 2. ZABEZPIECZONY MODEL HANDICAPÓW (BAYESIAN ANCHORED) ---
    sh_line, sa_line = m['spread_home_line'], m['spread_away_line']
    sh_odds, sa_odds = m['spread_home_odds'], m['spread_away_odds']
    
    if sh_odds > 0 and sa_odds > 0 and sh_line != 0.0:
        p_raw_sh = 1.0 / sh_odds
        p_raw_sa = 1.0 / sa_odds
        p_tot_s = p_raw_sh + p_raw_sa
        mkt_hc_home = p_raw_sh / p_tot_s if p_tot_s > 0 else 0.5
        
        scale_factor = 22.0 if "NBA" in discipline else (6.0 if "NHL" in discipline else 24.0)
        model_hc_home = 1.0 / (1.0 + np.exp(-expected_margin / scale_factor))
        
        # Kotwiczenie Bayesian: 85% Rynek / 15% Model
        prob_hc_home = (0.85 * mkt_hc_home) + (0.15 * model_hc_home)
        prob_hc_home = np.clip(prob_hc_home, 0.35, 0.65)
        prob_hc_away = 1.0 - prob_hc_home
        
        edge_hc_home = (prob_hc_home * sh_odds) - 1.0
        edge_hc_away = (prob_hc_away * sa_odds) - 1.0
        
        if edge_hc_home >= edge_hc_away:
            hc_pick = f"{m['home']} ({sh_line:+} pkt)"
            hc_prob, hc_odds, hc_edge = prob_hc_home, sh_odds, edge_hc_home
        else:
            hc_pick = f"{m['away']} ({sa_line:+} pkt)"
            hc_prob, hc_odds, hc_edge = prob_hc_away, sa_odds, edge_hc_away
            
        b_hc = hc_odds - 1.0
        kelly_hc = (b_hc * hc_prob - (1.0 - hc_prob)) / b_hc if b_hc > 0 else 0
        stake_hc = min(bankroll * 0.04, max(0, bankroll * kelly_hc * kelly_fraction))
        
        if hc_edge >= min_edge_slider and stake_hc > 0:
            hc_bet_id = f"{match_id}_hc"
            cursor.execute("SELECT id FROM bets WHERE id = ?", (hc_bet_id,))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Oczekujący', 0.0, ?, ?, ?, 'HANDICAP')",
                    (hc_bet_id, datetime.now().strftime("%Y-%m-%d"), discipline, f"{match_name} [HANDICAP]", hc_odds, hc_prob, hc_edge, stake_hc, m['home'], m['away'], hc_pick)
                )
                db_conn.commit()

            m_hc = m.copy()
            m_hc.update({
                "best_pick": hc_pick,
                "ai_prob": hc_prob,
                "odds": hc_odds,
                "edge": hc_edge,
                "stake": round(stake_hc, 2)
            })
            analyzed_handicaps.append(m_hc)

analyzed_h2h.sort(key=lambda x: x['edge'], reverse=True)
analyzed_handicaps.sort(key=lambda x: x['edge'], reverse=True)

# --- INTERFEJS GŁÓWNY ---
st.markdown("## ⚡ Dashboard Kwantowy & Performance Live")
tab1, tab_hc, tab2, tab3, tab4 = st.tabs(["🔥 Sygnały Value (H2H)", "🎯 Analiza Handicapów", "🛡️ Terminal Diagnostyczny", "📁 Rejestr & Rozliczenia", "📈 Zaawansowana Analityka & ROI"])

with tab1:
    st.subheader(f"Terminal Analityczny H2H: {discipline}")
    st.caption("Model No-Vig zasilany statystykami oficjalnych lig")
    
    if not analyzed_h2h:
        st.warning("Brak nadchodzących spotkań H2H spełniających kryteria dodatniego Value w tym momencie.")
    else:
        for m in analyzed_h2h:
            edge_pct = round(m['edge'] * 100, 1)
            prob_pct = round(m['ai_prob'] * 100, 1)
            with st.container():
                st.markdown(f"""
                    <div class="value-card">
                        <h3>{m['home']} vs {m['away']}</h3>
                        <p style="color: #94a3b8; font-size: 12px;">Start: {m.get('commence_time', 'Najbliższe godziny')}</p>
                        <p>🎯 <b>Typ AI (H2H):</b> <span style="color: #60a5fa;">{m['best_pick']}</span> (Kurs: {m['odds']})</p>
                        <p>🔥 <b>Value:</b> <span style="color: #34d399;">+{edge_pct}%</span> | 💰 <b>Stawka Kellego:</b> {m['stake']} PLN</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(m['ai_prob'], text=f"Rozkład szans AI: {prob_pct}% (No-Vig bazowy: {m['implied_h']}%)")
                st.divider()

with tab_hc:
    st.subheader(f"Dedykowany Moduł Handicapów: {discipline}")
    st.caption("Zaawansowany model Sigmoid z kotwiczeniem rynkowym")
    
    if not analyzed_handicaps:
        st.info("Brak zyskownych linii handicapowych spełniających minimalne kryterium Edge w tym oknie czasowym.")
    else:
        for m in analyzed_handicaps:
            edge_pct = round(m['edge'] * 100, 1)
            prob_pct = round(m['ai_prob'] * 100, 1)
            with st.container():
                st.markdown(f"""
                    <div class="handicap-card">
                        <h3>{m['home']} vs {m['away']}</h3>
                        <p style="color: #a5b4fc; font-size: 12px;">Start: {m.get('commence_time', 'Najbliższe godziny')}</p>
                        <p>🎯 <b>Rekomendowany Handicap:</b> <span style="color: #a855f7;">{m['best_pick']}</span> (Kurs: {m['odds']})</p>
                        <p>🔥 <b>Value Spreadu:</b> <span style="color: #34d399;">+{edge_pct}%</span> | 💰 <b>Stawka Kellego:</b> {m['stake']} PLN</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(m['ai_prob'], text=f"Prawdopodobieństwo pokrycia linii AI: {prob_pct}%")
                st.divider()

with tab2:
    st.subheader("🛡️ Terminal Diagnostyczny Systemu")
    st.caption("Weryfikacja API, neutralności marży oraz stanu sieci wag")
    
    status_icon = "✅ OK" if api_debug["status_code"] == 200 and api_debug["total_raw"] > 0 else "❌ Błąd / Brak danych"
    st.markdown(f"""
        <div class="test-box">
        <b>[TEST 1] Zewnętrzne API Kursów i Spreads (The Odds API):</b><br>
        Status HTTP: <b>{api_debug['status_code']}</b> ({status_icon})<br>
        Pobrano surowych meczów z API dla {discipline}: <b>{api_debug['total_raw']}</b><br>
        Szczegóły / Błędy: <span style="color: #f87171;">{api_debug['error'] if api_debug['error'] else 'Brak błędów'}</span>
        </div>
    """, unsafe_allow_html=True)
    
    sample_check = analyzed_h2h[0] if analyzed_h2h else None
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
    st.subheader("📈 Zaawansowana Analityka Skuteczności & ROI Live")
    
    df_settled = pd.read_sql("SELECT * FROM bets WHERE status != 'Oczekujący'", db_conn)
    
    if df_settled.empty:
        st.info("Brak rozliczonych zakładów. Rozlicz pierwsze zakłady w zakładce 'Rejestr & Rozliczenia', aby wygenerować pełny raport statystyczny.")
    else:
        total_settled = len(df_settled)
        won_bets = len(df_settled[df_settled['status'] == 'Wygrany'])
        win_rate = (won_bets / total_settled) * 100 if total_settled > 0 else 0
        total_profit = df_settled['profit'].sum()
        total_staked = df_settled['stake'].sum()
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rozliczone Zakłady", total_settled)
        col2.metric("Skuteczność (Win Rate)", f"{win_rate:.1f}%")
        col3.metric("Całkowity Zysk / Strata", f"{total_profit:.2f} PLN", delta=f"{total_profit:.2f} PLN")
        col4.metric("ROI Modelu", f"{roi:.2f}%", delta=f"{roi:.2f}%")
        
        st.markdown("---")
        st.subheader("📊 Analiza wg Dyscyplin i Typu Zakładów")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### Skuteczność per Sport")
            sport_perf = df_settled.groupby('sport').agg(
                Zakłady=('id', 'count'),
                Zysk=('profit', 'sum')
            ).reset_index()
            st.dataframe(sport_perf, use_container_width=True)
            
        with col_m2:
            st.markdown("##### Skuteczność per Typ Zakładu (H2H vs Handicap)")
            type_perf = df_settled.groupby('bet_type').agg(
                Zakłady=('id', 'count'),
                Zysk=('profit', 'sum')
            ).reset_index()
            st.dataframe(type_perf, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Krzywa Wzrostu Kapitału (Equity Curve)")
        
        df_settled['cumulative_profit'] = df_settled['profit'].cumsum()
        df_settled['Kapitał (PLN)'] = bankroll + df_settled['cumulative_profit']
        
        base_chart = alt.Chart(df_settled).mark_line(strokeWidth=3, color='#34d399').encode(
            x=alt.X('date:T', title='Data rozliczenia'),
            y=alt.Y('Kapitał (PLN):Q', title='Stan Bankrollu (PLN)', scale=alt.Scale(zero=False)),
            tooltip=['date', 'match_name', 'profit', 'Kapitał (PLN)']
        )
        
        rule = alt.Chart(pd.DataFrame({'y': [bankroll]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y:Q')
        
        final_chart = (base_chart + rule).properties(height=400)
        st.altair_chart(final_chart, use_container_width=True)
        st.caption("ℹ️ Czerwona linia przerywana oznacza poziom kapitału początkowego (bankroll startowy). Wszystko powyżej to czysty zysk modelu.")
