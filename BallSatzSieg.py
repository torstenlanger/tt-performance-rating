import math
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar, brentq
from bs4 import BeautifulSoup


# ---------------------------------------------------------
# 1. Analytische Rally → Satz Wahrscheinlichkeit
# ---------------------------------------------------------

def set_win_probability(p: float) -> float:
    """Analytische Wahrscheinlichkeit, einen TT-Satz bis 11 mit 2 Punkten Vorsprung zu gewinnen."""
    q = 1 - p

    normal = sum(
        math.comb(10 + k, 10) * (p ** 11) * (q ** k)
        for k in range(10)
    )

    p_10_10 = math.comb(20, 10) * (p ** 10) * (q ** 10)
    p_win_after_10_10 = p**2 / (1 - 2*p*q)
    deuce = p_10_10 * p_win_after_10_10

    return normal + deuce


# ---------------------------------------------------------
# 2. Analytische Satz → Match Wahrscheinlichkeit (Best of N)
# ---------------------------------------------------------

def match_win_probability(p_set: float, best_of: int = 5) -> float:
    """Analytische Wahrscheinlichkeit, ein Best-of-N Match zu gewinnen."""
    needed = best_of // 2 + 1
    q = 1 - p_set

    total = 0.0
    for k in range(needed):
        ways = math.comb(needed - 1 + k, k)
        total += ways * (p_set ** needed) * (q ** k)

    return total


# ---------------------------------------------------------
# 3. TTR-Siegwahrscheinlichkeit
# ---------------------------------------------------------

def win_probability(ttr_player: float, ttr_opponent: float) -> float:
    return 1.0 / (1.0 + 10 ** ((ttr_opponent - ttr_player) / 150.0))


# ---------------------------------------------------------
# 4. Kaskade: p_rally → p_set → p_match
# ---------------------------------------------------------

def cascade(p_rally: float, best_of: int = 5):
    p_set   = set_win_probability(p_rally)
    p_match = match_win_probability(p_set, best_of=best_of)
    return p_set, p_match


# ---------------------------------------------------------
# 5. Inversion: TTR → p_rally (via numerische Inversion der Kaskade)
# ---------------------------------------------------------

def p_rally_from_ttr(delta_ttr: float, best_of: int = 5) -> float:
    """Berechnet p_rally so, dass die Kaskade exakt die TTR-definierte p_match liefert."""
    p_match_target = 1.0 / (1.0 + 10 ** (-delta_ttr / 150.0))
    return brentq(
        lambda p: match_win_probability(set_win_probability(p), best_of) - p_match_target,
        0.001, 0.999
    )


# ---------------------------------------------------------
# 6. Analytische Inversion der TTR-Gleichung (Match → TTR)
# ---------------------------------------------------------

def ttr_from_rally_prob(p_rally: float, ttr_opp: float, best_of: int = 5,
                         low: float = 0, high: float = 3000) -> float:
    """
    Invertiert die Kaskade: p_rally → p_match → TTR.
    Arbeitet auf p_rally-Skala, vermeidet Saturierung bei p_match → 1.
    """
    p_set, p_match = cascade(p_rally, best_of)

    if p_match <= 0:
        return low
    if p_match >= 1:
        return high

    T = ttr_opp - 150 * math.log10((1 / p_match) - 1)
    return max(low, min(high, T))


def tagesform_ttr_multi(p_match_list, ttr_opponents, best_of: int = 5,
                         low: float = 0, high: float = 3000) -> float:
    """
    Performance Rating per Method-of-Moments:
    Sucht TTR* so, dass die Summe der TTR-basierten Siegwahrscheinlichkeiten
    mit der Summe der beobachteten Match-Siegwahrscheinlichkeiten übereinstimmt.

    Löst: Σ win_probability(TTR*, TTR_opp_i) = Σ p_match_i
    """
    if not p_match_list:
        return (low + high) / 2

    target_sum = sum(p_match_list)

    def residual(ttr):
        predicted_sum = sum(win_probability(ttr, ttr_opp) for ttr_opp in ttr_opponents)
        return predicted_sum - target_sum

    # Randfall: target_sum außerhalb des erreichbaren Bereichs
    if residual(low) > 0:
        return low
    if residual(high) < 0:
        return high

    return brentq(residual, low, high)


# ---------------------------------------------------------
# 7. MLE für p_rally aus Satzergebnissen
# ---------------------------------------------------------

def set_likelihood(a: int, b: int, p: float) -> float:
    """Wahrscheinlichkeit des konkreten Satzergebnisses a:b gegeben Rally-p."""
    q = 1 - p
    winner_p = p if a > b else q
    loser_p  = q if a > b else p
    w, l = max(a, b), min(a, b)

    if l < 10:
        return math.comb(w + l - 1, l) * (winner_p ** w) * (loser_p ** l)
    else:
        p_1010  = math.comb(20, 10) * (p ** 10) * (q ** 10)
        extra   = w - 10
        p_extra = (winner_p * loser_p) ** (extra - 1) * winner_p ** 2
        return p_1010 * p_extra


# χ²-Test Schwellwerte (Likelihood-Ratio-Test, p-Wert)
# p >= 0.10 → konsistent | 0.05–0.10 → auffällig | 0.01–0.05 → inkonsistent | < 0.01 → stark inkonsistent


def mle_rally_prob(sets: list) -> tuple:
    """
    Maximum-Likelihood-Schätzer für p_rally aus einer Liste von Satzergebnissen.
    Gibt (p_mle, player_points, opp_points, fit_score, lrt_D, lrt_p, lrt_df) zurück.

    fit_score : mittlerer log-likelihood pro Rally am MLE-Optimum (heuristisch)
    lrt_D     : Likelihood-Ratio-Teststatistik D = 2*(LL_saturiert - LL_H0)
    lrt_p     : p-Wert unter χ²(df=k-1); asymptotisch, konservativ bei wenigen Sätzen
    lrt_df    : Freiheitsgrade (Anzahl Sätze - 1)
    """
    from scipy.stats import chi2 as scipy_chi2

    if not sets:
        return 0.5, 0, 0, float('nan'), float('nan'), float('nan'), 0

    player_points = sum(a for a, _ in sets)
    opp_points    = sum(b for _, b in sets)
    n             = player_points + opp_points

    def neg_log_likelihood(p):
        if p <= 0 or p >= 1:
            return 1e10
        ll = sum(math.log(max(set_likelihood(a, b, p), 1e-300)) for a, b in sets)
        return -ll

    result    = minimize_scalar(neg_log_likelihood, bounds=(0.001, 0.999), method='bounded')
    p_mle     = result.x
    ll_h0     = -neg_log_likelihood(p_mle)
    fit_score = ll_h0 / n if n > 0 else float('nan')

    # Likelihood-Ratio-Test
    # H1 (saturiert): jeder Satz hat seinen eigenen p_i = Ballanteil des Satzes
    # D = 2 * (LL_saturiert - LL_H0), df = k - 1
    k = len(sets)
    ll_saturated = 0.0
    for a, b in sets:
        total = a + b
        if total > 0:
            p_i = a / total
            p_i = max(min(p_i, 1 - 1e-10), 1e-10)
            ll_saturated += math.log(max(set_likelihood(a, b, p_i), 1e-300))

    lrt_D  = 2.0 * (ll_saturated - ll_h0)
    lrt_D  = max(lrt_D, 0.0)   # numerische Absicherung
    lrt_df = max(k - 1, 1)
    lrt_p  = 1.0 - scipy_chi2.cdf(lrt_D, df=lrt_df) if lrt_df > 0 else float('nan')

    return p_mle, player_points, opp_points, fit_score, lrt_D, lrt_p, lrt_df



# ---------------------------------------------------------
# Bootstrap: Konfidenzintervall für Performance Rating
# ---------------------------------------------------------

def bootstrap_ttr(sets_list: list, ttr_opponents: list, best_of: int = 5,
                  n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """
    Parametric Bootstrap für das Performance Rating auf Rally-Ebene.

    p_rally_i* ~ Beta(pp_i, op_i)  – direkt aus den Rally-Counts, kein Prior.
    Alles andere (p_set, p_match, TTR) folgt deterministisch durch die Kaskade.
    """
    rng = np.random.default_rng(42)

    rally_counts = []
    for sets in sets_list:
        _, pp, op, *_ = mle_rally_prob(sets)
        rally_counts.append((max(pp, 0.1), max(op, 0.1)))

    ttr_samples = []
    for _ in range(n_boot):
        p_match_boot = []
        for (pp, op), ttr_opp in zip(rally_counts, ttr_opponents):
            p_star = rng.beta(pp, op)
            p_star = max(min(p_star, 0.999), 0.001)
            _, pm  = cascade(p_star, best_of)
            p_match_boot.append(pm)
        try:
            ttr_b = tagesform_ttr_multi(p_match_boot, ttr_opponents, best_of)
            ttr_samples.append(ttr_b)
        except Exception:
            pass

    ttr_samples = np.array(ttr_samples)
    alpha_tail  = (1 - ci) / 2
    lower = float(np.quantile(ttr_samples, alpha_tail))
    upper = float(np.quantile(ttr_samples, 1 - alpha_tail))
    mean  = float(np.mean(ttr_samples))
    return mean, lower, upper, ttr_samples

# ---------------------------------------------------------
# 8. Parsing
# ---------------------------------------------------------
# Subset-Analyse
# ---------------------------------------------------------

def run_subset_analysis(rows: list, label: str, best_of: int, n_boot: int):
    """
    Führt vollständige Analyse für eine Teilmenge von table_rows durch.
    Gibt None zurück wenn zu wenig Daten.
    """
    if not rows:
        return None

    parsed_sets   = [r["sets"]    for r in rows]
    ttr_opponents = [r["ttr_opp"] for r in rows]
    p_match_list  = [r["p_match"] for r in rows]

    try:
        ttr_hat = tagesform_ttr_multi(p_match_list, ttr_opponents, best_of)
        _, ci_low, ci_high, boot_samples = bootstrap_ttr(
            parsed_sets, ttr_opponents, best_of, n_boot=n_boot)
    except Exception:
        return None

    n    = len(rows)
    wins = sum(1 for r in rows if sum(a > b for a,b in r["sets"]) >
                                   sum(b > a for a,b in r["sets"]))
    avg_ttr  = sum(r["ttr_opp"] for r in rows) / n
    avg_pm   = sum(p_match_list) / n
    boot_std = float(np.std(boot_samples))
    boot_med = float(np.median(boot_samples))
    s1_lo    = float(np.quantile(boot_samples, 0.1587))
    s1_hi    = float(np.quantile(boot_samples, 0.8413))
    skew     = boot_med - ttr_hat

    return {
        "label":        label,
        "n":            n,
        "wins":         wins,
        "ttr_hat":      ttr_hat,
        "ci_low":       ci_low,
        "ci_high":      ci_high,
        "s1_lo":        s1_lo,
        "s1_hi":        s1_hi,
        "boot_std":     boot_std,
        "boot_med":     boot_med,
        "skew":         skew,
        "avg_ttr":      avg_ttr,
        "avg_pm":       avg_pm,
        "boot_samples": boot_samples,
    }


def render_subset(res: dict):
    """Rendert eine vollständige Subset-Analyse (wie Gesamtsaison)."""
    ttr_hat      = res["ttr_hat"]
    ci_low       = res["ci_low"]
    ci_high      = res["ci_high"]
    s1_lo        = res["s1_lo"]
    s1_hi        = res["s1_hi"]
    boot_std     = res["boot_std"]
    boot_med     = res["boot_med"]
    skew         = res["skew"]
    boot_samples = res["boot_samples"]
    n            = res["n"]
    wins         = res["wins"]
    avg_ttr      = res["avg_ttr"]
    avg_pm       = res["avg_pm"]

    skew_color = "#34d399" if skew > 5 else "#f87171" if skew < -5 else "#6b7280"
    skew_str   = f"{skew:+.0f}"

    # Performance Rating Box
    st.markdown(f"""
    <div class="ttr-result-box">
      <div>
        <div class="ttr-label">Performance Rating</div>
        <div class="ttr-value">{ttr_hat:.0f}</div>
      </div>
      <div style="flex:1;display:flex;gap:1.5rem;flex-wrap:wrap;">
        <div><div class="ttr-label">Spiele</div>
             <div style="font-size:1.4rem;font-weight:600;color:#e8eaf0;font-family:'DM Mono',monospace">{wins}W – {n-wins}L</div></div>
        <div><div class="ttr-label">Ø TTR Gegner</div>
             <div style="font-size:1.4rem;font-weight:600;color:#e8eaf0;font-family:'DM Mono',monospace">{avg_ttr:.0f}</div></div>
        <div><div class="ttr-label">Ø p̂ Match</div>
             <div style="font-size:1.4rem;font-weight:600;color:#e8eaf0;font-family:'DM Mono',monospace">{avg_pm:.1%}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KI-Box
    st.markdown(f"""
    <div style="display:flex;gap:1.2rem;margin:1rem 0;flex-wrap:wrap;">
      <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:0.8rem 1.2rem;">
        <div class="ttr-label">95%-KI (2σ)</div>
        <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:#e8eaf0;font-weight:600;">
          [{ci_low:.0f},&nbsp;{ci_high:.0f}]
        </div>
      </div>
      <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:0.8rem 1.2rem;">
        <div class="ttr-label">68%-KI (1σ)</div>
        <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:#e8eaf0;font-weight:600;">
          [{s1_lo:.0f},&nbsp;{s1_hi:.0f}]
        </div>
      </div>
      <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:0.8rem 1.2rem;">
        <div class="ttr-label">σ</div>
        <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:#e8eaf0;font-weight:600;">
          {boot_std:.0f}
        </div>
      </div>
      <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:0.8rem 1.2rem;">
        <div class="ttr-label">Median</div>
        <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:#e8eaf0;font-weight:600;">
          {boot_med:.0f}
          <span style="font-size:0.85rem;color:{skew_color};margin-left:0.4rem;">{skew_str}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Asymmetrie-Hinweis
    if ttr_hat < s1_lo or ttr_hat > s1_hi:
        direction = "unter" if ttr_hat < s1_lo else "über"
        st.info(f"ℹ️ **Asymmetrie:** Performance Rating ({ttr_hat:.0f}) liegt {direction} dem 1σ-KI [{s1_lo:.0f}, {s1_hi:.0f}].")

    # Histogramm
    b_min = int(np.floor(boot_samples.min() / 10) * 10)
    b_max = int(np.ceil(boot_samples.max()  / 10) * 10)
    n_bins = max(10, (b_max - b_min) // 10)
    hist_counts, hist_edges = np.histogram(boot_samples, bins=n_bins, range=(b_min, b_max))
    bin_labels = [int(round((hist_edges[i]+hist_edges[i+1])/2)) for i in range(len(hist_counts))]
    df_hist = pd.DataFrame({"TTR": bin_labels, "Häufigkeit": hist_counts.astype(float)}).set_index("TTR")
    st.bar_chart(df_hist["Häufigkeit"], color="#3d8ef8")
    st.caption(
        f"MoM = {ttr_hat:.0f}  |  Median = {boot_med:.0f} ({skew_str})  |  "
        f"σ = {boot_std:.0f}  |  1σ-KI: [{s1_lo:.0f}, {s1_hi:.0f}]  |  "
        f"2σ-KI: [{ci_low:.0f}, {ci_high:.0f}]"
    )



def parse_set_scores(score_str: str) -> list:
    """
    Zwei Eingabeformate werden unterstuetzt:
    Klassisch:  "11:9, 3:11, 11:4, 13:11"
    Kurzformat: "+9 -3 4 +11"  (ohne Vorzeichen = Satzgewinn)
      +n / n  Spieler gewinnt: n<=9->11:n, n=10->12:10, n=11->13:11
      -n      Gegner gewinnt:  n<=9->n:11, n=10->10:12, n=11->11:13
    """
    import re as _re
    sets = []
    if not score_str.strip():
        return sets
    short_tokens = _re.findall(r"[+\-]?\d+", score_str)
    classic_tokens = [p for p in score_str.replace(";", " ").replace(",", " ").split() if ":" in p]
    if short_tokens and not classic_tokens:
        for token in short_tokens:
            if token.startswith("-"):
                sign, n = -1, int(token[1:])
            elif token.startswith("+"):
                sign, n = 1, int(token[1:])
            else:
                sign, n = 1, int(token)
            w = 11 + max(0, n - 9)
            sets.append((w, n) if sign == 1 else (n, w))
        return sets
    for part in score_str.replace(";", ",").replace(" ", ",").split(","):
        if ":" not in part:
            continue
        try:
            a, b = part.strip().split(":")
            sets.append((int(a), int(b)))
        except ValueError:
            pass
    return sets
def build_win_prob_curve(center_ttr: float) -> pd.DataFrame:
    diffs = list(range(-400, 401, 20))
    probs = []
    for diff in diffs:
        ttr_player = center_ttr
        ttr_opp    = center_ttr - diff
        probs.append(win_probability(ttr_player, ttr_opp))
    return pd.DataFrame({"ΔTTR": diffs, "Siegwahrscheinlichkeit": probs})


# ---------------------------------------------------------
# 10. Streamlit App
# ---------------------------------------------------------

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hintergrund ── */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161920 !important;
    border-right: 1px solid #2a2d3a;
}
[data-testid="stSidebar"] * { color: #c8cad4 !important; }

/* ── Header-Block ── */
.tt-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #2a2d3a;
    margin-bottom: 2rem;
}
.tt-title {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
}
.tt-subtitle {
    font-size: 0.85rem;
    color: #6b7280;
    font-weight: 300;
    font-family: 'DM Mono', monospace;
}

/* ── Match-Karten ── */
.match-card {
    background: #161920;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem 1rem 1.4rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.match-card:hover { border-color: #3d8ef8; }
.match-card-title {
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    color: #4b5563;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Ergebnis-Tabelle ── */
.result-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin: 1rem 0 1.5rem 0;
}
.result-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4b5563;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #2a2d3a;
    text-align: right;
}
.result-table th:first-child { text-align: left; }
.result-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid #1e2130;
    color: #c8cad4;
    text-align: right;
    font-family: 'DM Mono', monospace;
}
.result-table td:first-child {
    text-align: left;
    color: #ffffff;
    font-weight: 500;
}
.result-table tr:last-child td { border-bottom: none; }
.result-table tr:hover td { background: #1a1e2b; }

/* ── Gewinn/Verlust Badges ── */
.badge-win {
    background: #0d2e1a;
    color: #34d399;
    border: 1px solid #065f46;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
}
.badge-loss {
    background: #2d0e0e;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
}

/* ── TTR Result-Box ── */
.ttr-result-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #161920 100%);
    border: 1px solid #3d8ef8;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.ttr-label {
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    color: #4b5563;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.ttr-value {
    font-size: 3.5rem;
    font-weight: 600;
    color: #3d8ef8;
    letter-spacing: -0.04em;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}

/* ── Section-Header ── */
.section-header {
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: #4b5563;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2d3a;
}

/* ── Inputs ── */
input[type="number"], input[type="text"] {
    background: #0f1117 !important;
    border: 1px solid #2a2d3a !important;
    color: #e8eaf0 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}
input:focus { border-color: #3d8ef8 !important; }

/* ── Button ── */
.stButton > button {
    background: #3d8ef8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #2563eb !important; }

/* ── Metric Chips ── */
.prob-chips {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.3rem;
}
.prob-chip {
    background: #1a1e2b;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
}
.prob-chip-label { color: #4b5563; font-size: 0.65rem; display: block; margin-bottom: 0.1rem; }
.prob-chip-val   { color: #e8eaf0; }

/* ── Hinweis ── */
.hint-text {
    font-size: 0.78rem;
    color: #4b5563;
    font-family: 'DM Mono', monospace;
    margin-top: 0.3rem;
}
</style>
"""


def fmt_prob(p: float):
    q = 1 - p
    small = min(p, q)
    side  = "p" if p < q else "1−p"
    if 0.05 <= p <= 0.95:
        p_str = f"{p:.1%}"
    else:
        p_str = f"{p:.6f}"
    if small < 0.001:
        comp_str = f"{side} = {small:.2e}"
    elif small < 0.01:
        comp_str = f"{side} = {small:.5f}"
    else:
        comp_str = f"{side} = {small:.3f}"
    return p_str, comp_str


def prob_bar(p: float, color: str = "#3d8ef8") -> str:
    pct = min(max(p * 100, 0), 100)
    p_str, comp_str = fmt_prob(p)
    return (
        f'<div style="display:flex;flex-direction:column;gap:0.2rem;">'
        f'  <div style="display:flex;align-items:center;gap:0.6rem;">'
        f'    <div style="flex:1;background:#1e2130;border-radius:3px;height:5px;">'
        f'      <div style="width:{pct:.1f}%;background:{color};border-radius:3px;height:5px;"></div>'
        f'    </div>'
        f'    <span style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:#e8eaf0;min-width:5rem;">{p_str}</span>'
        f'  </div>'
        f'  <span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#4b5563;">{comp_str}</span>'
        f'</div>'
    )


def result_badge(sets) -> str:
    won  = sum(1 for a, b in sets if a > b)
    lost = sum(1 for a, b in sets if b > a)
    if won > lost:
        return f'<span class="badge-win">{won}:{lost} ✓</span>'
    else:
        return f'<span class="badge-loss">{won}:{lost} ✗</span>'



# ---------------------------------------------------------
# Web-Import: bettv.tischtennislive.de Parser
# ---------------------------------------------------------

def _normalize_bettv_url(url: str) -> str:
    url = url.strip()
    if "tischtennislive.de/?" in url:
        url = url.replace("tischtennislive.de/?", "tischtennislive.de/default.aspx?")
    return url

def _fetch_html(url: str) -> str:
    url = _normalize_bettv_url(url)
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text

def _fetch_ttr_all(url: str) -> str:
    """
    Lädt alle TTR-Einträge indem alle Saison-Filter einzeln abgefragt
    und die Ergebnisse als kombiniertes HTML zurückgegeben werden.
    """
    url = _normalize_bettv_url(url)
    headers = {"User-Agent": "Mozilla/5.0"}

    # Erst GET um verfügbare Saison-Optionen zu lesen
    session = requests.Session()
    session.headers.update(headers)
    r0 = session.get(url, timeout=15)
    r0.raise_for_status()
    r0.encoding = "utf-8"
    html0 = r0.text

    # Saison-Optionen aus Select auslesen
    soup0 = BeautifulSoup(html0, "html.parser")
    zeit_select = soup0.find("select", {"name": "Zeit"})
    if not zeit_select:
        # Kein Filter-Formular → direkt zurückgeben
        return html0

    options = []
    for opt in zeit_select.find_all("option"):
        val = opt.get("value", "").strip()
        options.append(val)

    # Alle Saison-Einträge (S20xx) + leerer Wert (Alle) abfragen
    # Wir fragen nur Saison-Optionen ab (nicht M6/M12/M24 die Duplikate erzeugen)
    saison_options = [v for v in options if v.startswith("S") or v == ""]

    combined_tables = []
    for zeit_val in saison_options:
        try:
            s = requests.Session()
            s.headers.update(headers)
            s.get(url, timeout=15)  # Session/Cookie aufbauen
            r = s.post(url, data={"Zeit": zeit_val, "Design": "0"},
                       timeout=20,
                       headers={"Referer": url,
                                "Content-Type": "application/x-www-form-urlencoded"})
            r.raise_for_status()
            r.encoding = "utf-8"
            soup_r = BeautifulSoup(r.text, "html.parser")
            # Alle Tabellen mit TTR-Daten extrahieren (erkennbar an "vs." im Text)
            for tbl in soup_r.find_all("table"):
                if "vs." in tbl.get_text():
                    combined_tables.append(str(tbl))
        except Exception:
            continue

    if not combined_tables:
        return html0  # Fallback auf GET-Ergebnis

    # Kombiniertes HTML: alle Tabellen in einem wrapper
    return "<html><body>" + "\n".join(combined_tables) + "</body></html>"

def parse_ergebnis_page(html: str) -> list:
    """
    Parst Vorrunde/Rückrunde-Seite von bettv.tischtennislive.de.
    Gibt Liste von Einzel-Spielen zurück (Doppel werden ignoriert).

    Struktur: Pro Punktspiel können mehrere Tabellen existieren (eine pro Spiel).
    Erste Tabelle hat Datumszeile, Folgetabellen nicht → current_date tabellenübergreifend.
    Doppel-Erkennung: TD direkt vor der Tabelle enthält "Doppel".
    Duplikate (gleiche Datum+Gegner) werden am Ende entfernt.
    """
    soup = BeautifulSoup(html, "html.parser")
    matches = []
    current_date = None
    current_mannschaft = None
    current_heimgast = ""
    seen = set()  # Deduplizierung: (datum, gegner)

    for table in soup.find_all("table"):
        # Doppel-Tabelle: der TD direkt vor dieser Tabelle enthält "Doppel"
        prev_td = table.find_previous("td")
        if prev_td and "Doppel" in prev_td.get_text():
            continue

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            texts = [c.get_text(strip=True) for c in cells]

            if len(cells) >= 8 and re.match(r"\d{2}\.\d{2}\.\d{4}", texts[1]):
                # Erste Zeile eines Punktspiels:
                # [0]=leer [1]=Datum [2]=H/G [3]=Mannschaft [4]=Erg [5]=Gegner [6]=Sätze [7]=Erg
                current_date       = texts[1]
                current_heimgast   = texts[2]  # "H" oder "G"
                current_mannschaft = texts[3]
                gegner   = texts[5]
                saetze   = texts[6]
                ergebnis = texts[7]
            elif len(cells) == 6:
                # Folgezeile: [0]=leer [1]=leer [2]=colspan-3-leer [3]=Gegner [4]=Sätze [5]=Erg
                gegner   = texts[3]
                saetze   = texts[4]
                ergebnis = texts[5]
            else:
                continue

            if gegner in ("", "Gegenspieler") or not saetze or not ergebnis:
                continue
            if not current_date:
                continue

            key = (current_date, gegner)
            if key in seen:
                continue
            seen.add(key)

            matches.append({
                "datum":      current_date,
                "gegner":     gegner,
                "mannschaft": current_mannschaft or "",
                "heimgast":   current_heimgast,
                "saetze":     saetze,
                "ergebnis":   ergebnis,
            })
    return matches

def parse_ttr_page(html: str) -> dict:
    """
    Parst EntwicklungTTR-Seite.
    Erste Zeile:   12 TDs: [0]=leer [1]=Datum [2]=H/G [3]=Mannschaft [4]=Erg [5]=Gegner
                            [6,7,8]=OnlySmallScreen [9]=TTR [10]=+/- [11]=Erg
    Folgezeile:     7 TDs: [0]=leer [1]=leer [2]=colspan3 [3]=Gegner
                            [4]=TTR [5]=+/- [6]=Erg
    """
    soup = BeautifulSoup(html, "html.parser")
    ttr_map    = {}
    livepz_map = {}
    current_date = None
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 7:
                continue
            texts = [c.get_text(strip=True) for c in cells]

            if len(cells) >= 10 and re.match(r"\d{2}\.\d{2}\.\d{4}", texts[1]):
                # Erste Zeile: [0]=leer [1]=Datum [2]=H/G [3]=Mannschaft [4]=Erg
                #              [5]=Gegner [6]=colspan3 [7]=TTR [8]=+/- [9]=Erg
                current_date = texts[1]
                gegner   = texts[5]
                ttr_info = texts[7]
                delta_str = texts[8] if len(texts) > 8 else ""
            elif len(cells) == 8:
                # Folgezeile: [0]=leer [1]=leer [2]=colspan3 [3]=Gegner
                #             [4]=colspan3 [5]=TTR [6]=+/- [7]=Erg
                gegner    = texts[3]
                ttr_info  = texts[5]
                delta_str = texts[6] if len(texts) > 6 else ""
            else:
                continue

            if not current_date or not gegner or gegner == "Gegenspieler":
                continue

            m = re.search(r"(\d{3,4}) vs\.\s*(\d{3,4})", ttr_info)
            if m:
                eigener_ttr = int(m.group(1))
                ttr_map[(current_date, gegner)] = int(m.group(2))
                # Eigenen TTR + Delta für LivePZ-Verlauf speichern
                try:
                    delta = int(re.sub(r"[^\d\-+]", "", delta_str))
                    livepz_nach = eigener_ttr + delta
                except (ValueError, TypeError):
                    livepz_nach = eigener_ttr
                livepz_map[(current_date, gegner)] = {
                    "ttr_vor":    eigener_ttr,
                    "ttr_nach":   livepz_nach,
                    "delta":      delta_str,
                }
    return ttr_map, livepz_map

def merge_matches_with_ttr(matches: list, ttr_map: dict,
                            livepz_map: dict = None) -> list:
    """
    Verknüpft Spielergebnisse mit TTR-Werten des Gegners.
    Matching-Strategie (in Reihenfolge):
      1. Exakt: (datum, gegner)
      2. Fallback: nur gegner (nützlich wenn TTR-Seite zeitlich begrenzt ist)
    Rückgabe: (merged_list, match_stats_dict)
    """
    if livepz_map is None:
        livepz_map = {}
    # Fallback-Map: gegner → [(datum, ttr), ...] – nimm den Eintrag mit nächstem Datum
    from collections import defaultdict
    gegner_map = defaultdict(list)
    for (datum, gegner), ttr in ttr_map.items():
        gegner_map[gegner].append((datum, ttr))

    merged = []
    stats  = {"exact": 0, "fallback": 0, "missing": 0}

    for m in matches:
        key = (m["datum"], m["gegner"])
        if key in ttr_map:
            ttr = ttr_map[key]
            stats["exact"] += 1
        elif m["gegner"] in gegner_map:
            # Nächstgelegenes Datum wählen
            entries = gegner_map[m["gegner"]]
            def _date_dist(entry):
                try:
                    from datetime import datetime
                    d1 = datetime.strptime(m["datum"], "%d.%m.%Y")
                    d2 = datetime.strptime(entry[0], "%d.%m.%Y")
                    return abs((d1 - d2).days)
                except Exception:
                    return 0
            best = min(entries, key=_date_dist)
            ttr  = best[1]
            stats["fallback"] += 1
        else:
            stats["missing"] += 1
            continue

        # LivePZ nach dem Spiel (letztes Spiel des Tages = aktuellster Wert)
        lpz = livepz_map.get(key, {})

        merged.append({
            "datum":       m["datum"],
            "gegner":      m["gegner"],
            "mannschaft":  m["mannschaft"],
            "heimgast":    m.get("heimgast", ""),
            "runde":       m.get("runde", ""),
            "ttr_gegner":  ttr,
            "livepz_vor":  lpz.get("ttr_vor"),
            "livepz_nach": lpz.get("ttr_nach"),
            "livepz_delta":lpz.get("delta", ""),
            "saetze":      m["saetze"],
            "ergebnis":    m["ergebnis"],
        })
    return merged, stats

def main():
    st.set_page_config(page_title="TT Performance Rating", layout="wide", page_icon="🏓")
    st.markdown(STYLE, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="tt-header">
      <div class="tt-title">🏓 TT Performance Rating</div>
      <div class="tt-subtitle">Rally · Satz · Match → Performance Rating via Maximum-Likelihood</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    st.sidebar.markdown("### Einstellungen")
    best_of     = st.sidebar.selectbox("Modus", [3, 5, 7],
                                        index=1, format_func=lambda x: f"Best of {x}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="hint-text">Satzergebnisse eingeben als<br>'
        '<b>11:7, 9:11, 11:8</b><br>'
        '(Komma, Semikolon oder Leerzeichen)</div>',
        unsafe_allow_html=True
    )

    # ── Tabs ──
    tab_manuell, tab_web, tab_konzept, tab_glossar = st.tabs(["✏️ Manuelle Eingabe", "🌐 Web-Import (bettv)", "💡 Konzept", "📖 Glossar"])

    # ════════════════════════════════════════════════════════
    # TAB: Web-Import
    # ════════════════════════════════════════════════════════
    with tab_web:
        st.markdown('<div class="section-header">Web-Import</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hint-text">Füge eine beliebige bettv-URL des Spielers ein – '
            'die App erkennt automatisch Staffel und Spieler und lädt Vorrunde, Rückrunde '
            'und TTR-Entwicklung.<br>'
            '<b>Doppel werden automatisch ignoriert.</b></div>',
            unsafe_allow_html=True
        )

        url_input = st.text_input(
            "bettv-URL",
            placeholder="https://bettv.tischtennislive.de/default.aspx?...&L2P=8009&L3P=97662&..."
        )

        def _extract_bettv_params(url: str):
            """Extrahiert L2P und L3P aus einer bettv-URL."""
            import urllib.parse
            params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            l2p = params.get("L2P", [None])[0]
            l3p = params.get("L3P", [None])[0]
            return l2p, l3p

        def _build_bettv_url(l2p: str, l3p: str, page: str) -> str:
            return (f"https://bettv.tischtennislive.de/default.aspx"
                    f"?L1=Ergebnisse&L2=TTStaffeln&L2P={l2p}&L3=Spieler&L3P={l3p}&Page={page}")

        if st.button("🔄 Daten laden", key="btn_web_import"):
            if not url_input.strip():
                st.error("Bitte eine bettv-URL eingeben.")
            else:
                l2p, l3p = _extract_bettv_params(url_input.strip())
                if not l2p or not l3p:
                    st.error("❌ Konnte L2P (Staffel-ID) oder L3P (Spieler-ID) nicht aus der URL lesen. "
                             "Bitte eine vollständige bettv-URL einfügen.")
                else:
                    url_vr  = _build_bettv_url(l2p, l3p, "Vorrunde")
                    url_rr  = _build_bettv_url(l2p, l3p, "Rueckrunde")
                    url_ttr = _build_bettv_url(l2p, l3p, "EntwicklungTTR")
                    st.caption(f"Staffel-ID: {l2p} · Spieler-ID: {l3p}")

                    with st.spinner("Lade Daten von bettv.tischtennislive.de…"):
                        try:
                            # TTR-Seite laden (mit erweitertem Datumsbereich via POST)
                            try:
                                ttr_html = _fetch_ttr_all(url_ttr)
                            except Exception as e:
                                st.error(f"❌ TTR-Seite nicht erreichbar: {e}")
                                st.stop()
                            ttr_map, livepz_map = parse_ttr_page(ttr_html)
                            n_ttr = len(ttr_map)
                            st.info(f"TTR-Seite geladen: {n_ttr} Einträge gefunden.")
                            if not ttr_map:
                                st.warning("⚠️ TTR-Seite geladen, aber keine Ratings gefunden.")

                            # Ergebnis-Seiten laden
                            all_matches = []
                            for url, label in [(url_vr, "Vorrunde"), (url_rr, "Rückrunde")]:
                                try:
                                    html = _fetch_html(url)
                                except Exception as e:
                                    st.error(f"❌ {label}-Seite nicht erreichbar: {e}")
                                    continue
                                ms = parse_ergebnis_page(html)
                                st.info(f"{label}: {len(ms)} Einzel-Spiele geparst.")
                                if not ms:
                                    st.warning(f"⚠️ {label}: keine Spiele gefunden.")
                                for m in ms:
                                    m["runde"] = label
                                all_matches.extend(ms)

                            merged, mstats = merge_matches_with_ttr(
                                all_matches, ttr_map, livepz_map)

                            if not merged:
                                st.error("❌ Keine Übereinstimmungen zwischen Ergebnissen und TTR-Werten.")
                                if all_matches:
                                    sample_keys = [(m["datum"], m["gegner"]) for m in all_matches[:3]]
                                    st.write("Beispiel-Keys aus Ergebnissen:", sample_keys)
                                if ttr_map:
                                    sample_ttr = list(ttr_map.items())[:3]
                                    st.write("Beispiel-Keys aus TTR-Seite:", sample_ttr)
                                st.info("Tipp: Datum und Gegnername müssen auf beiden Seiten identisch sein.")
                            else:
                                st.session_state["web_matches"] = merged
                                msg = f"{len(merged)} Einzel-Spiele geladen."
                                if mstats["fallback"] > 0:
                                    msg += (f" ℹ️ {mstats['fallback']} Spiel(e) ohne exaktes Datum-Match – "
                                            f"TTR-Wert nach Gegner zugeordnet.")
                                if mstats["missing"] > 0:
                                    msg += f" ⚠️ {mstats['missing']} Spiel(e) ohne TTR-Wert übersprungen."
                                st.success(msg)

                        except Exception as e:
                            st.error(f"Fehler beim Laden: {e}")

        # Geladene Spiele anzeigen und auswählen
        if "web_matches" in st.session_state:
            merged = st.session_state["web_matches"]

            st.markdown('<div class="section-header">Geladene Spiele</div>', unsafe_allow_html=True)

            # Tabelle mit Checkboxen zur Auswahl
            header_web = (
                "<table class='result-table'><thead><tr>"
                "<th>#</th><th>Datum</th><th>Runde</th><th>Gegner</th>"
                "<th>TTR Gegner</th><th>Ergebnis</th><th>Sätze</th>"
                "</tr></thead><tbody>"
            )
            rows_web = ""
            for i, m in enumerate(merged):
                won, lost = m["ergebnis"].split(":")
                badge = f'<span class="badge-win">{won}:{lost} ✓</span>' if int(won) > int(lost)                         else f'<span class="badge-loss">{won}:{lost} ✗</span>'
                rows_web += (
                    f"<tr>"
                    f"<td>{i+1}</td>"
                    f"<td style='font-size:0.8rem'>{m['datum']}</td>"
                    f"<td style='font-size:0.8rem;color:#6b7280'>{m.get('runde','')}</td>"
                    f"<td style='font-size:0.85rem'>{m['gegner']}</td>"
                    f"<td style='font-family:DM Mono,monospace'>{m['ttr_gegner']}</td>"
                    f"<td>{badge}</td>"
                    f"<td style='font-family:DM Mono,monospace;font-size:0.8rem'>{m['saetze']}</td>"
                    f"</tr>"
                )
            st.markdown(header_web + rows_web + "</tbody></table>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Selektion welche Spiele analysiert werden sollen
            st.markdown("**Spiele für Analyse auswählen:**")
            select_all = st.checkbox("Alle auswählen", value=True, key="web_select_all")
            selected_indices = []
            if not select_all:
                cols_sel = st.columns(4)
                for i, m in enumerate(merged):
                    with cols_sel[i % 4]:
                        label = f"{m['datum']} · {m['gegner'][:15]}"
                        if st.checkbox(label, value=True, key=f"web_sel_{i}"):
                            selected_indices.append(i)
            else:
                selected_indices = list(range(len(merged)))

            if st.button("📊 In manuelle Eingabe übernehmen", key="btn_web_analyse") and selected_indices:
                selected = [merged[i] for i in selected_indices]
                st.session_state["web_prefill"]      = selected
                st.session_state["_prefill_applied"] = None  # Slider-Reset erzwingen
                st.session_state.pop("calc_results", None)   # alte Analyse löschen
                st.session_state["switch_to_manuell"] = True
                st.rerun()

        # Tab-Wechsel via JavaScript (muss außerhalb der with-Blöcke stehen)
        if st.session_state.pop("switch_to_manuell", False):
            st.components.v1.html("""
            <script>
            // Ersten Tab (✏️ Manuelle Eingabe) anklicken
            const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
            if (tabs.length > 0) tabs[0].click();
            </script>
            """, height=0)

    # ════════════════════════════════════════════════════════
    # TAB: Manuelle Eingabe
    # ════════════════════════════════════════════════════════
    with tab_manuell:
        # ── Spieleingabe ──
        st.markdown('<div class="section-header">Spieleingabe</div>', unsafe_allow_html=True)
        # Web-Import Prefill
        prefill = st.session_state.get("web_prefill", [])

        # Slider-Wert und Widget-Werte korrekt setzen wenn neues Prefill kam
        if prefill and st.session_state.get("_prefill_applied") != len(prefill):
            st.session_state["num_matches_slider"] = len(prefill)
            # Prefill-Werte direkt in session_state schreiben (überschreibt gecachte Werte)
            for i, pf in enumerate(prefill):
                st.session_state[f"ttr_{i}"]    = pf["ttr_gegner"]
                st.session_state[f"scores_{i}"] = pf["saetze"]
            st.session_state["_prefill_applied"] = len(prefill)

        if "num_matches_slider" not in st.session_state:
            st.session_state["num_matches_slider"] = 3

        slider_max = max(50, st.session_state["num_matches_slider"])
        num_matches = st.slider("Anzahl Spiele", 1, slider_max,
                                key="num_matches_slider")

        match_inputs = []

        if num_matches <= 6:
            # Kartenansicht für wenige Spiele
            cols = st.columns(min(num_matches, 3))
            for i in range(num_matches):
                with cols[i % 3]:
                    st.markdown(
                        f'<div class="match-card-title">Spiel {i+1}</div>',
                        unsafe_allow_html=True
                    )
                    ttr_opp    = st.number_input("TTR Gegner", 0, 3000, step=10,
                                                  key=f"ttr_{i}", label_visibility="visible")
                    scores_str = st.text_input("Satzergebnisse", key=f"scores_{i}",
                                                placeholder="z.B. 11:7, 9:11, 11:8")
                    match_inputs.append((ttr_opp, scores_str))
        else:
            # Kompakte Tabellenansicht für viele Spiele
            st.markdown(
                '<div class="hint-text" style="margin-bottom:0.6rem;">'
                'Kompaktansicht – leere Zeilen werden übersprungen.</div>',
                unsafe_allow_html=True
            )
            h1, h2, h3 = st.columns([0.4, 1.2, 3.0])
            h1.markdown('<div style="font-size:0.72rem;color:#6b7280;padding-bottom:2px;">#</div>', unsafe_allow_html=True)
            h2.markdown('<div style="font-size:0.72rem;color:#6b7280;padding-bottom:2px;">TTR Gegner</div>', unsafe_allow_html=True)
            h3.markdown('<div style="font-size:0.72rem;color:#6b7280;padding-bottom:2px;">Satzergebnisse</div>', unsafe_allow_html=True)

            for i in range(num_matches):
                c1, c2, c3 = st.columns([0.4, 1.2, 3.0])
                c1.markdown(
                    f'<div style="font-size:0.8rem;color:#6b7280;padding-top:0.55rem;text-align:right;">{i+1}</div>',
                    unsafe_allow_html=True
                )
                ttr_opp    = c2.number_input("TTR", 0, 3000, step=10,
                                              key=f"ttr_{i}", label_visibility="collapsed")
                scores_str = c3.text_input("Sätze", key=f"scores_{i}",
                                            placeholder="11:7, 9:11, 11:8",
                                            label_visibility="collapsed")
                match_inputs.append((ttr_opp, scores_str))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bootstrap-Einstellung (vor Berechnen, damit kein Rerun-Verlust) ──
        n_boot = st.sidebar.slider("Bootstrap-Samples", 500, 5000, 2000, step=500)

        # ── Berechnen ──
        if st.button("Berechnen", use_container_width=False):

            ttr_opponents = []
            p_match_list  = []
            table_rows    = []
            parsed_sets   = []

            for idx, (ttr_opp, scores_str) in enumerate(match_inputs):
                sets = parse_set_scores(scores_str)
                if not sets:
                    continue

                p_hat, player_points, opp_points, fit, lrt_D, lrt_p, lrt_df = mle_rally_prob(sets)
                if player_points + opp_points == 0:
                    continue

                p_set, p_match = cascade(p_hat, best_of)

                ttr_opponents.append(ttr_opp)
                p_match_list.append(p_match)
                parsed_sets.append(sets)

                pf = prefill[idx] if idx < len(prefill) else None

                table_rows.append({
                    "idx":            idx + 1,
                    "ttr_opp":        ttr_opp,
                    "gegner":         pf["gegner"]          if pf else "",
                    "datum":          pf["datum"]           if pf else "",
                    "mannschaft":     pf["mannschaft"]      if pf else "",
                    "heimgast":       pf.get("heimgast", "") if pf else "",
                    "runde":          pf.get("runde", "")    if pf else "",
                    "livepz_vor":     pf.get("livepz_vor")   if pf else None,
                    "livepz_nach":    pf.get("livepz_nach")  if pf else None,
                    "player_points":  player_points,
                    "opp_points":     opp_points,
                    "sets":           sets,
                    "p_rally":        p_hat,
                    "p_set":          p_set,
                    "p_match":        p_match,
                    "fit":            fit,
                    "lrt_D":          lrt_D,
                    "lrt_p":          lrt_p,
                    "lrt_df":         lrt_df,
                })

            if not table_rows:
                st.warning("Keine gültigen Spieleingaben gefunden. Bitte Satzergebnisse eintragen.")
            else:
                # Gruppen bilden
                has_meta = any(r["datum"] for r in table_rows)
                groups   = []
                if has_meta:
                    seen_keys = []
                    group_map = {}
                    for r in table_rows:
                        key = (r["datum"], r["mannschaft"])
                        if key not in group_map:
                            group_map[key] = []
                            seen_keys.append(key)
                        group_map[key].append(r)
                    groups = [(k, group_map[k]) for k in seen_keys]
                else:
                    groups = [(("", ""), table_rows)]

                # Punktspiel-Bootstrap vorab
                grp_stats = {}
                if has_meta:
                    with st.spinner("Berechne Punktspiel-Ratings…"):
                        for (datum, mannschaft), grp in groups:
                            key = (datum, mannschaft)
                            grp_sets = [r["sets"]    for r in grp]
                            grp_ttr  = [r["ttr_opp"] for r in grp]
                            grp_pm   = [r["p_match"] for r in grp]
                            ttr_hat_grp = tagesform_ttr_multi(grp_pm, grp_ttr, best_of)
                            _, _, _, boot = bootstrap_ttr(grp_sets, grp_ttr, best_of, n_boot=1000)
                            livepz_entries = [
                                (r["livepz_vor"], r["livepz_nach"])
                                for r in grp
                                if r.get("livepz_vor") is not None
                                and r.get("livepz_nach") is not None
                            ]
                            if livepz_entries:
                                ttr_vor_ps = livepz_entries[0][0]   # Eingangs-TTR des Punktspiels
                                delta_sum  = sum(nach - vor for vor, nach in livepz_entries)
                                livepz_ps  = ttr_vor_ps + delta_sum
                            else:
                                livepz_ps = None
                            grp_stats[key] = {
                                "ttr_hat": ttr_hat_grp,
                                "s1_lo":   float(np.quantile(boot, 0.1587)),
                                "s1_hi":   float(np.quantile(boot, 0.8413)),
                                "sigma":   float(np.std(boot)),
                                "livepz":  livepz_ps,
                            }

                # Gesamt-Bootstrap
                ttr_hat = tagesform_ttr_multi(p_match_list, ttr_opponents, best_of)
                with st.spinner("Bootstrap läuft..."):
                    _, ci_low, ci_high, boot_samples = bootstrap_ttr(
                        parsed_sets, ttr_opponents, best_of, n_boot=n_boot)

                # Alles in session_state speichern
                st.session_state["calc_results"] = {
                    "table_rows":    table_rows,
                    "parsed_sets":   parsed_sets,
                    "ttr_opponents": ttr_opponents,
                    "p_match_list":  p_match_list,
                    "has_meta":      has_meta,
                    "groups":        groups,
                    "grp_stats":     grp_stats,
                    "ttr_hat":       ttr_hat,
                    "ci_low":        ci_low,
                    "ci_high":       ci_high,
                    "boot_samples":  boot_samples,
                    "best_of":       best_of,
                    "n_boot":        n_boot,
                }

        # ── Ergebnisse anzeigen (aus session_state) ──
        res = st.session_state.get("calc_results")
        if res:
            table_rows    = res["table_rows"]
            parsed_sets   = res["parsed_sets"]
            ttr_opponents = res["ttr_opponents"]
            p_match_list  = res["p_match_list"]
            has_meta      = res["has_meta"]
            groups        = res["groups"]
            grp_stats     = res["grp_stats"]
            ttr_hat       = res["ttr_hat"]
            ci_low        = res["ci_low"]
            ci_high       = res["ci_high"]
            boot_samples  = res["boot_samples"]

            # ── Ergebnistabelle ──
            st.markdown('<div class="section-header">Ergebnisse pro Spiel</div>',
                        unsafe_allow_html=True)

            def lrt_badge(lrt_p, lrt_df):
                if math.isnan(lrt_p):
                    return '<span style="font-size:0.72rem;color:#4b5563;">–</span>'
                pct = lrt_p * 100
                mono = "font-family:'DM Mono',monospace;"
                if lrt_p >= 0.10:
                    col   = "#34d399"
                    label = "✓ konsistent"
                elif lrt_p >= 0.05:
                    col   = "#f59e0b"
                    label = "⚠ auffällig"
                elif lrt_p >= 0.01:
                    col   = "#f87171"
                    label = "✗ inkonsistent"
                else:
                    col   = "#ef4444"
                    label = "✗✗ stark inkonsistent"
                asym = (
                    '<span style="font-size:0.60rem;color:#4b5563;"> *</span>'
                    if lrt_df < 4 else ""
                )
                return (
                    f'<span style="{mono}font-size:0.72rem;color:{col};">{label}</span>{asym}'
                    f'<br><span style="{mono}font-size:0.68rem;color:#6b7280;">{pct:.1f}%</span>'
                )

            header = (
                "<table class='result-table'>"
                "<thead><tr>"
                "<th>Spiel</th>"
                "<th>TTR Gegner</th>"
                "<th>Ergebnis</th>"
                "<th>Punkte</th>"
                "<th>p&#772; Rally</th>"
                "<th>p&#772; Satz</th>"
                "<th>p&#772; Match</th>"
                "<th>Konstanz <span style='font-size:0.65rem;font-weight:400;color:#6b7280;'>(&#967;&#178;-Test)</span></th>"
                "</tr></thead><tbody>"
            )
            rows_html = ""
            has_asym  = False

            for (datum, mannschaft), grp in groups:
                # Punktspiel-Header
                if has_meta and datum:
                    rows_html += (
                        f"<tr style='background:#1e2130;'>"
                        f"<td colspan='8' style='font-size:0.78rem;color:#9ca3af;"
                        f"padding:0.35rem 0.6rem;font-family:DM Sans,sans-serif;'>"
                        f"📅 {datum} &nbsp;·&nbsp; {mannschaft}</td></tr>"
                    )

                for r in grp:
                    color_rally = "#34d399" if r["p_rally"] >= 0.5 else "#f87171"
                    color_match = "#34d399" if r["p_match"] >= 0.5 else "#f87171"
                    gegner_cell = (
                        f"<span style='font-size:0.78rem;color:#9ca3af;display:block;'>{r['gegner']}</span>"
                        if r["gegner"] else ""
                    )
                    rows_html += (
                        f"<tr>"
                        f"<td>#{r['idx']}{gegner_cell}</td>"
                        f"<td>{r['ttr_opp']}</td>"
                        f"<td>{result_badge(r['sets'])}</td>"
                        f"<td style='font-size:0.8rem'>{r['player_points']} : {r['opp_points']}</td>"
                        f"<td>{prob_bar(r['p_rally'], color_rally)}</td>"
                        f"<td>{prob_bar(r['p_set'])}</td>"
                        f"<td>{prob_bar(r['p_match'], color_match)}</td>"
                        f"<td>{lrt_badge(r['lrt_p'], r['lrt_df'])}</td>"
                        f"</tr>"
                    )
                    if r["lrt_df"] < 4:
                        has_asym = True

                # Zusammenfassungszeile pro Punktspiel
                if has_meta:
                    gs = grp_stats[(datum, mannschaft)]
                    rows_html += (
                        f"<tr style='background:#161920;border-top:1px solid #2a2d3a;'>"
                        f"<td colspan='7' style='font-size:0.76rem;color:#6b7280;"
                        f"padding:0.35rem 0.6rem 0.5rem;font-family:\"DM Mono\",monospace;'>"
                        f"&#x2514; Punktspiel-TTR:&nbsp;"
                        f"<span style='color:#e8eaf0;font-weight:600;font-size:0.85rem;'>{gs['ttr_hat']:.0f}</span>"
                        f"&ensp;1σ-KI:&nbsp;"
                        f"<span style='color:#c4c9d4;'>[{gs['s1_lo']:.0f},&thinsp;{gs['s1_hi']:.0f}]</span>"
                        f"&ensp;σ&nbsp;=&nbsp;"
                        f"<span style='color:#c4c9d4;'>{gs['sigma']:.0f}</span>"
                        f"</td><td></td></tr>"
                        f"<tr><td colspan='8' style='padding:0.2rem;'></td></tr>"  # Abstandszeile
                    )

            has_asym_flag = has_asym
            asym_note = (
                "<tr><td colspan='8' style='font-size:0.65rem;color:#4b5563;padding:0.4rem 0.6rem;'>"
                "* &#967;&#178;-Asymptotik bei &lt; 4 S&#228;tzen konservativ &#8211; p-Wert als Richtwert interpretieren."
                "</td></tr>"
            ) if has_asym_flag else ""
            st.markdown(header + rows_html + asym_note + "</tbody></table>", unsafe_allow_html=True)
            # ── Modellfit-Warnungen ──
            bad_fits = [r for r in table_rows if not math.isnan(r["lrt_p"]) and r["lrt_p"] < 0.10]
            if bad_fits:
                for r in bad_fits:
                    pct = r["lrt_p"] * 100
                    if r["lrt_p"] >= 0.05:
                        level = "⚠️"
                        text  = (f"Der Satzverlauf ist leicht auffällig (χ²-Test: p = {pct:.1f}%). "
                                 f"Ein so ungleichmäßiger Satzverlauf tritt in {pct:.1f}% der Fälle auf, "
                                 f"selbst wenn die Tagesform konstant war.")
                    elif r["lrt_p"] >= 0.01:
                        level = "🚨"
                        text  = (f"Die Tagesform war wahrscheinlich nicht konstant (χ²-Test: p = {pct:.1f}%). "
                                 f"Nur in {pct:.1f}% der Fälle wäre dieser Satzverlauf bei konstanter "
                                 f"Spielstärke zu erwarten. Das Rating mit erhöhter Unsicherheit interpretieren.")
                    else:
                        level = "🚨"
                        text  = (f"Die Tagesform war sehr wahrscheinlich nicht konstant (χ²-Test: p = {pct:.1f}%). "
                                 f"Dieser Satzverlauf ist unter der Modellannahme extrem unwahrscheinlich. "
                                 f"Das Rating hat eingeschränkte Aussagekraft.")
                    st.warning(f"{level} **Spiel #{r['idx']}:** {text}")

            # ── Performance Rating ──
            n        = len(table_rows)
            wins     = sum(1 for r in table_rows if sum(a > b for a,b in r["sets"]) >
                                                     sum(b > a for a,b in r["sets"]))
            avg_ttr  = sum(r["ttr_opp"] for r in table_rows) / n
            avg_pm   = sum(p_match_list) / n

            st.markdown('<div class="section-header">Performance Rating</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="ttr-result-box">
              <div>
                <div class="ttr-label">Performance Rating</div>
                <div class="ttr-value">{ttr_hat:.0f}</div>
              </div>
              <div style="flex:1;display:flex;gap:1.5rem;flex-wrap:wrap;">
                <div><div class="ttr-label">Spiele</div>
                     <div style="font-size:1.4rem;font-weight:600;color:#e8eaf0;font-family:'DM Mono',monospace">{wins}W – {n-wins}L</div></div>
                <div><div class="ttr-label">Ø TTR Gegner</div>
                     <div style="font-size:1.4rem;font-weight:600;color:#e8eaf0;font-family:'DM Mono',monospace">{avg_ttr:.0f}</div></div>
                <div><div class="ttr-label">Ø p̂ Match</div>
                     <div style="font-size:1.4rem;font-weight:600;color:#e8eaf0;font-family:'DM Mono',monospace">{avg_pm:.1%}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Bootstrap-Statistiken ──
            boot_std    = float(np.std(boot_samples))
            boot_median = float(np.median(boot_samples))
            skew        = boot_median - ttr_hat
            s1_low      = float(np.quantile(boot_samples, 0.1587))
            s1_high     = float(np.quantile(boot_samples, 0.8413))
            total_rallys = sum(r["player_points"]+r["opp_points"] for r in table_rows)

            skew_color = "#34d399" if skew > 5 else "#f87171" if skew < -5 else "#6b7280"
            skew_str   = f"{skew:+.0f}"

            st.markdown('<div class="section-header">Bootstrap-Konfidenzintervall</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div style="display:flex;gap:1.2rem;margin-bottom:1.5rem;flex-wrap:wrap;">
              <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:1rem 1.4rem;">
                <div class="ttr-label">95%-KI (2σ)</div>
                <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#e8eaf0;font-weight:600;">
                  [{ci_low:.0f},&nbsp;{ci_high:.0f}]
                </div>
              </div>
              <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:1rem 1.4rem;">
                <div class="ttr-label">68%-KI (1σ)</div>
                <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#e8eaf0;font-weight:600;">
                  [{s1_low:.0f},&nbsp;{s1_high:.0f}]
                </div>
              </div>
              <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:1rem 1.4rem;">
                <div class="ttr-label">σ (Standardunsicherheit)</div>
                <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#e8eaf0;font-weight:600;">
                  {boot_std:.0f}
                </div>
              </div>
              <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:1rem 1.4rem;">
                <div class="ttr-label">Median</div>
                <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#e8eaf0;font-weight:600;">
                  {boot_median:.0f}
                  <span style="font-size:0.85rem;color:{skew_color};margin-left:0.4rem;">{skew_str}</span>
                </div>
              </div>
              <div style="background:#161920;border:1px solid #2a2d3a;border-radius:10px;padding:1rem 1.4rem;">
                <div class="ttr-label">Rallys (Datenbasis)</div>
                <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#e8eaf0;font-weight:600;">
                  {total_rallys}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Hinweis bei starker Asymmetrie (MoM außerhalb 1σ-KI)
            if ttr_hat < s1_low or ttr_hat > s1_high:
                direction = "unter" if ttr_hat < s1_low else "über"
                st.info(
                    f"ℹ️ **Asymmetrie:** Das Performance Rating ({ttr_hat:.0f}) liegt {direction} dem "
                    f"1σ-Konfidenzintervall [{s1_low:.0f}, {s1_high:.0f}]. "
                    f"Dies tritt bei stark einseitigen Serien auf (Über- oder Unterforderung) und ist eine "
                    f"strukturelle Eigenschaft der nichtlinearen Rally→Match-Kaskade, kein Fehler. "
                    f"Der Median ({boot_median:.0f}) liegt näher am Zentrum der Bootstrap-Verteilung."
                )

            # Histogramm
            b_min = int(np.floor(boot_samples.min() / 10) * 10)
            b_max = int(np.ceil(boot_samples.max()  / 10) * 10)
            n_bins = max(10, (b_max - b_min) // 10)
            hist_counts, hist_edges = np.histogram(boot_samples, bins=n_bins, range=(b_min, b_max))
            bin_labels = [int(round((hist_edges[i]+hist_edges[i+1])/2)) for i in range(len(hist_counts))]
            df_hist = pd.DataFrame({
                "TTR": bin_labels, "Häufigkeit": hist_counts.astype(float)
            }).set_index("TTR")

            st.markdown('<div class="section-header">Bootstrap-Verteilung des Performance Ratings</div>',
                        unsafe_allow_html=True)
            st.bar_chart(df_hist["Häufigkeit"], color="#3d8ef8")
            st.caption(
                f"MoM-TTR = {ttr_hat:.0f}  |  "
                f"Median = {boot_median:.0f} ({skew_str})  |  "
                f"σ = {boot_std:.0f}  |  "
                f"68% KI: [{s1_low:.0f}, {s1_high:.0f}]  |  "
                f"95% KI: [{ci_low:.0f}, {ci_high:.0f}]"
            )

            # ── Saisonverlauf (nur bei Web-Import mit Metadaten) ──
            if has_meta and grp_stats:
                st.markdown('<div class="section-header">Saisonverlauf</div>',
                            unsafe_allow_html=True)

                show_trend    = False
                show_rolling  = False
                show_livepz   = False

                toggle_cols = st.columns([1, 1, 1])
                with toggle_cols[0]:
                    if len(groups) >= 3:
                        show_trend = st.toggle("📈 Trend", value=False)
                with toggle_cols[1]:
                    if len(groups) >= 2:
                        show_rolling = st.toggle("🔄 Gleitend", value=False)
                with toggle_cols[2]:
                    livepz_available = any(gs.get("livepz") is not None
                                           for gs in grp_stats.values())
                    if livepz_available:
                        show_livepz = st.toggle("🏅 LivePZ", value=False)

                # Datenpunkte: ein Eintrag pro Punktspiel
                ps_x      = []
                ps_ttr    = []
                ps_s1_lo  = []
                ps_s1_hi  = []
                ps_sigma  = []
                ps_label  = []
                ps_livepz = []   # LivePZ nach Punktspiel (None wenn nicht verfügbar)

                for i, ((datum, mannschaft), grp) in enumerate(groups):
                    gs = grp_stats[(datum, mannschaft)]
                    wins_grp = sum(1 for r in grp
                                   if sum(a > b for a,b in r["sets"]) >
                                      sum(b > a for a,b in r["sets"]))
                    livepz = gs.get("livepz")
                    ps_x.append(i + 1)
                    ps_ttr.append(gs["ttr_hat"])
                    ps_s1_lo.append(gs["s1_lo"])
                    ps_s1_hi.append(gs["s1_hi"])
                    ps_sigma.append(max(gs["sigma"], 1.0))
                    ps_livepz.append(livepz)
                    livepz_str = f"<br>LivePZ nach PS: {livepz}" if livepz else ""
                    ps_label.append(
                        f"<b>Punktspiel {i+1}</b><br>"
                        f"{datum}<br>"
                        f"{mannschaft}<br>"
                        f"Performance Rating: {gs['ttr_hat']:.0f}<br>"
                        f"1σ-KI: [{gs['s1_lo']:.0f}, {gs['s1_hi']:.0f}]<br>"
                        f"σ = {gs['sigma']:.0f}<br>"
                        f"W/L: {wins_grp}/{len(grp)-wins_grp}"
                        f"{livepz_str}"
                    )

                # Bootstrap-Dichte für Hintergrundband
                boot_min  = float(boot_samples.min())
                boot_max  = float(boot_samples.max())
                n_band    = 200
                band_y    = np.linspace(boot_min, boot_max, n_band)
                from scipy.stats import gaussian_kde, t as t_dist
                kde       = gaussian_kde(boot_samples)
                band_dens = kde(band_y)
                band_dens = band_dens / band_dens.max()

                fig = go.Figure()

                # 1) Bootstrap-Dichte als Hintergrund
                for j in range(n_band - 1):
                    alpha = float((band_dens[j] + band_dens[j+1]) / 2)
                    fig.add_shape(
                        type="rect",
                        x0=0.5, x1=len(groups) + 0.5,
                        y0=float(band_y[j]), y1=float(band_y[j+1]),
                        fillcolor=f"rgba(100,120,180,{alpha * 0.35:.3f})",
                        line_width=0, layer="below",
                    )

                # 2) 2σ-Grenzen (gestrichelt)
                for y_val, label in [(ci_low, f"−2σ: {ci_low:.0f}"),
                                     (ci_high, f"+2σ: {ci_high:.0f}")]:
                    fig.add_hline(y=y_val, line_color="#3d6ea8", line_width=1,
                                  line_dash="dash",
                                  annotation_text=label, annotation_position="right",
                                  annotation_font=dict(color="#3d6ea8", size=9))

                # 3) 1σ-Grenzen (gestrichpunktet)
                for y_val, label in [(s1_low, f"−1σ: {s1_low:.0f}"),
                                     (s1_high, f"+1σ: {s1_high:.0f}")]:
                    fig.add_hline(y=y_val, line_color="#5b8fe8", line_width=1,
                                  line_dash="dashdot",
                                  annotation_text=label, annotation_position="right",
                                  annotation_font=dict(color="#5b8fe8", size=9))

                # 4) Performance Rating (durchgezogen)
                fig.add_hline(y=ttr_hat, line_color="#3d8ef8", line_width=3,
                              annotation_text=f"PR: {ttr_hat:.0f}",
                              annotation_position="right",
                              annotation_font=dict(color="#3d8ef8", size=10))

                # 5) Linearer Trend (gewichtet nach 1/σ²)
                if show_trend and len(ps_x) >= 3:
                    xs  = np.array(ps_x,    dtype=float)
                    ys  = np.array(ps_ttr,  dtype=float)
                    ws  = 1.0 / np.array(ps_sigma) ** 2   # Gewichte: präzisere Punkte zählen mehr

                    # Gewichtete lineare Regression via least squares
                    W   = np.diag(ws)
                    X   = np.column_stack([np.ones_like(xs), xs])
                    XtW = X.T @ W
                    coeffs = np.linalg.solve(XtW @ X, XtW @ ys)
                    intercept, slope = coeffs

                    y_fit = intercept + slope * xs

                    # Konfidenzband (95%) des Fits
                    n     = len(xs)
                    dof   = n - 2
                    resid = ys - y_fit
                    s2    = float(np.sum(ws * resid**2) / dof)   # gewichtete Residualvarianz
                    Sw    = float(np.sum(ws))
                    Swx   = float(np.sum(ws * xs))
                    Swx2  = float(np.sum(ws * xs**2))
                    denom = Sw * Swx2 - Swx**2

                    x_plot = np.linspace(xs.min(), xs.max(), 100)
                    se_fit = np.sqrt(s2 * (Sw * (x_plot - Swx/Sw)**2 / denom + 1.0/Sw))
                    t_crit  = t_dist.ppf(0.975, dof)
                    t_crit1 = t_dist.ppf(0.8413, dof)  # ~1σ
                    y_plot = intercept + slope * x_plot
                    ci2_up = y_plot + t_crit  * se_fit
                    ci2_dn = y_plot - t_crit  * se_fit
                    ci1_up = y_plot + t_crit1 * se_fit
                    ci1_dn = y_plot - t_crit1 * se_fit

                    # Standardfehler der Fitparameter
                    # Var(slope)     = s2 * Sw / denom
                    # Var(intercept) = s2 * Swx2 / denom
                    se_slope     = float(np.sqrt(s2 * Sw    / denom))
                    se_intercept = float(np.sqrt(s2 * Swx2  / denom))

                    # 95%-Konfidenzband (2σ, außen)
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_plot, x_plot[::-1]]).tolist(),
                        y=np.concatenate([ci2_up, ci2_dn[::-1]]).tolist(),
                        fill="toself",
                        fillcolor="rgba(250,180,50,0.10)",
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    ))

                    # 68%-Konfidenzband (1σ, innen)
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_plot, x_plot[::-1]]).tolist(),
                        y=np.concatenate([ci1_up, ci1_dn[::-1]]).tolist(),
                        fill="toself",
                        fillcolor="rgba(250,180,50,0.20)",
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    ))

                    # Trend-Gerade
                    trend_dir   = "↑" if slope > 0 else "↓"
                    trend_color = "#34d399" if slope > 0 else "#f87171"
                    fig.add_trace(go.Scatter(
                        x=x_plot.tolist(),
                        y=y_plot.tolist(),
                        mode="lines",
                        line=dict(color=trend_color, width=2, dash="longdash"),
                        hovertemplate=(
                            f"<b>Linearer Trend</b><br>"
                            f"Steigung: {slope:+.1f} pro Punktspiel<extra></extra>"
                        ),
                        showlegend=False,
                    ))

                    # Trend-Kennzahl unter dem Chart
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:{trend_color};'
                        f'font-family:DM Mono,monospace;margin-top:0.3rem;">'
                        f'{trend_dir} Steigung: {slope:+.2f} ± {se_slope:.2f} Punkte/Punktspiel'
                        f'</div>'
                        f'<div style="font-size:0.78rem;color:#6b7280;'
                        f'font-family:DM Mono,monospace;margin-top:0.1rem;">'
                        f'Achsenabschnitt: {intercept:.0f} ± {se_intercept:.0f}'
                        f'&ensp;·&ensp;Freiheitsgrade: {dof}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                elif show_trend and len(ps_x) < 3:
                    st.info("Mindestens 3 Punktspiele für Trendberechnung erforderlich.")

                # 6) Punktspiel-Ratings mit Fehlerbalken
                fig.add_trace(go.Scatter(
                    x=ps_x,
                    y=ps_ttr,
                    mode="markers+lines",
                    name="Performance Rating",
                    marker=dict(color="#e8eaf0", size=8, symbol="circle"),
                    line=dict(color="#6b7280", width=1, dash="dot"),
                    error_y=dict(
                        type="data", symmetric=False,
                        array=[hi - m for hi, m in zip(ps_s1_hi, ps_ttr)],
                        arrayminus=[m - lo for m, lo in zip(ps_ttr, ps_s1_lo)],
                        color="#9ca3af", thickness=1.5, width=6,
                    ),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=ps_label,
                ))

                # 7) Gleitender MLE-Verlauf (rollierendes Fenster bis 5 Punktspiele)
                if show_rolling and len(ps_x) >= 2:
                    with st.spinner("Berechne gleitenden Verlauf…"):
                        roll_x    = []
                        roll_ttr  = []
                        roll_s1lo = []
                        roll_s1hi = []
                        roll_2slo = []
                        roll_2shi = []
                        roll_label = []

                        for i in range(len(groups)):
                            window_start = max(0, i - 4)  # bis 5 Punktspiele
                            window = list(groups[window_start:i+1])
                            w_rows = [r for _, grp in window for r in grp]
                            w_sets = [r["sets"]    for r in w_rows]
                            w_ttr  = [r["ttr_opp"] for r in w_rows]
                            w_pm   = [r["p_match"] for r in w_rows]
                            try:
                                w_pr = tagesform_ttr_multi(w_pm, w_ttr, best_of)
                                _, w_ci_lo, w_ci_hi, w_boot = bootstrap_ttr(
                                    w_sets, w_ttr, best_of, n_boot=500)
                                w_s1lo = float(np.quantile(w_boot, 0.1587))
                                w_s1hi = float(np.quantile(w_boot, 0.8413))
                                n_ps_window = i - window_start + 1
                                roll_x.append(i + 1)
                                roll_ttr.append(w_pr)
                                roll_s1lo.append(w_s1lo)
                                roll_s1hi.append(w_s1hi)
                                roll_2slo.append(w_ci_lo)
                                roll_2shi.append(w_ci_hi)
                                roll_label.append(
                                    f"<b>Gleitender Verlauf PS {i+1}</b><br>"
                                    f"Fenster: PS {window_start+1}–{i+1} "
                                    f"({n_ps_window} Punktspiel{'e' if n_ps_window>1 else ''})<br>"
                                    f"Rating: {w_pr:.0f}<br>"
                                    f"1σ-KI: [{w_s1lo:.0f}, {w_s1hi:.0f}]<br>"
                                    f"2σ-KI: [{w_ci_lo:.0f}, {w_ci_hi:.0f}]"
                                )
                            except Exception:
                                pass

                    if roll_x:
                        # 2σ-Band
                        fig.add_trace(go.Scatter(
                            x=roll_x + roll_x[::-1],
                            y=roll_2shi + roll_2slo[::-1],
                            fill="toself",
                            fillcolor="rgba(52,211,153,0.07)",
                            mode="lines",
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False,
                        ))
                        # 1σ-Band
                        fig.add_trace(go.Scatter(
                            x=roll_x + roll_x[::-1],
                            y=roll_s1hi + roll_s1lo[::-1],
                            fill="toself",
                            fillcolor="rgba(52,211,153,0.15)",
                            mode="lines",
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False,
                        ))
                        # Gleitende Linie
                        fig.add_trace(go.Scatter(
                            x=roll_x,
                            y=roll_ttr,
                            mode="lines+markers",
                            name="Gleitend (5 PS)",
                            marker=dict(color="#34d399", size=5, symbol="circle"),
                            line=dict(color="#34d399", width=2, dash="solid"),
                            hovertemplate="%{customdata}<extra></extra>",
                            customdata=roll_label,
                        ))

                # 8) LivePZ-Verlauf (toggle-bar)
                livepz_x = [x for x, v in zip(ps_x, ps_livepz) if v is not None]
                livepz_y = [v for v in ps_livepz if v is not None]
                if show_livepz and livepz_x:
                    fig.add_trace(go.Scatter(
                        x=livepz_x,
                        y=livepz_y,
                        mode="markers+lines",
                        name="LivePZ",
                        marker=dict(color="#f59e0b", size=6, symbol="diamond"),
                        line=dict(color="#f59e0b", width=1.5, dash="solid"),
                        hovertemplate="<b>LivePZ nach Punktspiel %{x}</b><br>%{y}<extra></extra>",
                    ))

                fig.update_layout(
                    paper_bgcolor="#0f1117",
                    plot_bgcolor="#161920",
                    font=dict(family="DM Sans, sans-serif", color="#9ca3af", size=11),
                    xaxis=dict(title="Punktspiel", tickmode="linear", tick0=1, dtick=1,
                               gridcolor="#2a2d3a", zeroline=False),
                    yaxis=dict(title="Rating", gridcolor="#2a2d3a", zeroline=False),
                    showlegend=show_rolling or (show_livepz and bool(livepz_x)),
                    legend=dict(
                        bgcolor="#161920", bordercolor="#2a2d3a", borderwidth=1,
                        font=dict(size=10, color="#9ca3af"),
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    ),
                    margin=dict(l=50, r=140, t=50, b=50),
                    height=480,
                )

                st.plotly_chart(fig, use_container_width=True)

            # ── Subset-Analysen (nur bei Web-Import) ──
            if has_meta:
                st.markdown('<div class="section-header">Subset-Analysen</div>',
                            unsafe_allow_html=True)

                # Subsets definieren
                all_groups_flat = [(datum, mannschaft, grp)
                                   for (datum, mannschaft), grp in groups]
                n_ps = len(groups)

                subsets = []

                # Vorrunde / Rückrunde
                vr_rows = [r for r in table_rows if r.get("runde") == "Vorrunde"]
                rr_rows = [r for r in table_rows if r.get("runde") == "Rückrunde"]
                if vr_rows:
                    subsets.append(("📅 Nur Vorrunde", vr_rows))
                if rr_rows:
                    subsets.append(("📅 Nur Rückrunde", rr_rows))

                # Heim / Auswärts
                heim_rows = [r for r in table_rows if r.get("heimgast", "").upper() == "H"]
                gast_rows = [r for r in table_rows if r.get("heimgast", "").upper() == "G"]
                if heim_rows:
                    subsets.append(("🏠 Nur Heimspiele", heim_rows))
                if gast_rows:
                    subsets.append(("✈️ Nur Auswärtsspiele", gast_rows))

                # Letzte 5 Punktspiele (nur ab 6 Punktspielen)
                if n_ps >= 6:
                    last5_keys = set(k for k, _ in groups[-5:])
                    last5_rows = [r for r in table_rows
                                  if (r["datum"], r["mannschaft"]) in last5_keys]
                    subsets.append(("🕐 Letzte 5 Punktspiele", last5_rows))

                if not subsets:
                    st.markdown(
                        '<div class="hint-text">Keine Subset-Daten verfügbar – '
                        'Vorrunde/Rückrunde und Heim/Auswärts werden aus dem Web-Import übernommen.</div>',
                        unsafe_allow_html=True)
                else:
                    with st.spinner("Berechne Subset-Analysen…"):
                        subset_results = []
                        for label, rows in subsets:
                            r = run_subset_analysis(rows, label, best_of, n_boot=1000)
                            if r:
                                subset_results.append(r)

                    for sr in subset_results:
                        with st.expander(f"{sr['label']} · {sr['wins']}W–{sr['n']-sr['wins']}L · PR: {sr['ttr_hat']:.0f}"):
                            render_subset(sr)


    # ════════════════════════════════════════════════════════
    # TAB: Konzept
    # ════════════════════════════════════════════════════════
    with tab_konzept:
        st.markdown('<div class="section-header">Was macht diese App?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.95rem;color:#c4c9d4;line-height:1.8;max-width:800px;">
        Tischtennis-Ratings wie TTR messen die <em>langfristige</em> Spielstärke – sie verändern sich
        langsam und reagieren kaum auf eine einzelne Saison. Diese App berechnet stattdessen ein
        <strong>Performance Rating</strong>: eine Schätzung der tatsächlichen Spielstärke, die sich
        allein aus den Satzergebnissen einer Saison oder eines Turniers ergibt.<br><br>
        Die Grundfrage lautet: <em>Welches TTR würde diese Ergebnisse am besten erklären –
        unabhängig davon, ob man gewonnen oder verloren hat?</em><br><br>
        Damit lassen sich Fragen beantworten wie: War diese Saison eine Über- oder Unterleistung?
        Wie stark war die Spielstärke bei einem einzelnen Punktspiel? Und wie sicher ist diese Schätzung?
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Wie funktioniert das?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.95rem;color:#c4c9d4;line-height:1.8;max-width:800px;">
        Das Modell behandelt Tischtennis als reines Zufallsspiel: Jeder einzelne Punkt (Rally)
        hat eine feste Wahrscheinlichkeit, gewonnen zu werden. Alles andere – Satz, Match, Rating –
        folgt daraus durch reine Mathematik. Keine Domänenkenntnisse, keine Ausnahmen.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Schritt 1 · Der einzelne Punkt (Rally)"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Angenommen, ein Spieler gewinnt jeden Punkt mit einer bestimmten Wahrscheinlichkeit –
            nennen wir sie <strong>p</strong>. Ist p = 0.5, sind beide gleich stark.
            Ist p = 0.55, gewinnt man etwas öfter als der Gegner.<br><br>
            Diese Wahrscheinlichkeit p kann man nicht direkt messen – aber man kann sie
            <em>schätzen</em>: Wenn ein Spieler in allen Sätzen einer Partie zusammen
            47 von 100 Punkten gewonnen hat, ist die naheliegendste Schätzung p = 0.47.
            Diese Methode heißt <strong>Maximum-Likelihood-Schätzung (MLE)</strong> –
            sie wählt den Wert, der die beobachteten Ergebnisse am wahrscheinlichsten macht.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Schritt 2 · Vom Punkt zum Satz"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Ein Satz geht bis 11 Punkte (mit Verlängerung ab 10:10 bis zum 2-Punkte-Vorsprung).
            Wenn man jeden Punkt mit Wahrscheinlichkeit p gewinnt, kann man exakt ausrechnen,
            wie wahrscheinlich ein Satzgewinn ist – genauso wie man die Gewinnchance beim
            Münzwurf berechnen kann, nur etwas komplizierter.<br><br>
            Diese Berechnung ist reine Mathematik (Binomialverteilung) und braucht
            keine Erfahrungswerte oder Faustregeln.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Schritt 3 · Vom Satz zum Match"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Ein Match geht über Best-of-3, Best-of-5 oder Best-of-7 Sätze.
            Wenn man die Wahrscheinlichkeit kennt, einen einzelnen Satz zu gewinnen,
            lässt sich genauso ausrechnen, wie wahrscheinlich ein Matchgewinn ist.<br><br>
            Diese Kette – <strong>p(Punkt) → p(Satz) → p(Match)</strong> – heißt
            <strong>Kaskade</strong>. Jede Stufe folgt deterministisch aus der vorherigen.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Schritt 4 · Vom Match zum Rating"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Das TTR-System sagt: Wenn zwei Spieler mit TTR-Werten A und B gegeneinander spielen,
            lässt sich die Siegwahrscheinlichkeit des Spielers A berechnen. Je größer der
            Unterschied, desto einseitiger die Partie.<br><br>
            Die App dreht diese Logik um: Gegeben die beobachtete Match-Wahrscheinlichkeit
            und das bekannte TTR des Gegners – welches TTR müsste der Spieler haben?
            Das Ergebnis ist das <strong>Performance Rating</strong>.<br><br>
            Es beantwortet die Frage: <em>Auf welchem Niveau habe ich heute tatsächlich gespielt?</em>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Schritt 5 · Wie sicher ist die Schätzung? (Bootstrap)"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Eine Schätzung ist immer mit Unsicherheit behaftet. Hätte man an einem anderen
            Tag gespielt, wären die Punktstände leicht anders ausgefallen – und damit auch
            das Rating.<br><br>
            Um diese Unsicherheit zu quantifizieren, verwendet die App <strong>Bootstrap</strong>:
            Das Verfahren simuliert 2000 alternative Versionen der gespielten Partien,
            indem es die Punkt-Wahrscheinlichkeit leicht variiert (nach den Regeln der
            Statistik). Für jede Version wird ein Rating berechnet. Die Streuung
            dieser 2000 Ratings ergibt das <strong>Konfidenzintervall</strong>.<br><br>
            Ein enges Intervall bedeutet: Die Schätzung ist robust, unabhängig vom
            Tagesglück. Ein weites Intervall bedeutet: Mit mehr Daten (mehr Partien,
            mehr Sätze) würde die Schätzung zuverlässiger.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Warum werden Sieg und Niederlage gleich behandelt?"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Das Modell wertet nicht, ob man gewonnen oder verloren hat – es schaut nur,
            <em>wie</em> die Punkte verteilt waren. Ein knappes 2:3 gegen einen viel
            stärkeren Gegner kann ein höheres Performance Rating ergeben als ein
            klares 3:0 gegen einen schwachen Gegner.<br><br>
            Das ist der entscheidende Unterschied zum offiziellen TTR: Dieses belohnt
            Siege und bestraft Niederlagen binär. Das Performance Rating fragt:
            <em>Wie stark habe ich heute tatsächlich gespielt?</em>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Was bedeutet ein Performance Rating außerhalb des Konfidenzintervalls?"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Das tritt bei stark einseitigen Serien auf – etwa 0 Siege aus 16 Partien
            gegen deutlich stärkere Gegner. Die Ursache ist mathematisch:
            Die Kaskade von Punkt zu Match ist nichtlinear. Kleine Änderungen in der
            Punkt-Wahrscheinlichkeit führen bei sehr ungleichen Partien zu überproportional
            großen Änderungen im Rating.<br><br>
            Das Bootstrap-Verfahren mittelt diese Asymmetrie teilweise heraus, weshalb
            der Median des Intervalls vom Punktschätzer abweichen kann.
            Das ist kein Fehler, sondern ein ehrlicher Hinweis auf die Grenzen des Modells
            bei extremen Datensituationen.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Saisonverlauf und Trendanalyse"):
            st.markdown("""
            <div style="font-size:0.92rem;color:#c4c9d4;line-height:1.7;">
            Wenn mehrere Punktspiele vorliegen, zeigt die App einen <strong>Saisonverlauf</strong>:
            für jedes Punktspiel ein eigenes Performance Rating mit Fehlerbalken.
            So lässt sich erkennen, ob einzelne Punktspiele aus dem Rahmen fallen –
            etwa weil die Gegner ungewöhnlich stark oder schwach waren.<br><br>
            Drei optionale Overlays lassen sich per Toggle einblenden:<br><br>
            <b>📈 Linearer Trend</b> – ab drei Punktspielen verfügbar. Eine gewichtete
            Regressionsgerade zeigt ob die Spielstärke im Saisonverlauf tendenziell
            zu- oder abnimmt. Punktspiele mit engerem Konfidenzintervall werden stärker
            gewichtet. Die Konfidenzbänder (1σ und 2σ) zeigen wie sicher der Trend ist.<br><br>
            <b>🔄 Gleitender Verlauf</b> – das Performance Rating aus dem aktuellen
            Punktspiel und bis zu vier vorherigen wird gemeinsam berechnet. Das ergibt
            einen geglätteten Verlauf der weniger empfindlich auf einzelne Ausreißer
            reagiert. Auch hier werden 1σ- und 2σ-Konfidenzbänder eingezeichnet.<br><br>
            <b>🏅 LivePZ-Verlauf</b> – zeigt den offiziellen LivePZ-Wert nach jedem
            Punktspiel als orangene Linie. Dient als Referenz zum Vergleich mit dem
            Performance Rating. Der LivePZ wird direkt aus der bettv-Seite gelesen:
            Eingangs-TTR des Punktspiels plus die Summe aller Einzeldeltas.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.88rem;color:#6b7280;line-height:1.8;">
        📄 Die vollständige statistische Methodik ist im
        <a href="https://github.com/torstenlanger/tt-performance-rating/blob/main/PAPER.md"
           target="_blank" style="color:#3d8ef8;">Paper auf GitHub</a> dokumentiert.
        &nbsp;·&nbsp;
        <a href="https://github.com/torstenlanger/tt-performance-rating"
           target="_blank" style="color:#3d8ef8;">Quellcode auf GitHub</a>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB: Glossar
    # ════════════════════════════════════════════════════════
    with tab_glossar:
        st.markdown('<div class="section-header">Glossar</div>', unsafe_allow_html=True)

        glossar = [
            ("p̄ Rally", "Geschätzte Wahrscheinlichkeit, einen einzelnen Rally (Punkt) zu gewinnen. "
             "Wird per Maximum-Likelihood aus allen Satzergebnissen einer Partie geschätzt. "
             "Sie ist die atomare Einheit des Modells – alle anderen Größen leiten sich daraus ab."),

            ("p̄ Satz", "Wahrscheinlichkeit, einen Satz zu gewinnen, berechnet aus p̄ Rally über "
             "eine analytische Formel (Binomialverteilung mit Verlängerungsregel ab 10:10)."),

            ("p̄ Match", "Wahrscheinlichkeit, das Match zu gewinnen, berechnet aus p̄ Satz im "
             "gewählten Best-of-Modus (Best of 3, 5 oder 7)."),

            ("Performance Rating", "Das TTR-äquivalente Rating, das die beobachteten Satzergebnisse "
             "einer Saison oder eines Punktspiels am besten erklärt. "
             "Berechnet per Maximum-Likelihood: p̄ Rally wird aus allen Rallies geschätzt, "
             "dann deterministisch durch die Kaskade (p̄ Rally → p̄ Satz → p̄ Match) transformiert. "
             "Gesucht wird der TTR-Wert, bei dem die erwartete Match-Gewinnwahrscheinlichkeit "
             "gegen alle Gegner der beobachteten entspricht. "
             "Diese Formulierung ist analytisch äquivalent zu Method of Moments – "
             "die Informationsgrundlage sind jedoch die einzelnen Rallies, "
             "nicht die binäre Sieg/Niederlage-Bilanz."),

            ("Bootstrap-Konfidenzintervall", "Schätzbereich für das Performance Rating unter "
             "Berücksichtigung der Stichprobenunsicherheit. Pro Bootstrap-Iteration wird "
             "p̄ Rally jeder Partie neu aus einer Beta-Verteilung gezogen (Beta(pp, op), "
             "wobei pp/op die gewonnenen/verlorenen Rallies sind), die Kaskade durchlaufen "
             "und ein neues Rating berechnet. Aus 2000 solcher Iterationen werden die "
             "Quantile bestimmt."),

            ("68%-KI (1σ)", "Konfidenzintervall das 68 % der Bootstrap-Verteilung umschließt "
             "(16.–84. Perzentile). Entspricht bei Normalverteilung dem ±1σ-Bereich. "
             "Gibt die wahrscheinlichste Schwankungsbreite des Ratings an."),

            ("95%-KI (2σ)", "Konfidenzintervall das 95 % der Bootstrap-Verteilung umschließt "
             "(2.5.–97.5. Perzentile). Entspricht bei Normalverteilung dem ±2σ-Bereich."),

            ("σ (Standardunsicherheit)", "Standardabweichung der Bootstrap-Verteilung. "
             "Maß für die Gesamtstreuung des geschätzten Ratings. Je mehr Rallies und Partien "
             "in die Schätzung eingehen, desto kleiner σ."),

            ("Median", "Mittelpunkt der Bootstrap-Verteilung (50. Perzentile). "
             "Bei symmetrischer Verteilung gleich dem Performance Rating. "
             "Eine deutliche Abweichung (Δ) zeigt Asymmetrie der Verteilung an, "
             "die bei stark einseitigen Serien (Über-/Unterforderung) durch die "
             "nichtlineare Rally→Match-Kaskade entsteht."),

            ("Konstanz (χ²-Test)", "Likelihood-Ratio-Test der Nullhypothese, dass p̄ Rally "
             "über alle Sätze einer Partie konstant war. Ein kleiner p-Wert (rot) deutet "
             "auf schwankende Spielstärke innerhalb der Partie hin. "
             "Grün (≥ 10 %): konsistent. Gelb (5–10 %): auffällig. "
             "Rot (1–5 %): inkonsistent. Dunkelrot (< 1 %): stark inkonsistent."),

            ("Punktspiel-Rating", "Performance Rating berechnet ausschließlich aus den "
             "Einzelspielen eines Punktspiels. Ermöglicht den Vergleich der Tagesform "
             "zwischen verschiedenen Punktspielen im Saisonverlauf."),

            ("Saisonverlauf-Diagramm", "Zeigt die Punktspiel-Ratings als Zeitverlauf mit "
             "asymmetrischen 1σ-Fehlerbalken. Die horizontale Linie markiert das "
             "Performance Rating der Gesamtsaison, gestrichelte Linien die ±1σ- und "
             "±2σ-Grenzen. Der grau-blaue Hintergrund zeigt die Bootstrap-Dichte "
             "(KDE) der Gesamtsaison – hellere Bereiche sind wahrscheinlicher. "
             "Drei optionale Overlays können per Toggle eingeblendet werden: "
             "linearer Trend, gleitender Verlauf und LivePZ-Referenz."),

            ("Linearer Trend", "Gewichtete lineare Regression durch die Punktspiel-Ratings. "
             "Punktspiele mit kleinerem σ (engeres Konfidenzintervall, also mehr Spiele und Rallies) "
             "werden stärker gewichtet als unsichere Schätzungen. "
             "Die Steigung gibt an, wie viele Rating-Punkte das Performance Rating "
             "pro Punktspiel im Schnitt zu- oder abnimmt. "
             "Das innere Band (dunkler) entspricht dem 68%-Konfidenzband (1σ) des Fits, "
             "das äußere Band (heller) dem 95%-Konfidenzband (2σ). "
             "Ein breites Band bedeutet: Die Datenlage reicht nicht aus um den Trend "
             "zuverlässig zu bestimmen."),

            ("Gleitender Verlauf", "Performance Rating berechnet aus dem jeweils aktuellen "
             "Punktspiel und bis zu vier vorherigen (rollierende Fenster von maximal 5 Punktspielen). "
             "Dadurch entsteht ein geglätteter Verlauf der weniger auf einzelne Ausreißer reagiert "
             "als die Einzelwerte. 1σ- und 2σ-Konfidenzbänder werden ebenfalls berechnet. "
             "Am Anfang der Saison (weniger als 5 Punktspiele verfügbar) wächst das Fenster "
             "schrittweise an."),

            ("LivePZ-Verlauf", "Zeigt den offiziellen LivePZ-Wert nach jedem Punktspiel "
             "als orangene Referenzlinie im Saisonverlaufsdiagramm. "
             "Der Wert wird aus der bettv-TTR-Entwicklungsseite berechnet: "
             "Eingangs-TTR des Punktspiels plus die Summe der +/−-Werte aller Einzelspiele. "
             "Dient als Vergleichsmaßstab zwischen offiziellem Rating und Performance Rating."),

            ("Subset-Analysen", "Vollständige Performance-Rating-Analysen für Teilmengen der "
             "geladenen Spiele – Vorrunde, Rückrunde, Heimspiele, Auswärtsspiele und die "
             "letzten 5 Punktspiele (ab 6 Punktspielen). "
             "Jedes Subset enthält Performance Rating, Konfidenzintervalle und Bootstrap-Histogramm. "
             "Ermöglicht den Vergleich der Spielstärke unter verschiedenen Bedingungen."),
        ]

        for term, erklärung in glossar:
            with st.expander(term):
                st.markdown(
                    f'<div style="font-size:0.92rem;color:#c4c9d4;line-height:1.6;">{erklärung}</div>',
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()