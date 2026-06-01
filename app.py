"""
=============================================================================
TABLEAU DE BORD INTERACTIF — SIMULATION D'UN SYSTÈME DE RETRAITE
Approche Monte-Carlo — 2025/2026
=============================================================================
Lancement :  streamlit run app.py
=============================================================================
"""

import math
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import simulation_retraite as sr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulation Discrète — Système de Retraite",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANNEES = list(range(2026, 2036))
C_S1 = "#2563EB"   # bleu saphir   — scénario actuel
C_S2 = "#DC2626"   # rouge         — scénario réforme
C_BAD = "#EF4444"  # rouge clair   — déficit (zone négative des jauges)
C_NEU = "#94a3b8"  # gris neutre   — zone positive des jauges

# ─── Sélecteur de thème (rendu dans la sidebar, lu ici en premier) ───────────
with st.sidebar:
    theme = st.radio("🎨 Thème", ["☀️ Clair", "🌙 Sombre"], horizontal=True, index=0)
LIGHT = theme.endswith("Clair")

if LIGHT:
    PAL = dict(app="#F6F8FC", panel="#FFFFFF", panel2="#FFFFFF", sidebar="#FFFFFF",
               text="#1F2937", muted="#6B7280", border="#E3E8F0", grid="#E6EBF3",
               tmpl="plotly_white", card="linear-gradient(180deg,#FFFFFF,#F7FAFE)",
               v_ok="#E7F8F0", v_bad="#FDECEC", anno="rgba(255,255,255,.9)")
else:
    PAL = dict(app="#0B0F17", panel="#141A26", panel2="#121724", sidebar="#0c111b",
               text="#E6EAF1", muted="#9aa4b2", border="#243049", grid="#1e2636",
               tmpl="plotly_dark", card="linear-gradient(180deg,#161d2b,#121724)",
               v_ok="#0a2a22", v_bad="#2b0a0a", anno="rgba(11,15,23,.85)")
TMPL = PAL["tmpl"]
GRID = PAL["grid"]
TXT = PAL["text"]

# ─────────────────────────────────────────────────────────────────────────────
# STYLE (CSS)
# ─────────────────────────────────────────────────────────────────────────────
_shadow = "0 8px 26px rgba(15,23,42,.10)" if LIGHT else "0 8px 28px rgba(0,0,0,.35)"
_tab_active = "#1d4ed8" if LIGHT else "#fff"
_kpi_label = "#888888" if LIGHT else "#94a3b8"   # label KPI atténué (contraste hiérarchique)
st.markdown(
    f"""
    <style>
      .stApp {{ background: {PAL['app']}; }}
      .main > div {{ padding-top: 0.8rem; }}
      .block-container {{ max-width: 1320px; padding-top: 1rem; }}
      h1, h2, h3, p, span, label {{ color: {PAL['text']}; }}
      h1, h2, h3 {{ font-family: 'Segoe UI', 'Inter', sans-serif; letter-spacing:.2px; }}

      /* ---- En-tête (compact) ---- */
      .hero {{
        background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 48%, #b91c1c 100%);
        padding: 13px 28px; border-radius: 16px; margin-bottom: 12px;
        box-shadow: {_shadow};
      }}
      /* Texte du héros forcé en blanc (priorité maximale sur les règles globales) */
      .block-container [data-testid="stMarkdownContainer"] .hero h1,
      .block-container [data-testid="stMarkdownContainer"] .hero p,
      .block-container [data-testid="stMarkdownContainer"] .hero .badge {{ color:#ffffff !important; }}
      .hero h1 {{ margin: 0; font-size: 1.62rem; font-weight: 800; }}
      .hero p  {{ margin: 4px 0 0 0; opacity: .96; font-size: .92rem; }}
      .badge {{
        display:inline-block; background: rgba(255,255,255,.20); padding: 3px 12px;
        border-radius: 999px; font-size: .76rem; margin-right: 7px; margin-top: 9px;
        color:#fff !important; border:1px solid rgba(255,255,255,.22);
      }}

      /* ---- KPI cards (contraste typographique renforcé) ---- */
      div[data-testid="stMetric"] {{
        background: {PAL['card']};
        border: 1px solid {PAL['border']}; border-radius: 16px;
        padding: 16px 18px 12px 18px; box-shadow: {_shadow};
      }}
      div[data-testid="stMetricValue"] {{
        font-size: 1.8em; font-weight: 800; color:{PAL['text']}; letter-spacing:-.4px; line-height:1.15;
      }}
      div[data-testid="stMetricLabel"] p {{
        font-size: .76rem; color:{_kpi_label} !important; font-weight:600;
        text-transform:uppercase; letter-spacing:.5px;
      }}

      /* ---- Onglets ---- */
      button[data-baseweb="tab"] {{ font-size: 1.0rem; font-weight: 600; padding: 6px 4px; }}
      div[data-baseweb="tab-highlight"] {{ background-color: #2563EB !important; height:3px; }}
      button[data-baseweb="tab"][aria-selected="true"] {{ color:{_tab_active} !important; }}

      /* ---- Verdict ---- */
      .verdict {{
        padding: 16px 20px; border-radius: 14px; font-size: 1.0rem; line-height: 1.55;
        border-left: 6px solid; margin-top: 8px; color:{PAL['text']};
      }}
      .verdict b {{ color:{PAL['text']}; }}
      .v-ok   {{ background:{PAL['v_ok']}; border-color:#10b981; }}
      .v-bad  {{ background:{PAL['v_bad']}; border-color:#ef4444; }}

      /* ---- Sidebar : compacter les espaces ---- */
      section[data-testid="stSidebar"] {{ background:{PAL['sidebar']}; border-right:1px solid {PAL['border']}; }}
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .55rem; }}
      section[data-testid="stSidebar"] hr {{ margin: .55rem 0; border-color:{PAL['border']}; }}
      section[data-testid="stSidebar"] .stSlider {{ padding-bottom: .2rem; }}

      /* ---- Contraste du texte adapté au thème (corrige le basculement) ---- */
      /* Sidebar : tout le texte suit le thème */
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
      section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
      section[data-testid="stSidebar"] label {{ color: {PAL['text']} !important; }}
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{ color: {PAL['muted']} !important; }}
      /* Labels de widgets (zone principale) */
      [data-testid="stWidgetLabel"] :is(p, label, div) {{ color: {PAL['text']} !important; }}
      /* Titres et paragraphes de la zone principale (héros protégé par style inline) */
      .block-container [data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4, p, li) {{ color: {PAL['text']} !important; }}
      .block-container [data-testid="stCaptionContainer"] * {{ color: {PAL['muted']} !important; }}
      /* Contenu des encarts dépliables (glossaire) */
      [data-testid="stExpander"] :is(p, li, summary, span) {{ color: {PAL['text']} !important; }}

      .footer {{ text-align:center; color:{PAL['muted']} !important; font-size:.8rem; margin-top:26px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def fr(x, dec=0, signed=False):
    """Formate un nombre à la française : espace pour les milliers, virgule décimale."""
    fmt = f"{{:+,.{dec}f}}" if signed else f"{{:,.{dec}f}}"
    return fmt.format(x).replace(",", " ").replace(".", ",").replace("-", "−")

# ─────────────────────────────────────────────────────────────────────────────
# MOTEUR : exécution mise en cache
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def lancer_simulations(ix, iy, iz, n_rep, reserve_init_mdh, incr,
                       age_max, fact_emp, rec_min, rec_max, augm):
    """Lance S1 (baseline figé) et S2 (réforme paramétrable) et renvoie les résultats."""
    reserve_init = reserve_init_mdh * 1e6
    p2 = dict(age_depart_max=age_max, facteur_employeur=fact_emp,
              recrut_min=rec_min, recrut_max=rec_max, taux_augmentation=augm)
    out = {}
    for sc in (1, 2):
        sim = sr.Simulation(sc, ix, iy, iz, n_rep,
                            reserve_initiale=reserve_init, increment_germe=incr,
                            verbose=False, params_caisse=(p2 if sc == 2 else {}))
        sim.lancer()
        out[sc] = sim.resultats
    return out


def annee_rupture(res_agg):
    """Première année où la réserve moyenne devient négative (ou None)."""
    for _, row in res_agg.iterrows():
        if row["mean"] < 0:
            return int(row["annee"])
    return None


@st.cache_data(show_spinner=False)
def generer_pdf_bytes(annees, m1, m2, r1f, r2f, rup1, rup2, info, n_rep):
    """Construit une synthèse PDF d'une page (matplotlib)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.965, "Simulation Discrète d'un Système de Retraite — Synthèse",
             ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945, "Synthèse de simulation · 2025/2026", ha="center",
             fontsize=10, color="#555")

    ax_t = fig.add_axes([0.08, 0.85, 0.84, 0.07]); ax_t.axis("off")
    ax_t.text(0, 1, f"Période 2026–2035   ·   {n_rep} réplications Monte-Carlo   ·   "
              f"générateur alea (Wichmann–Hill)\n{info}", va="top", fontsize=9.5)

    ax = fig.add_axes([0.11, 0.46, 0.80, 0.34])
    ax.plot(annees, m1, "-o", color="#2563EB", lw=2.2, ms=4, label="S1 — Actuel")
    ax.plot(annees, m2, "-o", color="#DC2626", lw=2.2, ms=4, label="S2 — Réforme")
    ax.axhline(0, ls="--", color="gray", lw=1)
    for ry, col in [(rup1, "#2563EB"), (rup2, "#DC2626")]:
        if ry:
            ax.axvline(ry, ls=":", color=col, alpha=.6)
    ax.set_xlabel("Année"); ax.set_ylabel("Réserve (Mdh)")
    ax.set_title("Évolution de la réserve moyenne", fontsize=12, fontweight="bold")
    ax.grid(alpha=.3); ax.legend()

    ax_k = fig.add_axes([0.08, 0.07, 0.84, 0.34]); ax_k.axis("off")
    verdict_s1 = f"déficit dès {rup1}" if rup1 else "réserve positive"
    verdict_s2 = f"déficit dès {rup2}" if rup2 else "réserve positive sur toute la période"
    lignes = [
        ("Réserve S1 (2035)", f"{r1f:,.0f} Mdh".replace(",", " "), verdict_s1),
        ("Réserve S2 (2035)", f"{r2f:,.0f} Mdh".replace(",", " "), verdict_s2),
        ("Écart Réforme − Actuel", f"{r2f - r1f:+,.0f} Mdh".replace(",", " "), ""),
    ]
    y = 0.98
    ax_k.text(0, y, "Indicateurs clés", fontsize=12, fontweight="bold"); y -= 0.13
    for nom, val, comm in lignes:
        ax_k.text(0.0, y, nom, fontsize=10.5)
        ax_k.text(0.40, y, val, fontsize=10.5, fontweight="bold")
        ax_k.text(0.68, y, comm, fontsize=9.5, color="#555")
        y -= 0.10
    y -= 0.04
    concl = ("Conclusion : le scénario actuel conduit à l'épuisement de la réserve, "
             "tandis que la réforme paramétrique (double cotisation employé–employeur, "
             "report volontaire du départ, recrutements accrus) maintient la caisse "
             "soutenable sur la décennie.")
    ax_k.text(0, y, concl, fontsize=10, wrap=True, va="top")

    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def to_df(resultats, scenario):
    rows = []
    for rep, annees in enumerate(resultats):
        for d in annees:
            r = dict(d)
            r["rep"] = rep + 1
            r["scenario"] = scenario
            rows.append(r)
    return pd.DataFrame(rows)


def agg(df, col):
    """Moyenne, écart-type et demi-IC 95% par année pour une colonne."""
    g = df.groupby("annee")[col]
    n = df["rep"].nunique()
    out = g.agg(["mean", "std"]).reset_index()
    out["demi_ic"] = 1.96 * out["std"] / math.sqrt(n)
    return out


def mdh(x):
    return x / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# DÉFINITIONS DES INDICATEURS (infobulles)
# ─────────────────────────────────────────────────────────────────────────────
DEF = {
    "TotEmp":  "Nombre total d'employés **actifs** (cotisants) dans la caisse à la fin de l'année.",
    "TotRet":  "Nombre total de **retraités** percevant une pension.",
    "TotCotis":"Somme des **cotisations** annuelles encaissées par la caisse. En S2, elles incluent la part employeur (double cotisation).",
    "TotPens": "Somme des **pensions** annuelles versées aux retraités. Pension = (NAT × 2 / 100) × DSAR.",
    "Reserve": "**Trésorerie cumulée** de la caisse : réserve précédente + cotisations − pensions. Une valeur **négative** signale un déficit (caisse insoutenable).",
    "NouvRet": "Nombre de **nouveaux départs** en retraite pendant l'année.",
    "NouvRec": "Nombre de **nouveaux recrutés** pendant l'année (250–400 en S1, 300–600 en S2).",
    "Plus63":  "Employés **actifs au-delà de 63 ans** ayant choisi de prolonger (réforme S2 uniquement).",
    "Plus63H": "Parmi les +63 ans, les **hommes**. Ils prolongent plus souvent (probabilités plus élevées dans le modèle).",
    "Plus63F": "Parmi les +63 ans, les **femmes**.",
    "Ratio":   "Nombre de **retraités par actif**. Plus il est élevé, plus la charge des pensions pèse sur les cotisants.",
    "Ecart":   "Différence de réserve entre la **réforme (S2)** et le **scénario actuel (S1)** : le gain net apporté par la réforme.",
    "IC":      "Intervalle de confiance à 95 % : on est sûr à 95 % que la vraie moyenne s'y trouve. Calculé par X̄ ± 1,96·σ/√n.",
}


def glossaire(cols):
    """Affiche un encart dépliable expliquant une liste d'indicateurs."""
    with st.expander("ℹ️  Que signifient ces indicateurs ?"):
        for c in cols:
            st.markdown(f"- **{c}** — {DEF[c]}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres de simulation")
    st.caption("Ajustez les paramètres puis lancez la simulation.")

    n_rep = st.slider("Nombre de réplications (Monte-Carlo)", 5, 100, 40, step=5)
    reserve_init = st.slider("Réserve initiale (Mdh)", 0, 1000, 200, step=10)

    with st.expander("🔧 Réforme sur mesure (S2)"):
        st.caption("Ces leviers ne modifient que le **scénario réforme**. Le scénario actuel (S1) reste le baseline figé.")
        age_max = st.slider("Âge de départ maximal", 63, 72, 70,
                            help="Au-delà de cet âge, le départ en retraite est obligatoire.")
        augm_pct = st.slider("Augmentation salariale (%)", 0, 20, 10,
                             help="Hausse appliquée en 2028, 2030, 2032, 2034.")
        fact_emp = st.slider("Part employeur (× la part employé)", 0.0, 2.0, 1.0, step=0.1,
                             help="1,0 = l'employeur cotise autant que l'employé (double cotisation).")
        rec_min, rec_max = st.slider("Recrutements / an", 100, 900, (300, 600), step=50,
                                     help="Fourchette du nombre de nouveaux recrutés chaque année.")

    lancer = st.button("🚀 Lancer la simulation", type="primary", width='stretch')

    with st.expander("⚙️ Paramètres avancés (Générateur aléatoire)", expanded=False):
        st.caption("Germes du générateur `alea` (Wichmann–Hill) — par défaut : (400, 400, 400).")
        c1, c2, c3 = st.columns(3)
        ix = c1.number_input("IX₀", 1, 30000, 400, step=1)
        iy = c2.number_input("IY₀", 1, 30000, 400, step=1)
        iz = c3.number_input("IZ₀", 1, 30000, 400, step=1)
        incr = st.slider("Incrément des germes / réplication", 1, 50, 5)

    st.markdown(
        "<div style='opacity:.6;font-size:.8rem;margin-top:10px'>"
        "Période : 2026 → 2035<br>"
        "Effectif initial : 10 000 actifs · 3 000 retraités<br>"
        "Pension : PR = (NAT × 2 / 100) × DSAR</div>",
        unsafe_allow_html=True,
    )

# Recalcul automatique dès qu'un paramètre change (résultats mis en cache)
custom_s2 = (age_max != 70 or augm_pct != 10 or fact_emp != 1.0
             or rec_min != 300 or rec_max != 600)
sig = (ix, iy, iz, n_rep, reserve_init, incr, age_max, fact_emp, rec_min, rec_max, augm_pct)
if lancer or st.session_state.get("sig") != sig:
    prog = st.sidebar.progress(0.0, text="Simulation en cours…")
    raw = lancer_simulations(ix, iy, iz, n_rep, reserve_init, incr,
                             age_max, fact_emp, rec_min, rec_max, augm_pct / 100.0)
    prog.progress(1.0, text="Terminé ✓")
    st.session_state["data"] = raw
    st.session_state["sig"] = sig
    st.session_state["params"] = dict(ix=ix, iy=iy, iz=iz, n_rep=n_rep,
                                       reserve_init=reserve_init, incr=incr,
                                       age_max=age_max, augm_pct=augm_pct, fact_emp=fact_emp,
                                       rec_min=rec_min, rec_max=rec_max, custom_s2=custom_s2)

raw = st.session_state["data"]
df1 = to_df(raw[1], 1)
df2 = to_df(raw[2], 2)

# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>🏛️ Simulation Discrète d'un Système de Retraite</h1>
      <p>Étude d'impact d'une réforme paramétrique — Approche Monte-Carlo</p>
      <span class="badge">Scénario 1 · Actuel (retraite 63 ans)</span>
      <span class="badge">Scénario 2 · Réforme (report 63→70 ans)</span>
      <span class="badge">Générateur alea (Wichmann–Hill)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Valeurs clés
res1 = agg(df1, "Reserve")
res2 = agg(df2, "Reserve")
r1_2035 = mdh(res1.loc[res1.annee == 2035, "mean"].iloc[0])
r2_2035 = mdh(res2.loc[res2.annee == 2035, "mean"].iloc[0])
r1_2026 = mdh(res1.loc[res1.annee == 2026, "mean"].iloc[0])
r2_2026 = mdh(res2.loc[res2.annee == 2026, "mean"].iloc[0])

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🏠  Vue d'ensemble", "📈  Évolution détaillée", "⚖️  Comparaison S1 vs S2",
     "👥  Seniors (>63 ans)", "📋  Données & export", "📚  Méthodologie"]
)

def jauge(val, titre, color, gmin, gmax):
    """Jauge de soutenabilité : arc à 3 zones (déficit / fragile / confortable)."""
    # Fond du demi-cercle : 🔴 déficit (<0) · 🟠 fragile (0–1000) · 🟢 confortable (>1000)
    steps = [dict(range=[gmin, 0], color="rgba(239,68,68,.30)")]
    if gmax > 1000:
        steps += [dict(range=[0, 1000], color="rgba(245,158,11,.28)"),
                  dict(range=[1000, gmax], color="rgba(16,185,129,.28)")]
    else:
        steps += [dict(range=[0, max(gmax, 1)], color="rgba(16,185,129,.28)")]
    g = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number=dict(suffix=" Mdh", font=dict(size=24, color=color)),
        title=dict(text=titre, font=dict(size=14, color=TXT)),
        gauge=dict(axis=dict(range=[gmin, gmax], tickformat=",.0f"),
                   bar=dict(color=color, thickness=0.32), bgcolor="rgba(0,0,0,0)",
                   borderwidth=0, steps=steps,
                   threshold=dict(line=dict(color="#94a3b8", width=3), thickness=0.92, value=0))))
    g.update_layout(height=250, margin=dict(l=24, r=24, t=52, b=8), template=TMPL,
                    separators=", ", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TXT))
    return g


rup1 = annee_rupture(res1)
rup2 = annee_rupture(res2)

# ── TAB 1 : VUE D'ENSEMBLE ───────────────────────────────────────────────────
with tab1:
    if st.session_state["params"].get("custom_s2"):
        p = st.session_state["params"]
        st.info(f"🔧 **Réforme sur mesure active (S2)** — âge max **{p['age_max']} ans** · "
                f"augmentation **{p['augm_pct']} %** · part employeur **×{p['fact_emp']:.1f}** · "
                f"recrutement **{p['rec_min']}–{p['rec_max']}/an**. (S1 reste le baseline figé.)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Réserve S1 — 2035", f"{fr(r1_2035)} Mdh", f"{fr(r1_2035 - r1_2026, signed=True)} vs 2026",
              help=DEF["Reserve"])
    k2.metric("Réserve S2 — 2035", f"{fr(r2_2035)} Mdh", f"{fr(r2_2035 - r2_2026, signed=True)} vs 2026",
              help=DEF["Reserve"])
    k3.metric("Écart Réforme − Actuel (2035)", f"{fr(r2_2035 - r1_2035, signed=True)} Mdh",
              help=DEF["Ecart"])
    ratio1 = df1[df1.annee == 2035]["TotRet"].mean() / df1[df1.annee == 2035]["TotEmp"].mean()
    k4.metric("Ratio retraités/actifs S1 (2035)", fr(ratio1, dec=2), help=DEF["Ratio"])

    st.markdown("#### Évolution de la réserve — moyenne des réplications (± écart-type)")
    fig = go.Figure()
    for res, c, lab in [(res1, C_S1, "Scénario 1 — Actuel"),
                        (res2, C_S2, "Scénario 2 — Réforme")]:
        y = mdh(res["mean"].values)
        s = mdh(res["std"].values)
        fig.add_trace(go.Scatter(
            x=ANNEES + ANNEES[::-1],
            y=list(y + s) + list((y - s)[::-1]),
            fill="toself", fillcolor=c, opacity=0.10,
            line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=ANNEES, y=y, mode="lines+markers", name=lab,
            line=dict(color=c, width=3.5), marker=dict(size=7),
            hovertemplate="%{x} : <b>%{y:,.0f} Mdh</b><extra>" + lab + "</extra>"))
        # Étiquette de valeur en bout de courbe
        fig.add_annotation(x=ANNEES[-1], y=y[-1], text=f"<b>{fr(y[-1])}</b>",
                           showarrow=False, xshift=42, font=dict(color=c, size=14),
                           bgcolor=PAL["anno"], bordercolor=c, borderwidth=1, borderpad=3)
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                  annotation_text="Réserve = 0", annotation_position="bottom right",
                  annotation_font_color="#94a3b8")
    for ry, c, lab in [(rup1, C_S1, "S1"), (rup2, C_S2, "S2")]:
        if ry:
            fig.add_vline(x=ry, line_dash="dash", line_color=c, opacity=0.55)
            fig.add_annotation(x=ry, yref="paper", y=1.0, yshift=8, showarrow=False,
                               text=f"⚠️ déficit {lab} dès {ry}", font=dict(color=c, size=12))
    fig.update_layout(
        template=TMPL, height=470, hovermode="x unified",
        separators=", ", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Année", dtick=1, showgrid=False),
        yaxis=dict(title="Réserve (Mdh)", tickformat=",.0f", gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=13)),
        margin=dict(l=10, r=70, t=40, b=10))
    st.plotly_chart(fig, width='stretch')

    # Jauges de soutenabilité
    st.markdown("#### Soutenabilité de la caisse en 2035")
    gmin = min(r1_2035, r2_2035, 0.0)
    gmax = max(r1_2035, r2_2035, 0.0)
    pad = (gmax - gmin) * 0.18 + 1
    gmin, gmax = gmin - pad, gmax + pad
    jc1, jc2 = st.columns(2)
    jc1.plotly_chart(jauge(r1_2035, "Scénario 1 — Actuel", C_S1, gmin, gmax), width='stretch')
    jc2.plotly_chart(jauge(r2_2035, "Scénario 2 — Réforme", C_S2, gmin, gmax), width='stretch')
    st.caption("🔴 déficit (réserve < 0) · 🟠 fragile (0–1 000 Mdh) · 🟢 confortable (> 1 000 Mdh). "
               "Le trait gris marque le seuil critique de 0.")

    # Verdict automatique
    rup1_txt = f"déficit dès <b>{rup1}</b>" if rup1 else "réserve positive maintenue"
    st.markdown(
        f"""<div class="verdict v-bad">
        <b>⚠️ Scénario actuel (S1).</b> Réserve en 2035 : <b>{fr(r1_2035)} Mdh</b> — {rup1_txt}.
        Le départ figé à 63 ans et la cotisation employé seule ne suffisent pas à couvrir les pensions.</div>""",
        unsafe_allow_html=True)
    cls = "v-ok" if r2_2035 >= 0 else "v-bad"
    icone = "✅ La réforme redresse la caisse." if r2_2035 >= 0 else "⚠️ Même réformée, la caisse reste fragile."
    rup2_txt = f"déficit dès <b>{rup2}</b>" if rup2 else "réserve positive sur toute la période"
    st.markdown(
        f"""<div class="verdict {cls}" style="margin-top:12px">
        <b>{icone}</b> Scénario réforme en 2035 : <b>{fr(r2_2035)} Mdh</b> — {rup2_txt}.
        Écart avec l'actuel : <b>{fr(r2_2035 - r1_2035, signed=True)} Mdh</b>
        (double cotisation employé–employeur, report volontaire du départ, recrutements accrus).</div>""",
        unsafe_allow_html=True)

# ── TAB 2 : ÉVOLUTION DÉTAILLÉE ──────────────────────────────────────────────
with tab2:
    sc_sel = st.radio("Scénario", ["Scénario 1 — Actuel", "Scénario 2 — Réforme"],
                      horizontal=True)
    sc = 1 if sc_sel.startswith("Scénario 1") else 2
    df = df1 if sc == 1 else df2
    couleur = C_S1 if sc == 1 else C_S2

    glossaire(["TotEmp", "TotRet", "TotCotis", "TotPens", "Reserve", "NouvRet"])

    indic = {
        "TotEmp": "Employés actifs", "TotRet": "Retraités",
        "TotCotis": "Cotisations (Mdh)", "TotPens": "Pensions (Mdh)",
        "Reserve": "Réserve (Mdh)", "NouvRet": "Nouveaux retraités",
        "NouvRec": "Nouveaux recrutés",
    }
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[indic[k] for k in
                        ["TotEmp", "TotRet", "TotCotis", "TotPens", "Reserve", "NouvRet"]])
    cells = [("TotEmp", 1, 1), ("TotRet", 1, 2), ("TotCotis", 1, 3),
             ("TotPens", 2, 1), ("Reserve", 2, 2), ("NouvRet", 2, 3)]
    for col, r, c in cells:
        a = agg(df, col)
        y = a["mean"].values
        s = a["std"].values
        div = 1e6 if col in ("TotCotis", "TotPens", "Reserve") else 1
        y, s = y / div, s / div
        fig.add_trace(go.Scatter(
            x=ANNEES + ANNEES[::-1], y=list(y + s) + list((y - s)[::-1]),
            fill="toself", fillcolor=couleur, opacity=0.12, line=dict(width=0),
            hoverinfo="skip", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=ANNEES, y=y, mode="lines+markers",
            line=dict(color=couleur, width=2.5), marker=dict(size=5),
            showlegend=False), row=r, col=c)
    fig.update_layout(template=TMPL, height=600, separators=", ",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10),
                      title=f"{sc_sel} — moyennes des {st.session_state['params']['n_rep']} réplications (± σ)")
    fig.update_xaxes(showgrid=False, dtick=2)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Trajectoires individuelles de la réserve")
    figt = go.Figure()
    for rep in df["rep"].unique():
        d = df[df.rep == rep].sort_values("annee")
        figt.add_trace(go.Scatter(
            x=d["annee"], y=mdh(d["Reserve"]), mode="lines",
            line=dict(color=couleur, width=0.7), opacity=0.18,
            hoverinfo="skip", showlegend=False))
    a = agg(df, "Reserve")
    figt.add_trace(go.Scatter(x=ANNEES, y=mdh(a["mean"]), mode="lines+markers",
                   name="Moyenne", line=dict(color=couleur, width=4)))
    figt.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1.5)
    figt.update_layout(template=TMPL, height=420, hovermode="x unified",
                       separators=", ", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis=dict(title="Année", dtick=1, showgrid=False),
                       yaxis=dict(title="Réserve (Mdh)", tickformat=",.0f", gridcolor=GRID, zeroline=False),
                       margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figt, width='stretch')

# ── TAB 3 : COMPARAISON ──────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Indicateurs clés — Scénario 1 vs Scénario 2")
    glossaire(["TotEmp", "TotRet", "TotCotis", "TotPens", "NouvRet", "NouvRec"])
    indc = {"TotEmp": "Employés actifs", "TotRet": "Retraités",
            "TotCotis": "Cotisations (Mdh)", "TotPens": "Pensions (Mdh)",
            "NouvRet": "Nouveaux retraités", "NouvRec": "Nouveaux recrutés"}
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(indc.values()))
    cells = [("TotEmp", 1, 1), ("TotRet", 1, 2), ("TotCotis", 1, 3),
             ("TotPens", 2, 1), ("NouvRet", 2, 2), ("NouvRec", 2, 3)]
    for col, r, c in cells:
        div = 1e6 if col in ("TotCotis", "TotPens") else 1
        for d, cc, lab in [(df1, C_S1, "S1"), (df2, C_S2, "S2")]:
            a = agg(d, col)
            fig.add_trace(go.Scatter(
                x=ANNEES, y=a["mean"].values / div, mode="lines+markers",
                line=dict(color=cc, width=2.3), marker=dict(size=4),
                name=lab, showlegend=(col == "TotEmp"),
                legendgroup=lab), row=r, col=c)
    fig.update_layout(template=TMPL, height=600, separators=", ",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=13)))
    fig.update_xaxes(showgrid=False, dtick=2)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Réserve moyenne & intervalles de confiance à 95 %")
    st.caption("ℹ️ " + DEF["IC"])
    annees_ic = [2026, 2030, 2035]
    figic = go.Figure()
    for d, cc, lab in [(df1, C_S1, "S1 — Actuel"), (df2, C_S2, "S2 — Réforme")]:
        a = agg(d, "Reserve")
        a = a[a.annee.isin(annees_ic)]
        figic.add_trace(go.Bar(
            x=[str(y) for y in annees_ic], y=mdh(a["mean"]),
            name=lab, marker_color=cc, opacity=0.85,
            error_y=dict(type="data", array=mdh(a["demi_ic"]), color="#e5e7eb",
                         thickness=1.6, width=8)))
    figic.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1.5)
    figic.update_layout(template=TMPL, height=420, barmode="group", separators=", ",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Année", showgrid=False),
                        yaxis=dict(title="Réserve (Mdh)", tickformat=",.0f", gridcolor=GRID, zeroline=False),
                        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=13)),
                        margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figic, width='stretch')

    # Tableau IC
    rows = []
    for y in annees_ic:
        for d, lab in [(df1, "Actuel"), (df2, "Réforme")]:
            a = agg(d, "Reserve")
            row = a[a.annee == y].iloc[0]
            m, hi = mdh(row["mean"]), mdh(row["demi_ic"])
            rows.append({"Année": y, "Scénario": lab, "Moyenne (Mdh)": round(m, 2),
                         "IC inf": round(m - hi, 2), "IC sup": round(m + hi, 2),
                         "Demi-IC": round(hi, 2)})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ── TAB 4 : SENIORS >63 ANS ──────────────────────────────────────────────────
with tab4:
    st.markdown("#### Scénario 2 — Employés actifs au-delà de 63 ans")
    st.caption("Effet du report volontaire du départ à la retraite (réforme).")
    fig = go.Figure()
    for col, lab, cc in [("Plus63", "Total", "#7C3AED"),
                         ("Plus63H", "Hommes", C_S1),
                         ("Plus63F", "Femmes", C_S2)]:
        a = agg(df2, col)
        fig.add_trace(go.Scatter(x=ANNEES, y=a["mean"], mode="lines+markers",
                      name=lab, line=dict(color=cc, width=3), marker=dict(size=6)))
    fig.update_layout(template=TMPL, height=470, hovermode="x unified",
                      separators=", ", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="Année", dtick=1, showgrid=False),
                      yaxis=dict(title="Nombre d'employés", gridcolor=GRID, zeroline=False),
                      legend=dict(orientation="h", y=1.05, x=0, font=dict(size=13)),
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')

    p2035 = df2[df2.annee == 2035]
    c1, c2, c3 = st.columns(3)
    c1.metric("Seniors >63 ans (2035)", fr(p2035['Plus63'].mean()), help=DEF["Plus63"])
    c2.metric("dont Hommes", fr(p2035['Plus63H'].mean()), help=DEF["Plus63H"])
    c3.metric("dont Femmes", fr(p2035['Plus63F'].mean()), help=DEF["Plus63F"])
    st.caption("Les hommes prolongent ~2 à 3× plus que les femmes, conformément aux "
               "probabilités de prolongation définies dans le modèle.")

# ── TAB 5 : DONNÉES & EXPORT ─────────────────────────────────────────────────
with tab5:
    st.markdown("#### Tableau récapitulatif — moyennes annuelles")
    recap = pd.DataFrame({"Année": ANNEES})
    for col, lab in [("TotEmp", "Actifs"), ("TotRet", "Retraités")]:
        recap[f"{lab} S1"] = agg(df1, col)["mean"].round(0).astype(int).values
        recap[f"{lab} S2"] = agg(df2, col)["mean"].round(0).astype(int).values
    recap["Réserve S1 (Mdh)"] = mdh(agg(df1, "Reserve")["mean"]).round(1).values
    recap["Réserve S2 (Mdh)"] = mdh(agg(df2, "Reserve")["mean"]).round(1).values
    recap["Écart (Mdh)"] = (recap["Réserve S2 (Mdh)"] - recap["Réserve S1 (Mdh)"]).round(1)
    st.dataframe(recap, width='stretch', hide_index=True)

    st.markdown("#### Détail par réplication")
    sc_d = st.selectbox("Scénario", ["Scénario 1", "Scénario 2"])
    an_d = st.selectbox("Année", ANNEES, index=len(ANNEES) - 1)
    dd = (df1 if sc_d == "Scénario 1" else df2)
    dd = dd[dd.annee == an_d].copy()
    for c in ["TotCotis", "TotPens", "Reserve"]:
        dd[c] = (dd[c] / 1e6).round(2)
    cols = ["rep", "TotEmp", "TotRet", "TotCotis", "TotPens", "Reserve", "NouvRet", "NouvRec"]
    if sc_d == "Scénario 2":
        cols = cols[:2] + ["Plus63", "Plus63H", "Plus63F"] + cols[2:]
    st.dataframe(dd[cols].rename(columns={
        "rep": "Sim", "TotCotis": "Cotis(Mdh)", "TotPens": "Pens(Mdh)",
        "Reserve": "Réserve(Mdh)"}), width='stretch', hide_index=True, height=380)

    # Téléchargements
    full = pd.concat([df1, df2], ignore_index=True)
    st.download_button("⬇️ Télécharger toutes les données (CSV)",
                       full.to_csv(index=False).encode("utf-8"),
                       "simulation_retraite_resultats.csv", "text/csv",
                       width='stretch')
    st.download_button("⬇️ Télécharger le récapitulatif (CSV)",
                       recap.to_csv(index=False).encode("utf-8"),
                       "simulation_recapitulatif.csv", "text/csv",
                       width='stretch')

    st.markdown("#### Synthèse PDF (1 page)")
    st.caption("KPIs, graphe de la réserve et conclusion — prêt à imprimer ou à projeter.")
    p = st.session_state["params"]
    info = (f"Réforme sur mesure : âge max {p['age_max']} ans · augmentation {p['augm_pct']} % · "
            f"part employeur ×{p['fact_emp']:.1f} · recrutement {p['rec_min']}–{p['rec_max']}/an"
            if p.get("custom_s2") else
            "Scénarios de référence (réforme standard : report 63→70 ans, double cotisation).")
    pdf_bytes = generer_pdf_bytes(
        ANNEES,
        [round(v, 1) for v in mdh(agg(df1, "Reserve")["mean"]).tolist()],
        [round(v, 1) for v in mdh(agg(df2, "Reserve")["mean"]).tolist()],
        round(r1_2035, 1), round(r2_2035, 1), rup1, rup2, info,
        st.session_state["params"]["n_rep"])
    st.download_button("📄 Télécharger la synthèse PDF", pdf_bytes,
                       "synthese_simulation_retraite.pdf", "application/pdf",
                       type="primary", width='stretch')

# ── TAB 6 : MÉTHODOLOGIE ─────────────────────────────────────────────────────
with tab6:
    st.markdown("#### 🎯 Démarche de simulation discrète (8 étapes)")
    cA, cB = st.columns(2)
    cA.markdown(
        "1. **Formulation du problème** — viabilité d'une caisse de retraite.\n"
        "2. **Définition du système** — actifs, retraités, réserve.\n"
        "3. **Construction du modèle** — cycle annuel + modèle mathématique.\n"
        "4. **Programmation** — Python orienté objet (`alea`, `Employe`, `CaisseRetraite`, `Simulation`).")
    cB.markdown(
        "5. **Validation** — reproductibilité par germes, contrôles manuels.\n"
        "6. **Vérification** — cohérence cotisations / pensions / réserve.\n"
        "7. **Expérimentation** — 40 réplications Monte-Carlo par scénario.\n"
        "8. **Interprétation** — comparaison S1 vs S2, intervalles de confiance.")

    st.divider()
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown("#### 🧮 Modèle mathématique")
        st.markdown("**Pension de retraite (mensuelle)**")
        st.latex(r"PR = \frac{NAT \times 2}{100} \times DSAR")
        st.caption("NAT = nombre d'années travaillées · DSAR = dernier salaire de référence.")
        st.markdown("**Mise à jour de la réserve**")
        st.latex(r"R_{t+1} = R_t + \text{TotCotis}_t - \text{TotPens}_t")
        st.markdown("**Intervalle de confiance à 95 %**")
        st.latex(r"IC_{95\%} = \left[\, \bar{X} - 1{,}96\,\tfrac{S}{\sqrt{n}} \;;\; \bar{X} + 1{,}96\,\tfrac{S}{\sqrt{n}} \,\right]")
    with c2:
        st.markdown("#### 🎲 Générateur `alea` (Wichmann–Hill)")
        st.code(
            "self.ix = 171*(ix % 177) - 2 *(ix // 177)\n"
            "self.iy = 172*(iy % 176) - 35*(iy // 176)\n"
            "self.iz = 170*(iz % 178) - 63*(iz // 178)\n"
            "# recadrage si négatif (+30269 / +30307 / +30323)\n"
            "u = ix/30269 + iy/30307 + iz/30323\n"
            "return u - floor(u)   # ∈ [0, 1[", language="python")
        st.caption("Germes par défaut : (400, 400, 400), incrémentés de 5 à chaque réplication.")

    st.divider()
    st.markdown("#### 📊 Distributions du modèle")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Salaire actuel des 10 000 actifs**")
        tr = sr.TRANCHES_SAL_ACTUEL
        st.dataframe(pd.DataFrame({
            "Tranche (dh)": [f"{a:,}–{b:,}".replace(",", " ") for a, b in tr],
            "Fréquence": [f"{f:.0%}" for f in sr.FREQS_SAL_ACTUEL]}),
            hide_index=True, width='stretch')
        st.markdown("**Cotisation — Scénario 1 (employé seul)**")
        st.dataframe(pd.DataFrame({
            "Tranche (dh)": ["< 5 000", "5 000–7 000", "7 000–10 000", "≥ 10 000"],
            "Taux": ["5 %", "6 %", "8 %", "10 %"]}), hide_index=True, width='stretch')
    with d2:
        st.markdown("**Âge actuel des actifs**")
        st.dataframe(pd.DataFrame({
            "Tranche": [f"{a}–{b} ans" for a, b in sr.TRANCHES_AGE_ACTUEL],
            "Fréquence": [f"{f:.0%}" for f in sr.FREQS_AGE_ACTUEL]}),
            hide_index=True, width='stretch')
        st.markdown("**Probabilité de prolongation — Scénario 2** *(k = âge − 63)*")
        st.dataframe(pd.DataFrame({
            "Salaire": ["≥ 30 000", "≥ 10 000", "≥ 5 000", "< 5 000"],
            "Hommes": ["70 − 5k %", "50 − 4k %", "30 − 2k %", "10 − 1k %"],
            "Femmes": ["50 − 4k %", "30 − 2k %", "15 − 1k %", "5 − 1k %"]}),
            hide_index=True, width='stretch')

    st.divider()
    st.markdown("#### ⚖️ Les deux scénarios")
    st.dataframe(pd.DataFrame({
        "Paramètre": ["Âge de départ", "Cotisation", "Augmentations salariales",
                      "Recrutements / an", "Cotisation > 10 000 dh"],
        "S1 — Actuel": ["63 ans (fixe)", "Employé seul (5–10 %)", "+5 % (2026, 2030, 2034)",
                        "250–400", "10 % (plafond)"],
        "S2 — Réforme": ["63 → 70 ans (volontaire)", "Employé + employeur",
                         "+10 % (2028, 2030, 2032, 2034)", "300–600",
                         "Progressive 10→30 % par tranche"]}),
        hide_index=True, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Simulation discrète d'un système de retraite · "
    "Approche Monte-Carlo · 2025/2026</div>",
    unsafe_allow_html=True,
)
