import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Perceptrón Interactivo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #f0ede6;
  }

  .stApp {
    background: #0d0d0d;
  }

  h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    letter-spacing: -0.5px;
  }

  /* Sliders */
  .stSlider > div > div > div > div {
    background: #f0c040 !important;
  }

  /* Metric boxes */
  [data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 12px 16px;
  }

  [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
    color: #f0c040 !important;
  }

  [data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Buttons */
  .stButton > button {
    font-family: 'Space Mono', monospace;
    background: #1a1a1a;
    color: #f0c040;
    border: 1.5px solid #f0c040;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 1px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #f0c040;
    color: #0d0d0d;
  }

  /* Selectbox / radio */
  .stRadio > div {
    gap: 8px;
  }

  /* Section header */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 4px;
  }

  .card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }

  .pill-pos {
    display: inline-block;
    background: #1a3a1a;
    color: #5fdf6f;
    border: 1px solid #5fdf6f;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
  }
  .pill-neg {
    display: inline-block;
    background: #3a1a1a;
    color: #df5f5f;
    border: 1px solid #df5f5f;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
  }
  .pill-none {
    display: inline-block;
    background: #222;
    color: #888;
    border: 1px solid #444;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
  }

  hr { border-color: #222; }

  .big-number {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0c040;
  }

  .score-bar-bg {
    background: #1a1a1a;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def perceptron_output(x1, x2, w1, w2, bias):
    """Weighted sum and classification."""
    z = w1 * x1 + w2 * x2 + bias
    return z, 1 if z > 0 else -1


def input_value(switch_on: bool) -> float:
    return 1.0 if switch_on else -1.0


def label_to_str(label: int) -> str:
    return "POSITIVA (+1)" if label == 1 else "NEGATIVA (−1)"


PATTERNS = [
    {"name": "Ambos OFF",    "s1": False, "s2": False},
    {"name": "S1 ON, S2 OFF","s1": True,  "s2": False},
    {"name": "S1 OFF, S2 ON","s1": False, "s2": True},
    {"name": "Ambos ON",     "s1": True,  "s2": True},
]

PRESETS = {
    "OR (∨)":   {"targets": [False, True, True, True],  "w1": 1.0,  "w2": 1.0,  "bias": 0.0},
    "AND (∧)":  {"targets": [False, False, False, True], "w1": 1.0,  "w2": 1.0,  "bias": -1.0},
    "NOT S1":   {"targets": [True,  False, True, False], "w1": -1.0, "w2": 0.0,  "bias": 0.0},
    "XOR (⊕)":  {"targets": [False, True,  True, False], "w1": 0.0,  "w2": 0.0,  "bias": 0.0},
    "Limpio":   {"targets": [False, False, False, False], "w1": 0.0,  "w2": 0.0,  "bias": 0.0},
}

# ─── SESSION STATE INIT ─────────────────────────────────────────────────────────
if "targets" not in st.session_state:
    st.session_state.targets = [False, False, False, False]  # False = NEG, True = POS
if "w1" not in st.session_state:
    st.session_state.w1 = 0.0
if "w2" not in st.session_state:
    st.session_state.w2 = 0.0
if "bias" not in st.session_state:
    st.session_state.bias = 0.0


def apply_preset(name):
    p = PRESETS[name]
    st.session_state.targets = list(p["targets"])
    st.session_state.w1   = p["w1"]
    st.session_state.w2   = p["w2"]
    st.session_state.bias = p["bias"]


# ─── TITLE ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 8px 0;">
  <div class="section-title">Autómatas, Gramáticas y Lenguajes · 2025</div>
  <h1 style="font-size:2.8rem; margin:0; color:#f0ede6;">
    🧠 Perceptrón <span style="color:#f0c040;">Interactivo</span>
  </h1>
  <p style="color:#666; font-family:'Space Mono',monospace; font-size:0.8rem; margin-top:8px;">
    Ajusta los pesos manualmente · Observa la frontera de decisión · Descubre los límites del perceptrón
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── PRESETS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚡ Cargar problema predefinido</div>', unsafe_allow_html=True)
cols_p = st.columns(len(PRESETS))
for i, (pname, _) in enumerate(PRESETS.items()):
    with cols_p[i]:
        if st.button(pname, key=f"preset_{pname}", use_container_width=True):
            apply_preset(pname)

st.markdown("---")

# ─── MAIN LAYOUT ────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1.1, 1.2, 1.5], gap="large")

# ══════════════════════════════════════════════════════
# LEFT: PATTERN TABLE
# ══════════════════════════════════════════════════════
with col_left:
    st.markdown("### 📋 Patrones de entrada")
    st.markdown('<div class="section-title">Define la etiqueta deseada para cada combinación</div>', unsafe_allow_html=True)

    w1   = st.session_state.w1
    w2   = st.session_state.w2
    bias = st.session_state.bias

    correct = 0
    for i, pat in enumerate(PATTERNS):
        x1v = input_value(pat["s1"])
        x2v = input_value(pat["s2"])
        z, pred = perceptron_output(x1v, x2v, w1, w2, bias)
        target_int = 1 if st.session_state.targets[i] else -1
        is_correct = pred == target_int
        if is_correct:
            correct += 1

        s1_icon = "🟢" if pat["s1"] else "🔴"
        s2_icon = "🟢" if pat["s2"] else "🔴"
        check   = "✅" if is_correct else "❌"

        with st.container():
            st.markdown(f"""
            <div class="card" style="padding:14px 18px; margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; font-size:0.9rem;">{check} {pat['name']}</span>
                <span style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#888;">
                  S1:{s1_icon} S2:{s2_icon}
                </span>
              </div>
              <div style="margin-top:6px; font-family:'Space Mono',monospace; font-size:0.75rem; color:#aaa;">
                z = {z:+.2f} → pred = <span style="color:{'#5fdf6f' if pred==1 else '#df5f5f'}">{'(+1)' if pred==1 else '(−1)'}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            new_target = st.toggle(
                f"Etiqueta positiva (+1)",
                value=st.session_state.targets[i],
                key=f"target_{i}"
            )
            st.session_state.targets[i] = new_target

    # Score
    pct = correct / 4
    bar_color = "#5fdf6f" if correct == 4 else "#f0c040" if correct >= 2 else "#df5f5f"
    st.markdown(f"""
    <div class="card" style="text-align:center; margin-top:8px;">
      <div class="section-title">Clasificados correctamente</div>
      <div class="big-number">{correct}<span style="font-size:1.2rem; color:#555;"> / 4</span></div>
      <div class="score-bar-bg">
        <div style="width:{pct*100}%; background:{bar_color}; height:10px; border-radius:8px; transition:width 0.4s;"></div>
      </div>
      {'<div style="color:#5fdf6f; font-weight:700; margin-top:8px;">¡Problema resuelto! 🎉</div>' if correct == 4 else ''}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MIDDLE: WEIGHT CONTROLS + PERCEPTRON DIAGRAM
# ══════════════════════════════════════════════════════
with col_mid:
    st.markdown("### 🎛️ Ajuste de pesos (perillas)")
    st.markdown('<div class="section-title">Mueve los sliders para configurar el perceptrón</div>', unsafe_allow_html=True)

    w1 = st.slider("Peso w₁", min_value=-3.0, max_value=3.0,
                   value=st.session_state.w1, step=0.05,
                   key="slider_w1",
                   help="Multiplica la entrada x₁")
    st.session_state.w1 = w1

    w2 = st.slider("Peso w₂", min_value=-3.0, max_value=3.0,
                   value=st.session_state.w2, step=0.05,
                   key="slider_w2",
                   help="Multiplica la entrada x₂")
    st.session_state.w2 = w2

    bias = st.slider("Bias b", min_value=-3.0, max_value=3.0,
                     value=st.session_state.bias, step=0.05,
                     key="slider_bias",
                     help="Desplazamiento independiente de las entradas")
    st.session_state.bias = bias

    st.markdown("---")

    # ── Perceptron diagram ──────────────────────────────
    st.markdown("### 🔬 Diagrama del perceptrón")

    fig_d, ax_d = plt.subplots(figsize=(5, 3.2))
    fig_d.patch.set_facecolor("#141414")
    ax_d.set_facecolor("#141414")
    ax_d.axis("off")

    node_kw = dict(ha="center", va="center",
                   fontfamily="monospace")

    # Draw nodes
    for (nx, ny, label, col) in [
        (0.08, 0.75, "x₁", "#4a9eff"),
        (0.08, 0.25, "x₂", "#4a9eff"),
        (0.08, -0.25, "b",  "#f0c040"),
        (0.65, 0.25,  "Σ",  "#888"),
        (0.92, 0.25,  "ŷ",  "#5fdf6f"),
    ]:
        circle = plt.Circle((nx, ny), 0.09, color=col, alpha=0.18, zorder=3)
        border = plt.Circle((nx, ny), 0.09, color=col, fill=False, linewidth=1.5, zorder=4)
        ax_d.add_patch(circle)
        ax_d.add_patch(border)
        ax_d.text(nx, ny, label, **node_kw, fontsize=10, color=col, zorder=5)

    # Draw arrows + weight labels
    arrow_kw = dict(arrowstyle="-|>", color="#555", lw=1.2,
                    mutation_scale=12, zorder=2)
    ax_d.annotate("", xy=(0.56, 0.30), xytext=(0.17, 0.72),
                  arrowprops=dict(**arrow_kw, color="#4a9eff"))
    ax_d.annotate("", xy=(0.56, 0.25), xytext=(0.17, 0.25),
                  arrowprops=dict(**arrow_kw, color="#4a9eff"))
    ax_d.annotate("", xy=(0.56, 0.20), xytext=(0.17, -0.18),
                  arrowprops=dict(**arrow_kw, color="#f0c040"))
    ax_d.annotate("", xy=(0.83, 0.25), xytext=(0.74, 0.25),
                  arrowprops=dict(**arrow_kw, color="#5fdf6f"))

    ax_d.text(0.33, 0.60, f"w₁={w1:+.2f}", fontsize=7.5, color="#4a9eff",
              fontfamily="monospace", ha="center")
    ax_d.text(0.36, 0.22, f"w₂={w2:+.2f}", fontsize=7.5, color="#4a9eff",
              fontfamily="monospace", ha="center")
    ax_d.text(0.35, -0.08, f"b={bias:+.2f}", fontsize=7.5, color="#f0c040",
              fontfamily="monospace", ha="center")

    ax_d.text(0.65, 0.25, "Σ", fontsize=13, ha="center", va="center",
              color="white", fontfamily="monospace", zorder=6)
    ax_d.text(0.92, 0.25, "ŷ", fontsize=10, ha="center", va="center",
              color="#5fdf6f", fontfamily="monospace", zorder=6)

    ax_d.set_xlim(-0.1, 1.1)
    ax_d.set_ylim(-0.5, 1.0)

    st.pyplot(fig_d, use_container_width=True)
    plt.close(fig_d)

    # ── Formula ─────────────────────────────────────────
    st.markdown(f"""
    <div class="card" style="text-align:center;">
      <div class="section-title">Fórmula</div>
      <div style="font-family:'Space Mono',monospace; font-size:0.82rem; color:#ccc; margin-top:6px;">
        z = ({w1:+.2f})·x₁ + ({w2:+.2f})·x₂ + ({bias:+.2f})
      </div>
      <div style="font-family:'Space Mono',monospace; font-size:0.78rem; color:#888; margin-top:4px;">
        ŷ = +1 si z > 0, else −1
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# RIGHT: DECISION BOUNDARY PLOT
# ══════════════════════════════════════════════════════
with col_right:
    st.markdown("### 📊 Frontera de decisión")
    st.markdown('<div class="section-title">Plano 2D · los 4 patrones y la línea separadora</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")

    # Grid background
    xx, yy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    Z = w1 * xx + w2 * yy + bias
    ax.contourf(xx, yy, Z, levels=[-999, 0, 999],
                colors=["#2a1515", "#152a15"], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors=["#f0c040"], linewidths=2.5)

    # Decision boundary label
    if abs(w2) > 1e-6:
        xline = np.linspace(-1.8, 1.8, 100)
        yline = (-w1 * xline - bias) / w2
        mask = (yline >= -1.8) & (yline <= 1.8)
        if mask.sum() > 1:
            mid_idx = len(xline) // 2
            ax.annotate("frontera", xy=(xline[mid_idx], yline[mid_idx]),
                        fontsize=7, color="#f0c040", fontfamily="monospace",
                        xytext=(10, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#f0c040", lw=0.7))

    # Region labels
    ax.text(1.5, 1.6, "+1", fontsize=20, color="#5fdf6f", alpha=0.3,
            ha="center", va="center", fontweight="bold")
    ax.text(-1.5, -1.6, "−1", fontsize=20, color="#df5f5f", alpha=0.3,
            ha="center", va="center", fontweight="bold")

    # Plot the 4 points
    point_coords = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    for i, (px, py) in enumerate(point_coords):
        target_int = 1 if st.session_state.targets[i] else -1
        _, pred = perceptron_output(px, py, w1, w2, bias)
        is_correct = pred == target_int

        edge_color = "#5fdf6f" if target_int == 1 else "#df5f5f"
        fill_color = "#1a3a1a" if target_int == 1 else "#3a1a1a"
        marker     = "^" if target_int == 1 else "v"

        ax.scatter(px, py, s=220, marker=marker,
                   color=fill_color, edgecolors=edge_color,
                   linewidths=2.5, zorder=5)

        # Checkmark / X
        check_sym = "✓" if is_correct else "✗"
        check_col = "#ffffff" if is_correct else "#ff6666"
        ax.text(px, py + 0.22, check_sym, fontsize=9, ha="center",
                color=check_col, fontweight="bold", zorder=6)

        ax.text(px, py - 0.28, PATTERNS[i]["name"], fontsize=6.5,
                ha="center", color="#aaa", fontfamily="monospace", zorder=6)

    # Axes
    ax.axhline(0, color="#333", lw=0.8, zorder=1)
    ax.axvline(0, color="#333", lw=0.8, zorder=1)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x₁ (Entrada 1)", color="#666", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("x₂ (Entrada 2)", color="#666", fontsize=9, fontfamily="monospace")
    ax.tick_params(colors="#444", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Legend
    leg_pos = mpatches.Patch(color="#1a3a1a", ec="#5fdf6f", lw=1.5, label="Objetivo: +1 (▲)")
    leg_neg = mpatches.Patch(color="#3a1a1a", ec="#df5f5f", lw=1.5, label="Objetivo: −1 (▼)")
    ax.legend(handles=[leg_pos, leg_neg], loc="upper left",
              facecolor="#1a1a1a", edgecolor="#333",
              labelcolor="#ccc", fontsize=7.5)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Live output table ────────────────────────────────
    st.markdown("### 🔢 Tabla de salidas en tiempo real")
    header_html = """
    <div style="display:grid; grid-template-columns:1fr 0.6fr 0.6fr 0.7fr 0.7fr 0.5fr;
                gap:4px; font-family:'Space Mono',monospace; font-size:0.7rem;
                color:#555; text-transform:uppercase; letter-spacing:1px;
                padding: 0 4px 6px 4px;">
      <div>Patrón</div><div>x₁</div><div>x₂</div>
      <div>z</div><div>Pred</div><div>OK?</div>
    </div>
    """
    rows_html = ""
    for i, pat in enumerate(PATTERNS):
        x1v = input_value(pat["s1"])
        x2v = input_value(pat["s2"])
        z, pred = perceptron_output(x1v, x2v, w1, w2, bias)
        target_int = 1 if st.session_state.targets[i] else -1
        ok = pred == target_int
        pred_col = "#5fdf6f" if pred == 1 else "#df5f5f"
        ok_sym = "✅" if ok else "❌"
        rows_html += f"""
        <div style="display:grid; grid-template-columns:1fr 0.6fr 0.6fr 0.7fr 0.7fr 0.5fr;
                    gap:4px; font-family:'Space Mono',monospace; font-size:0.75rem;
                    color:#ccc; padding:5px 4px; border-bottom:1px solid #1e1e1e;">
          <div style="color:#eee;">{pat['name']}</div>
          <div>{x1v:+.0f}</div>
          <div>{x2v:+.0f}</div>
          <div style="color:#f0c040;">{z:+.2f}</div>
          <div style="color:{pred_col};">{'(+1)' if pred==1 else '(−1)'}</div>
          <div>{ok_sym}</div>
        </div>"""

    st.markdown(f"""
    <div class="card" style="padding:12px 16px;">
      {header_html}{rows_html}
    </div>
    """, unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace;
            font-size:0.7rem; color:#444; padding:12px 0 24px 0;">
  Inspirado en la máquina perceptrón de Welch Labs · Frank Rosenblatt (1958) · Autómatas, Gramáticas y Lenguajes
</div>
""", unsafe_allow_html=True)
