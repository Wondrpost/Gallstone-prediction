
import os
import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None


st.set_page_config(
    page_title="Gallstone Risk Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BODY_COMPOSITION = [
    ("tbw",  "Total Body Water (TBW)",      "L",   "Total Body Water (TBW)",          13.0, 66.0, 0.1, 39.8, 40.59, 7.93,  0.10),
    ("ecw",  "Extracellular Water (ECW)",   "L",   "Extracellular Water (ECW)",        9.0, 28.0, 0.1, 17.1, 17.07, 3.16,  0.15),
    ("lm",   "Lean Mass (LM)",              "%",   "Lean Mass (LM) (%)",              49.0, 94.0, 0.1, 72.1, 71.64, 8.44, -0.40),
    ("tbfr", "Total Body Fat Ratio (TBFR)", "%",   "Total Body Fat Ratio (TBFR) (%)",  6.0, 51.0, 0.1, 27.8, 28.27, 8.44,  0.50),
    ("bm",   "Bone Mass (BM)",              "kg",  "Bone Mass (BM)",                   1.4,  4.0, 0.1,  2.8,  2.80, 0.51, -0.10),
    ("tfc",  "Total Fat Content (TFC)",     "kg",  "Total Fat Content (TFC)",          3.0, 63.0, 0.1, 22.6, 23.49, 9.61,  0.40),
]

BLOOD_MARKERS = [
    ("vitd", "Vitamin D",                   "ng/mL", "Vitamin D",                      0.0, 55.0, 0.1, 22.0, 21.40, 9.98, -0.50),
    ("crp",  "C-Reactive Protein (CRP)",    "mg/L",  "C-Reactive Protein (CRP)",       0.0, 44.0, 0.1,  0.2,  1.85, 4.99,  0.70),
    ("hgb",  "Hemoglobin (HGB)",            "g/dL",  "Hemoglobin (HGB)",               8.0, 19.0, 0.1, 14.4, 14.42, 1.78, -0.20),
]

ALL_FEATURES = BODY_COMPOSITION + BLOOD_MARKERS
DEFAULTS = {f[0]: f[7] for f in ALL_FEATURES}

THRESHOLD = 0.50          
MODEL_NAME = "Random Forest"
RECALL = "76.1%"     


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "gallstone_model.joblib")
    if joblib is not None and os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

MODEL = load_model()


def predict_probability(values: dict) -> float:
    """Return P(gallstone) in [0,1]. Uses the trained model when available,
    otherwise a transparent logistic demo estimate."""
    if MODEL is not None:
        col_for = {f[0]: f[3] for f in ALL_FEATURES}
        row = {col_for[k]: values[k] for k in values}
        X = pd.DataFrame([row])
        feat_names = getattr(MODEL, "feature_names_in_", None)
        if feat_names is not None:
            X = X.reindex(columns=list(feat_names), fill_value=0)
        try:
            return float(MODEL.predict_proba(X)[0][1])
        except Exception:
            pass
    
    z = -0.15
    for f in ALL_FEATURES:
        key, mean, std, w = f[0], f[8], f[9], f[10]
        z += w * ((values[key] - mean) / std)
    return 1.0 / (1.0 + np.exp(-z))



st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

      .stApp, .main, [data-testid="stAppViewContainer"] { background: #F5F3EF !important; }
      html, body, [class*="css"], .stMarkdown, input, button, label, p, h1, h2, h3 {
          font-family: 'Manrope', sans-serif !important;
          color: #33302B;
      }
      .block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 3rem; }
      header[data-testid="stHeader"] { background: transparent; }
      #MainMenu, footer { visibility: hidden; }

      /* ---- White section cards: target st.container(border=True) ---- */
      div[data-testid="stVerticalBlockBorderWrapper"] {
          background: #FFFFFF;
          border: 1px solid #ECE9E2 !important;
          border-radius: 20px !important;
          box-shadow: 0 1px 2px rgba(60,55,45,.04);
      }
      div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 6px 6px; }

      /* ---- Eyebrow + helper text ---- */
      .gs-eyebrow {
          display:flex; align-items:center; gap:9px;
          font-size: 12.5px; font-weight: 700; letter-spacing: .1em;
          text-transform: uppercase; color: #7C7790; margin: 4px 0 2px;
      }
      .gs-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
      .gs-help { font-size: 13px; color: #A29C90; margin: 0 0 10px; }

      /* ---- Reference hint under each input ---- */
      .gs-hint { font-size: 11.5px; color: #B0AA9E; margin: 3px 0 12px; line-height: 1.3; }
      .gs-hint b { color: #7C7790; font-weight: 700; }

      /* ---- Number inputs ---- */
      /* Fixed-height labels so wrapping (2-line) labels keep every input row aligned */
      div[data-testid="stNumberInput"] label {
          min-height: 40px !important;
          align-items: flex-start !important;
          margin-bottom: 2px !important;
      }
      div[data-testid="stNumberInput"] label p {
          font-size: 13px !important; font-weight: 600 !important; color: #4A463C !important;
          line-height: 1.35 !important;
      }
      div[data-testid="stNumberInput"] div[data-baseweb="input"] {
          background: #FAF9F6 !important;
          border-radius: 11px !important;
          border: 1px solid #E6E2DA !important;
      }
      div[data-testid="stNumberInput"] input {
          background: transparent !important;
          font-weight: 700 !important; font-size: 16px !important; color: #33302B !important;
      }
      /* Stepper buttons — subtle, cream */
      div[data-testid="stNumberInput"] button {
          background: #F1EEE8 !important; border: none !important; color: #8C8678 !important;
      }
      div[data-testid="stNumberInput"] button:hover { background: #E8E4DC !important; color: #5B5670 !important; }

      /* ---- Reset link-button ---- */
      div[data-testid="stButton"] button {
          background: transparent !important; border: none !important;
          color: #8C8678 !important; font-weight: 600 !important; font-size: 13.5px !important;
          padding: 4px 0 !important; box-shadow: none !important;
      }
      div[data-testid="stButton"] button:hover { color: #5B5670 !important; }

      /* ---- Prediction panel ---- */
      .gs-badge {
          display:inline-flex; align-items:center; gap:8px; padding:8px 15px;
          border-radius:999px; font-size:14px; font-weight:700; margin-bottom:18px;
      }
      .gs-prob { font-size:64px; font-weight:800; letter-spacing:-.03em; color:#33302B; line-height:1; }
      .gs-prob span { font-size:27px; font-weight:700; color:#8C8678; }
      .gs-sub { font-size:14px; color:#857F75; margin:6px 0 24px; }
      .gs-bar { position:relative; height:11px; background:#EFECE5; border-radius:999px; overflow:hidden; }
      .gs-bar > div { height:100%; background:linear-gradient(90deg,#C4BEEC,#9D97D8); border-radius:999px; }
      .gs-tickwrap { position:relative; height:18px; margin-top:2px; }
      .gs-tick { position:absolute; left:50%; top:0; transform:translateX(-50%);
                 font-size:11px; color:#B0AA9E; font-weight:600; }
      .gs-tick::before { content:""; position:absolute; left:50%; top:-9px; height:7px; width:1px; background:#CFC9BE; }
      .gs-meta { margin-top:22px; padding-top:18px; border-top:1px solid #EFECE5; }
      .gs-row { display:flex; justify-content:space-between; font-size:13.5px; padding:6px 0; }
      .gs-row .k { color:#A29C90; } .gs-row .v { color:#5B5670; font-weight:700; }
      .gs-note { font-size:12px; color:#B0AA9E; line-height:1.6; margin-top:16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div style="display:flex; gap:8px; margin-bottom:12px;">
      <span style="width:13px;height:13px;border-radius:4px;background:#B7B0E8;"></span>
      <span style="width:13px;height:13px;border-radius:4px;background:#AEDCC8;"></span>
      <span style="width:13px;height:13px;border-radius:4px;background:#F3C5C1;"></span>
    </div>
    <h1 style="margin:0;font-size:32px;font-weight:800;letter-spacing:-.02em;">Gallstone Risk Screening</h1>
    <p style="margin:8px 0 28px;font-size:15px;color:#857F75;max-width:52ch;line-height:1.5;">
      Estimate the likelihood of gallstones from nine body-composition and blood markers.
    </p>
    """,
    unsafe_allow_html=True,
)


def render_field(feature):
    """Render one styled, empty number_input with a reference hint below it."""
    key, label, unit = feature[0], feature[1], feature[2]
    lo, hi, mean = feature[4], feature[5], feature[8]
    st.number_input(
        f"{label} · {unit}",
        min_value=float(lo),
        max_value=float(hi),
        step=float(feature[6]),
        value=None,
        placeholder="—",
        key=key,
    )
    st.markdown(
        f'<div class="gs-hint">typical ≈ <b>{mean:g}</b> &nbsp;·&nbsp; range {lo:g}–{hi:g}</div>',
        unsafe_allow_html=True,
    )



left, right = st.columns([1.55, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            '<div class="gs-eyebrow"><span class="gs-dot" style="background:#B7B0E8;"></span>'
            'Body Composition</div><p class="gs-help">Enter the measured values.</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, f in enumerate(BODY_COMPOSITION):
            with cols[i % 3]:
                render_field(f)

    st.write("")

    with st.container(border=True):
        st.markdown(
            '<div class="gs-eyebrow"><span class="gs-dot" style="background:#AEDCC8;"></span>'
            'Blood Markers</div><p class="gs-help">Enter precise lab values.</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, f in enumerate(BLOOD_MARKERS):
            with cols[i % 3]:
                render_field(f)


raw = {k: st.session_state.get(k) for k in DEFAULTS}
complete = all(v is not None for v in raw.values())

if complete:
    values = {k: float(raw[k]) for k in DEFAULTS}
    prob = predict_probability(values)
    pct = round(prob * 100)
    is_high = prob >= THRESHOLD

    if is_high:
        badge = ('<div class="gs-badge" style="background:#F8E9E7;border:1px solid #F0D5D2;color:#A65B55;">'
                 '<span style="width:9px;height:9px;border-radius:50%;background:#E0938C;"></span>Gallstone likely</div>')
    else:
        badge = ('<div class="gs-badge" style="background:#E5F2EC;border:1px solid #D0E7DC;color:#42826A;">'
                 '<span style="width:9px;height:9px;border-radius:50%;background:#7FC2A6;"></span>No gallstone detected</div>')
    prob_block = f'<div class="gs-prob">{pct}<span> %</span></div>'
    bar_fill = f'{pct}%'
else:
    badge = ('<div class="gs-badge" style="background:#F1EEE8;border:1px solid #E6E2DA;color:#8C8678;">'
             '<span style="width:9px;height:9px;border-radius:50%;background:#C9C3B6;"></span>Awaiting inputs</div>')
    prob_block = '<div class="gs-prob" style="color:#CFC9BE;">—<span> %</span></div>'
    bar_fill = '0%'

sub_text = ('estimated probability of gallstones' if complete
            else 'fill in all nine values to see a prediction')

with right:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="gs-eyebrow" style="margin-bottom:16px;">Prediction</div>
            {badge}
            {prob_block}
            <p class="gs-sub">{sub_text}</p>
            <div class="gs-bar"><div style="width:{bar_fill};"></div></div>
            <div class="gs-tickwrap"><span class="gs-tick">50%</span></div>
            <div class="gs-meta">
              <div class="gs-row"><span class="k">Model</span><span class="v">{MODEL_NAME}</span></div>
              <div class="gs-row"><span class="k">Decision threshold</span><span class="v">50%</span></div>
              <div class="gs-row"><span class="k">Recall</span><span class="v">{RECALL}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="gs-note">Screening aid based on a trained model. Not a medical diagnosis.</p>',
        unsafe_allow_html=True,
    )
