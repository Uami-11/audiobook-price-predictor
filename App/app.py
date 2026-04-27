import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
INR_TO_NPR = 1.6  # price in dataset is INR; display in NPR

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mars Audiobook Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700; }
    .stApp { background-color: #0f1117; color: #e8e8e8; }
    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2535 100%);
        border: 1px solid #2a3145;
        border-radius: 4px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 0.8rem;
    }
    .metric-card h4 {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.3rem 0;
    }
    .metric-card .value {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: #f0f0f0;
        line-height: 1;
    }
    .metric-card .sub { font-size: 0.78rem; color: #6b7280; margin-top: 0.3rem; }
    .result-high {
        background: linear-gradient(135deg, #0d2b1a, #1a3d28);
        border: 1px solid #2d6a47;
        border-radius: 4px;
        padding: 2rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #2b0d0d, #3d1a1a);
        border: 1px solid #6a2d2d;
        border-radius: 4px;
        padding: 2rem;
        text-align: center;
    }
    .result-price {
        background: linear-gradient(135deg, #0d1f2b, #1a2e3d);
        border: 1px solid #2d5a6a;
        border-radius: 4px;
        padding: 2rem;
        text-align: center;
    }
    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-sub { font-size: 0.9rem; color: #9ca3af; }
    .insight-box {
        background: #1a1f2e;
        border-left: 3px solid #3b82f6;
        border-radius: 0 4px 4px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        color: #c8cdd8;
    }
    .insight-warn {
        background: #1f1a0d;
        border-left: 3px solid #f59e0b;
        border-radius: 0 4px 4px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        color: #c8cdd8;
    }
    .placeholder-box {
        background: #1a1f2e;
        border: 1px dashed #2a3145;
        border-radius: 4px;
        padding: 3.5rem 2rem;
        text-align: center;
        color: #4b5563;
        font-size: 0.9rem;
    }
    .tag {
        display: inline-block;
        background: #1e2535;
        border: 1px solid #2a3145;
        border-radius: 3px;
        padding: 0.15rem 0.55rem;
        font-size: 0.74rem;
        color: #6b7280;
        margin-right: 0.35rem;
        font-family: monospace;
    }
    div[data-testid="stMetric"] {
        background: #1a1f2e;
        border: 1px solid #2a3145;
        border-radius: 4px;
        padding: 0.8rem 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 4px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
    hr { border-color: #2a3145; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "model_ready.csv")
    df = pd.read_csv(path)
    if "price" in df.columns:
        df["price_npr"] = df["price"] * INR_TO_NPR
    return df


@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    try:
        pm = joblib.load(os.path.join(base, "price_model.pkl"))
        ps = joblib.load(os.path.join(base, "price_scaler.pkl"))
        lf = joblib.load(os.path.join(base, "linear_features.pkl"))
        rm = joblib.load(os.path.join(base, "rating_model.pkl"))
        rs = joblib.load(os.path.join(base, "rating_scaler.pkl"))
        lgf = joblib.load(os.path.join(base, "logistic_features.pkl"))
        return pm, ps, lf, rm, rs, lgf
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return None, None, None, None, None, None


df = load_data()
price_model, price_scaler, linear_feats, rating_model, rating_scaler, logistic_feats = (
    load_models()
)

LANGUAGES = ["English", "German", "Spanish", "French", "Italian", "Other"]
LANG_COL_MAP = {
    "German": "lang_German",
    "Spanish": "lang_Spanish",
    "Italian": "lang_Italian",
    "Other": "lang_Other",
}


def build_row(
    feats,
    duration_min,
    star_score,
    log_ratings,
    release_year,
    is_top_nar,
    is_top_auth,
    price_inr=None,
    language="English",
):
    row = {f: 0 for f in feats}
    row["duration_minutes"] = duration_min
    row["log_num_ratings"] = log_ratings
    row["release_year"] = release_year
    row["is_top_narrator"] = int(is_top_nar)
    row["is_top_author"] = int(is_top_auth)
    if "star_score" in feats and star_score is not None:
        row["star_score"] = star_score
    if "price" in feats and price_inr is not None:
        row["price"] = price_inr
    lang_col = LANG_COL_MAP.get(language)
    if lang_col and lang_col in feats:
        row[lang_col] = 1
    return pd.DataFrame([row])[feats]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Mars Audiobook Analyzer")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.8rem;margin-top:-0.5rem'>"
        "Team Mars &middot; DATA200</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "EDA Dashboard", "Price Predictor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.75rem'>"
        "Dataset: ~87k Audible audiobooks<br>"
        "Source prices: INR<br>"
        "Displayed prices: NPR (INR x 1.6)"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Mars Audiobook Analyzer")
    st.markdown(
        "<p style='color:#9ca3af;font-size:1.05rem;margin-top:-0.8rem'>"
        "Applied Statistics Project &middot; Team Mars &middot; DATA200</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Problem Statement")
        st.markdown(
            "<div class='insight-box'>"
            "What audiobook characteristics — including duration, price, language, release year, "
            "and narrator prominence — are significant predictors of audiobook pricing and listener "
            "satisfaction on Audible?"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Approach")
        st.markdown("""
Two statistical models were applied to approximately 87,000 audiobooks from the Audible catalog:

- **Linear Regression** — predict audiobook price from content and catalog features
- **Logistic Regression** — classify whether an audiobook receives a high star rating (4.0 or above)

Both models were trained on an 80/20 train-test split with standardised features.
Prices in the source dataset are in Indian Rupees (INR) and are converted to
Nepali Rupees (NPR) throughout this application by multiplying by 1.6.
        """)

        st.markdown("### Dataset Columns")
        col_info = {
            "name": "Title of the audiobook",
            "author": "Author — cleaned from Writtenby: prefix",
            "narrator": "Narrator — cleaned from Narratedby: prefix",
            "time": "Duration — parsed into duration_minutes (integer)",
            "releasedate": "Release date — year extracted as release_year",
            "language": "Language — grouped into 6 categories for encoding",
            "stars": "Star score out of 5",
            "ratings": "Number of listener reviews — log-transformed for modeling",
            "price": "Original price in INR — displayed in NPR (x 1.6) throughout this app",
        }
        for col, desc in col_info.items():
            st.markdown(
                f"<span class='tag'>{col}</span>"
                f"<span style='color:#9ca3af;font-size:0.87rem'>{desc}</span><br>",
                unsafe_allow_html=True,
            )

        st.markdown("### Key Findings")
        st.markdown("""
- **Duration** is the strongest predictor of price — the largest coefficient in the linear model
- **Language** significantly affects both price and rating — German, Italian, and Spanish audiobooks
  are priced lower and less likely to be highly rated than English titles
- **Narrator and author prominence** reduce price but are not statistically significant predictors of rating
- **Number of ratings** is the dominant predictor of listener satisfaction — by a large margin
        """)

    with col2:
        st.markdown("### Dataset")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "<div class='metric-card'><h4>Total Records</h4><div class='value'>87k</div><div class='sub'>After cleaning</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='metric-card'><h4>Features</h4><div class='value'>11</div><div class='sub'>Per model</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<div class='metric-card'><h4>Languages</h4><div class='value'>30+</div><div class='sub'>Grouped to 6</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='metric-card'><h4>Years</h4><div class='value'>2000–2023</div><div class='sub'>Release range</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Model Performance")
        st.markdown(
            "<div class='metric-card'><h4>Linear Regression R&sup2;</h4><div class='value'>0.527</div><div class='sub'>52.7% of price variance explained</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='metric-card'><h4>Logistic Regression AUC</h4><div class='value'>0.993</div><div class='sub'>Near-perfect classification</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='metric-card'><h4>Price Currency</h4><div class='value'>NPR</div><div class='sub'>Source: INR &times; 1.6</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Team Members")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            "<div class='metric-card'><h4>Team Leader</h4><div class='value' style='font-size:1.1rem'>Nirwan Maharjan</div><div class='sub'>Data Cleaning &middot; EDA &middot; Feature Engineering</div></div>",
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            "<div class='metric-card'><h4>Statistical Modeler</h4><div class='value' style='font-size:1.1rem'>Ugen Basnet</div><div class='sub'>Model Building &middot; Validation &middot; Diagnostics</div></div>",
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            "<div class='metric-card'><h4>Literature &amp; Dev</h4><div class='value' style='font-size:1.1rem'>Bishesh Chapagain</div><div class='sub'>Literature Review &middot; Application</div></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════
elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    st.markdown(
        "<p style='color:#9ca3af;margin-top:-0.8rem'>"
        "Interactive visualisations from the cleaned Audible dataset. "
        "All prices are shown in NPR (Rs.)."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    PLOT_BG = "#1a1f2e"
    BASE_LAYOUT = dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color="#e8e8e8"),
        margin=dict(t=44, b=20, l=10, r=10),
    )

    # ── Distributions ─────────────────────────────────────
    st.markdown("### Distributions")
    d1, d2 = st.columns(2)

    with d1:
        price_cap_npr = st.slider("Price cap (NPR Rs.)", 500, 20000, 6000, step=500)
        price_col = "price_npr" if "price_npr" in df.columns else "price"
        filtered_p = df[df[price_col].between(1, price_cap_npr)][price_col]
        fig = px.histogram(
            filtered_p,
            nbins=60,
            color_discrete_sequence=["#3b82f6"],
            template="plotly_dark",
            labels={"value": "Price (NPR)"},
            title="Price Distribution (NPR)",
        )
        fig.update_layout(**BASE_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        if "star_score" in df.columns:
            rated = df[df["star_score"] > 0]["star_score"]
            fig2 = px.histogram(
                rated,
                nbins=30,
                color_discrete_sequence=["#f59e0b"],
                template="plotly_dark",
                labels={"value": "Star Score"},
                title="Star Score Distribution (Rated Books Only)",
            )
            fig2.add_vline(
                x=4.0,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="4.0 classification cutoff",
                annotation_position="top right",
            )
            fig2.update_layout(**BASE_LAYOUT, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Catalog Breakdown ─────────────────────────────────
    st.markdown("### Catalog Breakdown")
    e1, e2 = st.columns(2)

    with e1:
        if "language_grouped" in df.columns:
            lc = df["language_grouped"].value_counts()
        else:
            lc = pd.Series(
                {
                    "English": 70000,
                    "German": 5000,
                    "Spanish": 4000,
                    "French": 3000,
                    "Italian": 2000,
                    "Other": 3000,
                }
            )
        fig3 = px.bar(
            x=lc.values,
            y=lc.index,
            orientation="h",
            color=lc.values,
            color_continuous_scale="Blues",
            template="plotly_dark",
            labels={"x": "Number of Audiobooks", "y": "Language"},
            title="Audiobooks by Language",
        )
        fig3.update_layout(**BASE_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with e2:
        if "release_year" in df.columns:
            yr = (
                df[df["release_year"].between(2000, 2023)]["release_year"]
                .value_counts()
                .sort_index()
            )
            fig4 = px.line(
                x=yr.index,
                y=yr.values,
                markers=True,
                color_discrete_sequence=["#10b981"],
                template="plotly_dark",
                labels={"x": "Year", "y": "Number of Releases"},
                title="Audiobooks Released Per Year",
            )
            fig4.update_layout(**BASE_LAYOUT, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Ratings ───────────────────────────────────────────
    st.markdown("### Ratings")
    r1, r2 = st.columns(2)

    with r1:
        if "num_ratings" in df.columns:
            fig_r = px.histogram(
                np.log1p(df[df["num_ratings"] > 0]["num_ratings"]),
                nbins=50,
                color_discrete_sequence=["#8b5cf6"],
                template="plotly_dark",
                labels={"value": "log(Number of Ratings + 1)"},
                title="Ratings Count — Log Scale",
            )
            fig_r.update_layout(**BASE_LAYOUT, showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)

    with r2:
        if "num_ratings" in df.columns and "star_score" in df.columns:
            samp = df[(df["num_ratings"] > 0) & (df["star_score"] > 0)].sample(
                min(4000, len(df)), random_state=42
            )
            fig_rs = px.scatter(
                samp,
                x=np.log1p(samp["num_ratings"]),
                y="star_score",
                opacity=0.3,
                color_discrete_sequence=["#10b981"],
                template="plotly_dark",
                labels={"x": "log(Number of Ratings + 1)", "star_score": "Star Score"},
                title="Star Score vs Number of Ratings",
            )
            fig_rs.update_layout(**BASE_LAYOUT, showlegend=False)
            st.plotly_chart(fig_rs, use_container_width=True)

    # ── Price vs Duration ─────────────────────────────────
    st.markdown("### Price vs Duration")
    lang_filter = st.multiselect(
        "Filter by language",
        options=LANGUAGES,
        default=["English", "German", "Spanish"],
    )

    pc = "price_npr" if "price_npr" in df.columns else "price"
    sc_df = (
        df[(df[pc].between(1, 50000)) & (df["duration_minutes"] < 1500)].copy()
        if "duration_minutes" in df.columns
        else df.copy()
    )
    if "language_grouped" in sc_df.columns and lang_filter:
        sc_df = sc_df[sc_df["language_grouped"].isin(lang_filter)]
    sc_samp = sc_df.sample(min(5000, len(sc_df)), random_state=42)

    fig5 = px.scatter(
        sc_samp,
        x="duration_minutes",
        y=pc,
        color="language_grouped" if "language_grouped" in sc_samp.columns else None,
        opacity=0.35,
        trendline="ols",
        template="plotly_dark",
        labels={
            "duration_minutes": "Duration (minutes)",
            pc: "Price (NPR)",
            "language_grouped": "Language",
        },
        title="Price (NPR) vs Duration — with OLS trend line",
    )
    fig5.update_layout(**BASE_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    # ── Star Score by Language ────────────────────────────
    st.markdown("### Star Score by Language")
    if "language_grouped" in df.columns and "star_score" in df.columns:
        top6 = df["language_grouped"].value_counts().head(6).index.tolist()
        box_df = df[df["language_grouped"].isin(top6) & (df["star_score"] > 0)]
        fig_box = px.box(
            box_df,
            x="language_grouped",
            y="star_score",
            color="language_grouped",
            template="plotly_dark",
            labels={"language_grouped": "Language", "star_score": "Star Score"},
            title="Star Score Distribution by Language (Top 6)",
        )
        fig_box.update_layout(**BASE_LAYOUT, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation Heatmap ───────────────────────────────
    st.markdown("### Correlation Heatmap")
    num_cols = [
        c
        for c in [
            "price",
            "star_score",
            "log_num_ratings",
            "duration_minutes",
            "release_year",
        ]
        if c in df.columns
    ]
    corr = df[num_cols].corr().round(2)
    labels = ["price (INR)" if c == "price" else c for c in corr.columns]

    fig6 = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig6.update_layout(**BASE_LAYOUT, title="Feature Correlation Matrix", height=420)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown(
        "<div class='insight-warn'>The correlation heatmap uses the original INR price column, "
        "which is the value the model was trained on. All other price displays in this app are "
        "in NPR (INR x 1.6).</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
# PAGE 3 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "Price Predictor":
    st.title("Price Predictor")
    st.markdown(
        "<p style='color:#9ca3af;margin-top:-0.8rem'>"
        "Linear Regression &middot; R&sup2; = 0.527 &middot; Predictions shown in NPR"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Audiobook Details")

        duration_hrs = st.number_input("Duration (hours)", 0.0, 100.0, 8.0, 0.5)
        duration_min = duration_hrs * 60

        star_val = st.slider("Star Score", 0.0, 5.0, 4.2, 0.1)

        num_rat = st.number_input("Number of Ratings", 0, 500000, 1500, 100)
        log_rat = np.log1p(num_rat)

        rel_year = st.slider("Release Year", 2000, 2024, 2020)
        language = st.selectbox("Language", LANGUAGES)

        nc, ac = st.columns(2)
        with nc:
            top_nar = st.toggle("Top Narrator", False)
        with ac:
            top_auth = st.toggle("Top Author", False)

        run = st.button("Predict Price")

    with col_result:
        st.markdown("#### Result")

        if run and price_model is not None:
            Xi = build_row(
                linear_feats,
                duration_min,
                star_val,
                log_rat,
                rel_year,
                top_nar,
                top_auth,
                language=language,
            )
            Xs = price_scaler.transform(Xi)
            inr_val = max(0.0, float(price_model.predict(Xs)[0]))
            npr_val = inr_val * INR_TO_NPR

            st.markdown(
                f"""
            <div class='result-price'>
                <div class='result-label'>Rs. {npr_val:,.0f}</div>
                <div class='result-sub'>Predicted Audible price in NPR</div>
                <div class='result-sub' style='font-size:0.78rem;color:#4b5563;margin-top:0.35rem'>
                    ({inr_val:,.0f} INR before conversion)
                </div>
            </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**What drove this prediction**")

            if duration_hrs > 15:
                st.markdown(
                    "<div class='insight-box'>Duration is the strongest price driver in the model. Books over 15 hours carry a significant price premium.</div>",
                    unsafe_allow_html=True,
                )
            if language in ["German", "Italian", "Spanish"]:
                st.markdown(
                    f"<div class='insight-box'>{language} audiobooks have a large negative coefficient — they are priced significantly lower than English titles in this dataset.</div>",
                    unsafe_allow_html=True,
                )
            if top_nar or top_auth:
                st.markdown(
                    "<div class='insight-box'>Top narrator and top author flags are negatively associated with price — prolific creators appear across all price tiers including budget releases.</div>",
                    unsafe_allow_html=True,
                )
            if num_rat > 10000:
                st.markdown(
                    "<div class='insight-box'>High ratings count has a small negative effect on price — widely reviewed books tend to be mainstream, competitively priced titles.</div>",
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                "<div class='placeholder-box'>Fill in the details on the left<br>and click "
                "<strong style='color:#6b7280'>Predict Price</strong> to get a result.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", "0.527")
        m2.metric("MAE (INR)", "162.59")
        m3.metric("RMSE (INR)", "220.90")
        st.markdown(
            "<div class='insight-warn'>MAE and RMSE are in INR (the unit used during training). "
            "In NPR these are approximately Rs. 260 and Rs. 354 respectively. "
            "Error values are elevated because the dataset contains prices from multiple "
            "Audible regional stores — not all entries are in INR. "
            "This is noted as a limitation in the project report.</div>",
            unsafe_allow_html=True,
        )
