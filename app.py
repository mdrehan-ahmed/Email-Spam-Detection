import streamlit as st
import streamlit.components.v1 as components
import email
import io
from email import policy
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import re
import time
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Email Spam Detection", page_icon="🛡️", layout="wide")

# Show a realistic loading screen only for a new app session/page refresh.
# It is cleared after the model is ready and does not reappear on normal
# widget interactions such as typing, uploading, or clicking Analyze.
if "app_loaded" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <style>
        .startup-loading {
            position: fixed;
            inset: 0;
            z-index: 999999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #070a13;
            color: #f3f4f6;
            font-family: sans-serif;
        }
        .startup-loading-title { font-size: 24px; font-weight: 700; letter-spacing: .3px; }
        .startup-loading-subtitle { color: #94a3b8; font-size: 13px; margin-top: 8px; }
        .startup-dots { color: #60a5fa; font-size: 28px; letter-spacing: 4px; margin-top: 6px; }
        .startup-dots span { animation: startup-blink 1.4s infinite both; }
        .startup-dots span:nth-child(2) { animation-delay: .2s; }
        .startup-dots span:nth-child(3) { animation-delay: .4s; }
        .startup-dots span:nth-child(4) { animation-delay: .6s; }
        .startup-dots span:nth-child(5) { animation-delay: .8s; }
        @keyframes startup-blink { 0%, 80%, 100% { opacity: .2; } 40% { opacity: 1; } }
    </style>
    <div class="startup-loading">
        <div class="startup-loading-title">Loading</div>
        <div class="startup-dots"><span>.</span><span>.</span><span>.</span><span>.</span><span>.</span></div>
        <div class="startup-loading-subtitle">Preparing Email Spam Detection</div>
    </div>
    """, unsafe_allow_html=True)
else:
    loading_placeholder = None

# 1. Cache EasyOCR model. It is loaded only when an image is uploaded,
#    instead of slowing down every app start.
@st.cache_resource
def load_ocr_reader():
    # Import EasyOCR only when an image actually needs to be analyzed.
    import easyocr
    return easyocr.Reader(['en'], gpu=False)


@st.cache_data(show_spinner=False)
def extract_text_from_image(image_bytes):
    """Run OCR once per image and cache the extracted text."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # OCR on a very large screenshot is unnecessarily slow on CPU.
    # Keep enough resolution for email text while limiting processing time.
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    image_np = np.array(image)

    reader = load_ocr_reader()
    results = reader.readtext(
        image_np,
        detail=0,
        paragraph=True,
        canvas_size=1600,
        mag_ratio=1.0,
        decoder="greedy",
    )
    return " ".join(results).strip()

# 2. MACHINE LEARNING MODEL TRAINING (Cached so it runs only once)
@st.cache_resource
def train_spam_model():
    csv_path = os.path.join("data", "spam.csv")
    
    if os.path.exists(csv_path):
        try:
            # Read CSV with fallback encoding for Kaggle datasets
            df = pd.read_csv(csv_path, encoding='latin-1')
            
            # Auto-detect column names (handles 'v1/v2' or 'Category/Message' or 'label/text')
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df[['v1', 'v2']]
                df.columns = ['label', 'text']
            elif 'Category' in df.columns and 'Message' in df.columns:
                df = df[['Category', 'Message']]
                df.columns = ['label', 'text']
            
            # Map ham/spam to numeric values 0 and 1
            df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
            df = df.dropna(subset=['label', 'text'])
            
            X = df['text']
            y = df['label'].astype(int)
            dataset_type = "Real CSV Dataset (data/spam.csv)"
        except Exception as e:
            st.error(f"Error loading CSV from data folder: {e}")
            X, y, dataset_type = load_dummy_data()
    else:
        X, y, dataset_type = load_dummy_data()

    # Vectorization & Training
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_vectorized = vectorizer.fit_transform(X)
    
    model = MultinomialNB()
    model.fit(X_vectorized, y)
    
    return vectorizer, model, dataset_type

def load_dummy_data():
    # Short dummy dataset for fallback so app never crashes
    data = {
        'text': [
            "Win a free iPhone right now, click here to claim your cash prize!",
            "Urgent: Your bank account has been compromised, verify password.",
            "Congratulations you have won the lottery of $500000.",
            "Hey bro, are we still meeting for lunch tomorrow?",
            "Please find the attached project report for Q3.",
            "nhvjfhchvvn njmv asdasd", 
            "Can you review the pull request on GitHub when you have time?"
        ],
        'label': [1, 1, 1, 0, 0, 1, 0] # 1 is Spam, 0 is Authentic
    }
    df = pd.DataFrame(data)
    return df['text'], df['label'], "In-built Dummy Dataset"

# Load the ML Model
vectorizer, model, current_dataset = train_spam_model()

if loading_placeholder is not None:
    loading_placeholder.empty()
    st.session_state.app_loaded = True


# Custom CSS (Awesome UI & Uploader Fixes)
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #070a13 !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .main .block-container { max-width: 1050px; padding-top: 2rem; padding-bottom: 2rem; }
    textarea { background-color: #080d1a !important; color: #f3f4f6 !important; border: 1px solid #1e293b !important; border-radius: 10px !important; font-size: 14px !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #080d1a !important; border: 1.5px dashed #2563eb !important; border-radius: 12px !important; height: 200px !important; display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; padding: 10px !important; position: relative !important; }
    [data-testid="stFileUploaderDropzone"] > div { display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; text-align: center !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] small, [data-testid="stFileUploaderDropzoneInstructions"] span:not(button span) { display: none !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] button { border-radius: 50% !important; width: 50px !important; height: 50px !important; min-width: 50px !important; min-height: 50px !important; padding: 0 !important; display: flex !important; align-items: center !important; justify-content: center !important; background-color: #0f172a !important; border: 1.5px solid #2563eb !important; margin: 0 auto 12px auto !important; box-shadow: 0 0 12px rgba(37, 99, 235, 0.25) !important; position: relative !important; cursor: pointer !important; z-index: 10 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] button * { pointer-events: none !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] button::after { content: "↑" !important; font-size: 24px !important; font-weight: 700 !important; color: #3b82f6 !important; display: flex !important; align-items: center !important; justify-content: center !important; line-height: 1 !important; pointer-events: none !important; }
    [data-testid="stFileUploaderDropzoneInstructions"]::after { content: "Or Drag & Drop file here.\\A Maximum 100 MB per file." !important; white-space: pre !important; text-align: center !important; color: #94a3b8 !important; font-size: 13px !important; line-height: 25px !important; display: block !important; margin: 0 !important; pointer-events: none !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] button:hover { background-color: #2563eb !important; box-shadow: 0 0 18px rgba(37, 99, 235, 0.6) !important; border-color: #60a5fa !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] button:hover::after { color: #ffffff !important; }
    [data-testid="stFileUploaderDeleteBtn"], [data-testid="stFileUploaderFileData"] button { background: transparent !important; border: none !important; box-shadow: none !important; width: auto !important; height: auto !important; min-width: 0 !important; min-height: 0 !important; border-radius: 4px !important; padding: 4px !important; margin: 0 !important; }
    [data-testid="stFileUploaderDeleteBtn"] *, [data-testid="stFileUploaderFileData"] button * { display: inline-block !important; }
    div.stButton > button { width: 100% !important; background: linear-gradient(90deg, #2563eb, #1d4ed8) !important; color: #ffffff !important; font-size: 17px !important; font-weight: 600 !important; height: 52px !important; border-radius: 10px !important; }
    .metric-card { background-color: #0d1322; border: 1px solid #1e293b; padding: 18px 20px; border-radius: 12px; }
    .metric-title { font-size: 14px; color: #94a3b8; font-weight: 500; }
    .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
    .result-card { background-color: #0d1322; border: 1px solid #1e293b; padding: 20px; border-radius: 12px; margin-top: 15px; }
    .status-badge { background-color: #064e3b; color: #34d399; padding: 5px 14px; border-radius: 20px; font-size: 12px; border: 1px solid #059669; }

    /* Responsive layout for phones and tablets */
    @media (max-width: 768px) {
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem 0.85rem 1.5rem !important;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.75rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        h2.app-title {
            font-size: 20px !important;
            line-height: 1.25 !important;
        }
        .app-subtitle { font-size: 12px !important; }
        .input-label { font-size: 14px !important; }
        textarea { min-height: 150px !important; }
        [data-testid="stFileUploaderDropzone"] {
            height: 155px !important;
            min-height: 155px !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"]::after {
            font-size: 12px !important;
            line-height: 20px !important;
        }
        div.stButton > button {
            min-height: 48px !important;
            font-size: 15px !important;
        }
        .metric-card { padding: 14px 16px !important; }
        .metric-title { font-size: 13px !important; }
        .metric-value { font-size: 24px !important; }
        .result-card { padding: 16px !important; }
        [data-testid="stDataFrame"] {
            max-width: 100% !important;
            overflow-x: auto !important;
        }
        [data-testid="stDataFrame"] iframe { max-width: 100% !important; }
    }

    @media (min-width: 769px) and (max-width: 1100px) {
        .main .block-container {
            max-width: 94% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    @media (max-width: 420px) {
        .main .block-container { padding-left: 0.65rem !important; padding-right: 0.65rem !important; }
        h2.app-title { font-size: 18px !important; }
        .app-subtitle { font-size: 11px !important; }
        [data-testid="stFileUploaderDropzone"] { height: 140px !important; min-height: 140px !important; }
        .metric-value { font-size: 22px !important; }
    }
</style>
""", unsafe_allow_html=True)

# Precise Capture-Phase Click Filter Script
components.html("""
<script>
    (function() {
        function attachClickFilter() {
            try {
                const doc = window.parent.document;
                const dropzones = doc.querySelectorAll('[data-testid="stFileUploaderDropzone"]');
                dropzones.forEach(dropzone => {
                    if (!dropzone.dataset.clickRestricted) {
                        dropzone.dataset.clickRestricted = "true";
                        dropzone.addEventListener('click', function(e) {
                            // Allow every button inside the uploader, including
                            // Streamlit's + button shown after the first file.
                            const clickedButton = e.target.closest('button');
                            const isUploaderButton = clickedButton && dropzone.contains(clickedButton);
                            if (!isUploaderButton) { e.stopPropagation(); e.stopImmediatePropagation(); }
                        }, true);
                    }
                });
            } catch(e) {}
        }
        attachClickFilter();
        setInterval(attachClickFilter, 400);
    })();
</script>
""", height=0, width=0)

# Header
with st.container():
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background-color: #1e293b; padding: 10px; border-radius: 10px; border: 1px solid #334155;">
                <span style="font-size: 24px;">✉️</span>
            </div>
            <div>
                <h2 class="app-title" style="margin: 0; color: #ffffff; font-size: 22px;">Email Spam Detection</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='input-label' style='color: #f3f4f6; font-weight: 600; font-size: 15px; margin-bottom: 4px;'>📄 Paste Email</div>", unsafe_allow_html=True)
    email_input = st.text_area("Paste text", placeholder="Paste your email content here...", height=200, label_visibility="collapsed")

with col2:
    st.markdown("<div class='input-label' style='color: #f3f4f6; font-weight: 600; font-size: 15px; margin-bottom: 4px;'>☁️ Upload File / Screenshot</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "eml", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Analyze Button
analyze_btn = st.button("☑️ Analyze Email", use_container_width=True)

def analyze_text(text):
    """Return the three metrics and result message for one email."""
    lower_text = text.lower()
    word_count = len(lower_text.split())

    if bool(re.search(r'[bcdfghjklmnpqrstvwxz]{6,}', lower_text)) or (word_count < 3 and len(lower_text) > 15):
        return "99.0%", "1/100", "100%", "⚠️ SUSPICIOUS / GIBBERISH! The text appears to be random typing.", "#f59e0b"
    if word_count < 3:
        return "--", "--", "--", "ℹ️ TOO SHORT! Please provide a longer sentence or full email.", "#3b82f6"

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    prob_ham, prob_spam = probabilities[0], probabilities[1]
    result = (
        "🚨 SPAM DETECTED BY AI! The text matches known spam patterns."
        if prediction == 1
        else "✅ AUTHENTIC EMAIL! The text appears likely to be legitimate."
    )
    return (
        f"{prob_spam * 100:.1f}%",
        f"{int(prob_ham * 100)}/100",
        f"{max(prob_ham, prob_spam) * 100:.1f}%",
        result,
        "#ef4444" if prediction == 1 else "#10b981",
    )

def render_result(title, result):
    spam_prob_text, auth_score_text, confidence_text, result_text, result_color = result
    st.markdown(f"<h4 style='color:#f3f4f6; margin:18px 0 8px;'>🖼️ {title}</h4>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">🚨 Spam Probability</div><div class="metric-value">{spam_prob_text}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">🛡️ Authenticity Score</div><div class="metric-value">{auth_score_text}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">📊 Confidence</div><div class="metric-value">{confidence_text}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-card" style="border-left: 4px solid {result_color};"><div style="font-weight:600;color:#f3f4f6;font-size:15px;margin-bottom:6px;">⭐ Analysis Result</div><div style="color:{result_color};font-size:14px;">{result_text}</div></div>', unsafe_allow_html=True)

def show_analysis_progress(item_number, total_items):
    """Display a short, realistic analysis phase before rendering a result."""
    progress = st.progress(0, text=f"Analyzing Email {item_number}/{total_items}...")
    status = st.empty()
    steps = [
        (0.20, "Reading email content..."),
        (0.45, "Checking message patterns..."),
        (0.70, "Comparing with spam signals..."),
        (0.90, "Calculating confidence..."),
        (1.00, "Analysis complete."),
    ]
    for value, message in steps:
        progress.progress(value, text=f"Analyzing Email {item_number}/{total_items}...")
        status.caption(message)
        time.sleep(0.9)
    progress.empty()
    status.empty()

def render_batch_summary(batch_results):
    """Render aggregate metrics, charts, and a downloadable CSV report."""
    if not batch_results:
        return
    total = len(batch_results)
    spam_count = sum(row["prediction"] == "Spam" for row in batch_results)
    authentic_count = total - spam_count
    confidence_values = [row["confidence"] for row in batch_results]
    average_confidence = sum(confidence_values) / total

    st.markdown("<h3 style='color:#f3f4f6; margin-top:24px;'>📦 Batch Analysis Summary</h3>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">📄 Total Emails</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">🚨 Spam Detected</div><div class="metric-value">{spam_count}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">✅ Authentic</div><div class="metric-value">{authentic_count}</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">📊 Avg. Confidence</div><div class="metric-value">{average_confidence:.1f}%</div></div>', unsafe_allow_html=True)

    # Visual dashboard based on the actual batch results.
    st.markdown("<h3 style='color:#f3f4f6; margin-top:24px;'>📈 Batch Charts Dashboard</h3>", unsafe_allow_html=True)

    def selected_value(event, selection_name):
        """Read a Vega-Lite click selection safely across Streamlit versions."""
        try:
            selection = event.selection.get(selection_name, [])
            if isinstance(selection, dict):
                return selection
            return selection[0] if selection else None
        except (AttributeError, KeyError, IndexError, TypeError):
            return None

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        prediction_counts = pd.DataFrame({
            "Prediction": ["Spam", "Authentic"],
            "Emails": [spam_count, authentic_count],
        }).set_index("Prediction")
        st.markdown("**Spam vs Authentic**")
        prediction_event = st.vega_lite_chart(
            prediction_counts.reset_index(),
            {
                "params": [{"name": "chart_click", "select": {"type": "point", "on": "click", "clear": "dblclick"}}],
                "mark": {"type": "bar", "color": "#2563eb"},
                "encoding": {
                    "x": {"field": "Prediction", "type": "nominal", "axis": {"title": None}},
                    "y": {"field": "Emails", "type": "quantitative", "axis": {"title": "Emails"}},
                    "tooltip": [
                        {"field": "Prediction", "type": "nominal", "title": "Category"},
                        {"field": "Emails", "type": "quantitative", "title": "Emails"},
                    ],
                },
                "config": {"view": {"stroke": None}},
            },
            use_container_width=True,
            on_select="rerun",
            key="prediction_chart",
        )
        prediction_selected = selected_value(prediction_event, "chart_click")
        if prediction_selected:
            st.info(f"Selected: {prediction_selected.get('Prediction', '')} — {prediction_selected.get('Emails', 0)} email(s)")
    with chart_col2:
        confidence_df = pd.DataFrame({"Confidence (%)": confidence_values})
        confidence_df["Range"] = pd.cut(
            confidence_df["Confidence (%)"],
            bins=[-1, 60, 80, 90, 100.01],
            labels=["0–60%", "61–80%", "81–90%", "91–100%"],
        )
        confidence_counts = confidence_df["Range"].value_counts().reindex(
            ["0–60%", "61–80%", "81–90%", "91–100%"], fill_value=0
        )
        st.markdown("**Confidence Distribution**")
        confidence_chart = confidence_counts.rename("Emails").reset_index()
        confidence_chart.columns = ["Range", "Emails"]
        confidence_event = st.vega_lite_chart(
            confidence_chart,
            {
                "params": [{"name": "chart_click", "select": {"type": "point", "on": "click", "clear": "dblclick"}}],
                "mark": {"type": "bar", "color": "#10b981"},
                "encoding": {
                    "x": {"field": "Range", "type": "ordinal", "sort": ["0–60%", "61–80%", "81–90%", "91–100%"], "axis": {"title": None}},
                    "y": {"field": "Emails", "type": "quantitative", "axis": {"title": "Emails"}},
                    "tooltip": [
                        {"field": "Range", "type": "ordinal", "title": "Confidence Range"},
                        {"field": "Emails", "type": "quantitative", "title": "Emails"},
                    ],
                },
                "config": {"view": {"stroke": None}},
            },
            use_container_width=True,
            on_select="rerun",
            key="confidence_chart",
        )
        confidence_selected = selected_value(confidence_event, "chart_click")
        if confidence_selected:
            st.info(f"Selected: {confidence_selected.get('Range', '')} — {confidence_selected.get('Emails', 0)} email(s)")

    probability_df = pd.DataFrame({
        "File": [row["file"] for row in batch_results],
        "Spam Probability (%)": [row["spam_probability"] for row in batch_results],
    }).set_index("File")
    st.markdown("**File-wise Spam Probability**")
    probability_event = st.vega_lite_chart(
        probability_df.reset_index(),
        {
            "params": [{"name": "chart_click", "select": {"type": "point", "on": "click", "clear": "dblclick"}}],
            "mark": {"type": "bar", "color": "#ef4444"},
            "encoding": {
                "x": {"field": "File", "type": "nominal", "sort": None, "axis": {"title": None, "labelAngle": -35}},
                "y": {"field": "Spam Probability (%)", "type": "quantitative", "scale": {"domain": [0, 100]}, "axis": {"title": "Spam Probability (%)"}},
                "tooltip": [
                    {"field": "File", "type": "nominal", "title": "File"},
                    {"field": "Spam Probability (%)", "type": "quantitative", "title": "Spam Probability"},
                ],
            },
            "config": {"view": {"stroke": None}},
        },
        use_container_width=True,
        on_select="rerun",
        key="probability_chart",
    )
    probability_selected = selected_value(probability_event, "chart_click")
    if probability_selected:
        st.info(f"Selected: {probability_selected.get('File', '')} — {probability_selected.get('Spam Probability (%)', 0):.1f}% spam probability")

    report_df = pd.DataFrame(batch_results)
    report_df["confidence"] = report_df["confidence"].map(lambda value: f"{value:.1f}%")
    report_df["spam_probability"] = report_df["spam_probability"].map(lambda value: f"{value:.1f}%")
    report_df = report_df.rename(columns={
        "file": "File",
        "prediction": "Prediction",
        "spam_probability": "Spam Probability",
        "confidence": "Confidence",
    })[["File", "Prediction", "Spam Probability", "Confidence"]]
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    csv_data = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Batch CSV Report",
        csv_data,
        file_name=f"email_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

if analyze_btn:
    items = []
    # Keep pasted content as its own item even when files are also uploaded.
    if email_input.strip():
        items.append(("Paste Email", email_input))
    if uploaded_files:
        items.extend((file.name, file) for file in uploaded_files)

    if not items:
        st.warning("Please paste email text or upload at least one file first!")
    else:
        upload_index = 0
        batch_results = []
        for index, (name, item) in enumerate(items, start=1):
            try:
                if isinstance(item, str):
                    text = item
                else:
                    upload_index += 1
                    file_ext = name.rsplit('.', 1)[-1].lower()
                    if file_ext in ["png", "jpg", "jpeg"]:
                        with st.spinner(f"Extracting text from image {index}/{len(items)}..."):
                            text = extract_text_from_image(item.getvalue())
                    elif file_ext == "eml":
                        msg = email.message_from_bytes(item.getvalue(), policy=policy.default)
                        body = msg.get_body(preferencelist=('plain'))
                        text = body.get_content() if body else str(msg)
                    else:
                        text = item.getvalue().decode("utf-8")
                if text.strip():
                    show_analysis_progress(index, len(items))
                    analysis = analyze_text(text)
                    if isinstance(item, str):
                        title = f"Paste Email: {name}"
                    else:
                        title = f"Upload File {upload_index}: {name}"
                    spam_probability, _, confidence, _, _ = analysis
                    try:
                        spam_probability_value = float(spam_probability.replace("%", ""))
                        confidence_value = float(confidence.replace("%", ""))
                    except ValueError:
                        spam_probability_value = 0.0
                        confidence_value = 0.0
                    batch_results.append({
                        "file": title,
                        "prediction": "Spam" if spam_probability_value >= 50 else "Authentic",
                        "spam_probability": spam_probability_value,
                        "confidence": confidence_value,
                    })
                    render_result(title, analysis)
                else:
                    st.warning(f"{name}: No readable text found.")
            except Exception as e:
                st.error(f"{name}: Error processing file: {e}")
        # Show the complete summary/dashboard/report for one or many items.
        if batch_results:
            render_batch_summary(batch_results)

# Footer
st.markdown("<br><p style='text-align: center; color: #64748b; font-size: 12px;'>🔒 Your data is processed locally using Scikit-Learn.</p>", unsafe_allow_html=True)
