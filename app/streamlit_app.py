"""
KNSD Learn - Premium UI Edition
A modern, polished Kannada numeral learning experience
"""
import streamlit as st
import os
import sys
import cv2
import numpy as np
import joblib
import random
from PIL import Image
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from knsd_features import (preprocess_image, extract_knsd_features, 
                           extract_loop_features, extract_endpoint_features, 
                           extract_junction_features, extract_stroke_statistics,
                           IMAGE_SIZE)
from reference_features import compute_quality_score, get_digit_description

KANNADA_NUMERALS = ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯']

# ============================================================================
# DESIGN SYSTEM
# ============================================================================

DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    /* Color Palette - Premium Dark Mode */
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-elevated: #1a1a24;
    --bg-card: #1e1e2a;
    
    --surface-dim: rgba(255, 255, 255, 0.03);
    --surface-subtle: rgba(255, 255, 255, 0.06);
    --surface-medium: rgba(255, 255, 255, 0.1);
    
    --text-primary: #ffffff;
    --text-secondary: #a0a0b0;
    --text-muted: #6b6b78;
    
    --accent-primary: #6366f1;        /* Electric Indigo */
    --accent-secondary: #8b5cf6;      /* Purple */
    --accent-glow: rgba(99, 102, 241, 0.4);
    
    --success: #22c55e;
    --success-glow: rgba(34, 197, 94, 0.3);
    --error: #ef4444;
    --error-glow: rgba(239, 68, 68, 0.3);
    --warning: #f59e0b;
    
    --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    --gradient-success: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    --gradient-canvas: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    
    /* Spacing Scale */
    --space-xs: 4px;
    --space-sm: 8px;
    --space-md: 16px;
    --space-lg: 24px;
    --space-xl: 32px;
    --space-2xl: 48px;
    
    /* Border Radius */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --radius-full: 9999px;
    
    /* Shadows */
    --shadow-glow: 0 0 40px var(--accent-glow);
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4);
    --shadow-elevated: 0 8px 32px rgba(0, 0, 0, 0.6);
}

/* Global Overrides */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.main .block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* ==================== TYPOGRAPHY ==================== */

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: var(--text-secondary);
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: var(--space-md);
}

/* ==================== TARGET DIGIT DISPLAY ==================== */

.target-card {
    background: var(--bg-card);
    border: 1px solid var(--surface-medium);
    border-radius: var(--radius-xl);
    padding: var(--space-xl);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.target-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-primary);
}

.target-label {
    color: var(--text-muted);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: var(--space-sm);
}

.target-digit {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 8rem;
    line-height: 1;
    color: var(--text-primary);
    margin: var(--space-md) 0;
    text-shadow: 0 0 60px var(--accent-glow);
    animation: digit-pulse 2s ease-in-out infinite;
}

@keyframes digit-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.9; transform: scale(1.02); }
}

.target-description {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: var(--space-md);
}

/* ==================== CANVAS HERO ==================== */

.canvas-container {
    background: var(--gradient-canvas);
    border: 2px solid var(--surface-medium);
    border-radius: var(--radius-xl);
    padding: var(--space-lg);
    position: relative;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-card);
}

.canvas-container:hover {
    border-color: var(--accent-primary);
    box-shadow: var(--shadow-glow);
}

.canvas-label {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: var(--space-md);
}

.canvas-dot {
    width: 8px;
    height: 8px;
    background: var(--accent-primary);
    border-radius: 50%;
    animation: dot-pulse 1.5s ease-in-out infinite;
}

@keyframes dot-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--accent-glow); }
    50% { opacity: 0.8; box-shadow: 0 0 0 8px transparent; }
}

/* ==================== BUTTONS ==================== */

.btn-primary {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-md) var(--space-xl) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    cursor: pointer;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--accent-glow) !important;
}

.btn-secondary {
    background: var(--surface-subtle) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--surface-medium) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-sm) var(--space-lg) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.btn-secondary:hover {
    background: var(--surface-medium) !important;
    border-color: var(--accent-primary) !important;
}

/* ==================== RESULT STATES ==================== */

.result-card {
    border-radius: var(--radius-xl);
    padding: var(--space-xl);
    text-align: center;
    animation: result-appear 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes result-appear {
    0% { opacity: 0; transform: scale(0.9) translateY(20px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.result-success {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.1) 100%);
    border: 2px solid var(--success);
    box-shadow: 0 0 40px var(--success-glow);
}

.result-error {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%);
    border: 2px solid var(--error);
    box-shadow: 0 0 40px var(--error-glow);
}

.result-icon {
    font-size: 4rem;
    margin-bottom: var(--space-md);
    animation: icon-bounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

@keyframes icon-bounce {
    0% { transform: scale(0); }
    60% { transform: scale(1.2); }
    100% { transform: scale(1); }
}

.result-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: var(--space-sm);
}

.result-success .result-text { color: var(--success); }
.result-error .result-text { color: var(--error); }

.result-subtext {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

/* ==================== STATS & PROGRESS ==================== */

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--space-md);
    margin-bottom: var(--space-xl);
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--surface-subtle);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    text-align: center;
    transition: all 0.2s ease;
}

.stat-card:hover {
    border-color: var(--accent-primary);
    transform: translateY(-2px);
}

.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}

.stat-label {
    color: var(--text-muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: var(--space-xs);
}

/* Progress Ring */
.progress-ring {
    position: relative;
    width: 120px;
    height: 120px;
    margin: 0 auto var(--space-lg);
}

.progress-ring svg {
    transform: rotate(-90deg);
}

.progress-ring-bg {
    fill: none;
    stroke: var(--surface-subtle);
    stroke-width: 8;
}

.progress-ring-fill {
    fill: none;
    stroke: url(#gradient);
    stroke-width: 8;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.5s ease;
}

.progress-ring-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}

.progress-ring-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}

.progress-ring-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
}

/* Streak Counter */
.streak-badge {
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
    padding: var(--space-sm) var(--space-md);
    border-radius: var(--radius-full);
    font-weight: 600;
    font-size: 0.9rem;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    animation: streak-glow 2s ease-in-out infinite;
}

@keyframes streak-glow {
    0%, 100% { box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3); }
    50% { box-shadow: 0 4px 24px rgba(245, 158, 11, 0.5); }
}

/* ==================== FEEDBACK ==================== */

.feedback-container {
    background: var(--bg-card);
    border: 1px solid var(--surface-subtle);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    margin-top: var(--space-lg);
}

.feedback-item {
    display: flex;
    align-items: flex-start;
    gap: var(--space-md);
    padding: var(--space-md) 0;
    border-bottom: 1px solid var(--surface-subtle);
    animation: feedback-slide 0.4s ease;
}

.feedback-item:last-child {
    border-bottom: none;
}

@keyframes feedback-slide {
    0% { opacity: 0; transform: translateX(-10px); }
    100% { opacity: 1; transform: translateX(0); }
}

.feedback-icon {
    font-size: 1.2rem;
}

.feedback-text {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ==================== QUALITY METER ==================== */

.quality-meter {
    margin: var(--space-lg) 0;
}

.quality-bar-bg {
    height: 8px;
    background: var(--surface-subtle);
    border-radius: var(--radius-full);
    overflow: hidden;
}

.quality-bar-fill {
    height: 100%;
    border-radius: var(--radius-full);
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.quality-label {
    display: flex;
    justify-content: space-between;
    margin-top: var(--space-sm);
    font-size: 0.85rem;
}

.quality-label span:first-child {
    color: var(--text-secondary);
}

.quality-label span:last-child {
    font-weight: 600;
}

/* ==================== SIDEBAR ==================== */

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--surface-subtle) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: var(--space-xl) var(--space-lg) !important;
}

/* ==================== CANVAS TOOLBAR ==================== */

/* Force ALL buttons near canvas to have white icons */
[data-testid="stDrawableCanvas"] ~ div button,
[data-testid="stDrawableCanvas"] + div button,
[data-testid="stDrawableCanvas"] button,
.drawable-canvas-toolbar button,
div[class*="drawable"] button,
div[class*="canvas"] button,
iframe + div button {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stDrawableCanvas"] ~ div button:hover,
[data-testid="stDrawableCanvas"] + div button:hover,
[data-testid="stDrawableCanvas"] button:hover,
.drawable-canvas-toolbar button:hover,
div[class*="drawable"] button:hover,
div[class*="canvas"] button:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
}

/* Force ALL SVGs near canvas to be white - ultra aggressive */
[data-testid="stDrawableCanvas"] ~ div svg,
[data-testid="stDrawableCanvas"] ~ div svg *,
[data-testid="stDrawableCanvas"] + div svg,
[data-testid="stDrawableCanvas"] + div svg *,
[data-testid="stDrawableCanvas"] svg,
[data-testid="stDrawableCanvas"] svg *,
.drawable-canvas-toolbar svg,
.drawable-canvas-toolbar svg *,
div[class*="drawable"] svg,
div[class*="drawable"] svg *,
div[class*="canvas"] svg,
div[class*="canvas"] svg *,
iframe + div svg,
iframe + div svg * {
    fill: #ffffff !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
}

/* Target specific icon classes that might exist */
[class*="icon"],
[class*="Icon"] {
    color: #ffffff !important;
}

button svg, button svg * {
    fill: currentColor !important;
}

/* ==================== TABS ==================== */

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    padding: var(--space-xs);
    gap: var(--space-xs) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    padding: var(--space-sm) var(--space-lg) !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-primary) !important;
    color: white !important;
}

/* ==================== ANIMATIONS ==================== */

@keyframes confetti {
    0% { transform: translateY(0) rotate(0); opacity: 1; }
    100% { transform: translateY(-100px) rotate(720deg); opacity: 0; }
}

.confetti-piece {
    position: fixed;
    width: 10px;
    height: 10px;
    background: var(--accent-primary);
    animation: confetti 1s ease-out forwards;
}

/* Loading State */
.loading-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--surface-subtle);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: var(--space-xl) auto;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ==================== MISC ==================== */

.divider {
    height: 1px;
    background: var(--surface-subtle);
    margin: var(--space-xl) 0;
}

.badge {
    display: inline-block;
    padding: var(--space-xs) var(--space-sm);
    background: var(--surface-subtle);
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-accent {
    background: var(--accent-primary);
    color: white;
}
</style>
"""

# ============================================================================
# APP CONFIG
# ============================================================================

st.set_page_config(
    page_title="KNSD Learn", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'knsd_model.pkl')
    if not os.path.exists(model_path):
        return None, None
    data = joblib.load(model_path)
    return data['model'], data['scaler']


def preprocess_canvas(canvas_data):
    """Process canvas drawing to binary image."""
    if canvas_data is None:
        return None
    
    if len(canvas_data.shape) == 3:
        if canvas_data.shape[2] == 4:
            rgb = canvas_data[:, :, :3].astype(np.uint8)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(canvas_data.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas_data.astype(np.uint8)
    
    inverted = 255 - gray
    
    if np.max(inverted) < 50:
        return None
    
    _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    resized = cv2.resize(binary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    _, final = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    return final


def predict_canvas(canvas_img, model, scaler):
    """Predict from canvas drawing."""
    processed = preprocess_canvas(canvas_img)
    if processed is None:
        return None, 0, None, {}
    
    features = extract_knsd_features(processed)
    features = np.nan_to_num(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    
    loop_feats = extract_loop_features(processed)
    endpoint_feats = extract_endpoint_features(processed)
    junction_feats = extract_junction_features(processed)
    stroke_feats = extract_stroke_statistics(processed)
    
    return pred, proba[pred] * 100, processed, {
        'loops': int(loop_feats[0]),
        'endpoints': int(endpoint_feats[0]),
        'junctions': int(junction_feats[0]),
        'aspect_ratio': float(stroke_feats[0]),
        'circularity': float(stroke_feats[4])
    }


def render_progress_ring(percentage, label="Accuracy"):
    """Render SVG progress ring."""
    circumference = 2 * 3.14159 * 45
    offset = circumference - (percentage / 100) * circumference
    
    return f'''
    <div class="progress-ring">
        <svg width="120" height="120" viewBox="0 0 120 120">
            <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#6366f1"/>
                    <stop offset="100%" style="stop-color:#8b5cf6"/>
                </linearGradient>
            </defs>
            <circle class="progress-ring-bg" cx="60" cy="60" r="45"/>
            <circle class="progress-ring-fill" cx="60" cy="60" r="45" 
                    stroke-dasharray="{circumference}" 
                    stroke-dashoffset="{offset}"/>
        </svg>
        <div class="progress-ring-text">
            <div class="progress-ring-value">{percentage:.0f}%</div>
            <div class="progress-ring-label">{label}</div>
        </div>
    </div>
    '''


def render_quality_meter(score):
    """Render quality score meter."""
    if score >= 75:
        color = "linear-gradient(90deg, #22c55e 0%, #16a34a 100%)"
        label_color = "#22c55e"
    elif score >= 50:
        color = "linear-gradient(90deg, #f59e0b 0%, #d97706 100%)"
        label_color = "#f59e0b"
    else:
        color = "linear-gradient(90deg, #ef4444 0%, #dc2626 100%)"
        label_color = "#ef4444"
    
    return f'''
    <div class="quality-meter">
        <div class="quality-bar-bg">
            <div class="quality-bar-fill" style="width: {score}%; background: {color};"></div>
        </div>
        <div class="quality-label">
            <span>Quality Score</span>
            <span style="color: {label_color};">{score}/100</span>
        </div>
    </div>
    '''


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    model, scaler = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Run `python experiments/train_evaluate.py` first.")
        return
    
    # Initialize session state
    if 'target_digit' not in st.session_state:
        st.session_state.target_digit = random.randint(0, 9)
    if 'correct_count' not in st.session_state:
        st.session_state.correct_count = 0
    if 'total_attempts' not in st.session_state:
        st.session_state.total_attempts = 0
    if 'streak' not in st.session_state:
        st.session_state.streak = 0
    if 'best_streak' not in st.session_state:
        st.session_state.best_streak = 0
    
    # ========== HEADER ==========
    st.markdown('<h1 class="hero-title">KNSD Learn</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Master Kannada numerals with AI-powered feedback</p>', unsafe_allow_html=True)
    
    # ========== STATS BAR ==========
    accuracy = (st.session_state.correct_count / st.session_state.total_attempts * 100) if st.session_state.total_attempts > 0 else 0
    
    st.markdown(f'''
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{st.session_state.correct_count}</div>
            <div class="stat-label">Correct</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{st.session_state.total_attempts}</div>
            <div class="stat-label">Attempts</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{accuracy:.0f}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Streak badge
    if st.session_state.streak >= 2:
        st.markdown(f'''
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span class="streak-badge">🔥 {st.session_state.streak} Streak!</span>
        </div>
        ''', unsafe_allow_html=True)
    
    # ========== MAIN CONTENT ==========
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        target = st.session_state.target_digit
        st.markdown(f'''
        <div class="target-card">
            <div class="target-label">Draw This</div>
            <div class="target-digit">{KANNADA_NUMERALS[target]}</div>
            <div class="target-description">{get_digit_description(target)}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎲 New Digit", use_container_width=True):
            st.session_state.target_digit = random.randint(0, 9)
            st.rerun()
    
    with col2:
        st.markdown('''
        <div class="canvas-label">
            <span class="canvas-dot"></span>
            <span>Your Drawing</span>
        </div>
        ''', unsafe_allow_html=True)
        
        try:
            from streamlit_drawable_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=12,
                stroke_color="#1a1a2e",
                background_color="#ffffff",
                height=320,
                width=320,
                drawing_mode="freedraw",
                key="canvas",
            )
        except ImportError:
            st.error("📦 Install: `pip install streamlit-drawable-canvas`")
            return
    
    # ========== CHECK BUTTON ==========
    st.markdown("<br>", unsafe_allow_html=True)
    
    check_col1, check_col2, check_col3 = st.columns([1, 2, 1])
    with check_col2:
        check_pressed = st.button("✨ Check My Drawing", use_container_width=True, type="primary")
    
    # ========== RESULTS ==========
    if check_pressed:
        if canvas_result.image_data is not None:
            pred, conf, processed, feats = predict_canvas(canvas_result.image_data, model, scaler)
            
            if pred is not None:
                st.session_state.total_attempts += 1
                target = st.session_state.target_digit
                is_correct = pred == target
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Result display
                if is_correct:
                    st.session_state.correct_count += 1
                    st.session_state.streak += 1
                    if st.session_state.streak > st.session_state.best_streak:
                        st.session_state.best_streak = st.session_state.streak
                    
                    st.markdown(f'''
                    <div class="result-card result-success">
                        <div class="result-icon">🎉</div>
                        <div class="result-text">Perfect!</div>
                        <div class="result-subtext">That's exactly right. Keep going!</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.session_state.streak = 0
                    st.markdown(f'''
                    <div class="result-card result-error">
                        <div class="result-icon">💪</div>
                        <div class="result-text">Almost!</div>
                        <div class="result-subtext">You drew {KANNADA_NUMERALS[pred]} ({pred}), but we wanted {KANNADA_NUMERALS[target]} ({target})</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Quality feedback
                score, feedback = compute_quality_score(target, feats)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(render_quality_meter(score), unsafe_allow_html=True)
                
                # Feedback items
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                for msg in feedback:
                    emoji = "✅" if "✅" in msg or "correct" in msg.lower() else "⚠️" if "⚠️" in msg else "💡"
                    clean_msg = msg.replace("✅", "").replace("⚠️", "").replace("❌", "").strip()
                    st.markdown(f'''
                    <div class="feedback-item">
                        <span class="feedback-icon">{emoji}</span>
                        <span class="feedback-text">{clean_msg}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature breakdown
                with st.expander("🔬 Technical Details"):
                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.metric("Loops", feats['loops'])
                    fc2.metric("Endpoints", feats['endpoints'])
                    fc3.metric("Junctions", feats['junctions'])
                    fc4.metric("Circularity", f"{feats['circularity']:.2f}")
                    
                    if processed is not None:
                        st.image(processed, caption="Processed Input", width=150)
            else:
                st.warning("✏️ Please draw something first!")
        else:
            st.warning("✏️ Please draw something first!")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("### 📊 Session Stats")
        st.markdown(render_progress_ring(accuracy, "Accuracy"), unsafe_allow_html=True)
        
        st.markdown(f'''
        <div style="text-align: center; margin: 1rem 0;">
            <div style="color: var(--text-muted); font-size: 0.85rem;">Best Streak</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">🔥 {st.session_state.best_streak}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔄 Reset Progress", use_container_width=True):
            st.session_state.correct_count = 0
            st.session_state.total_attempts = 0
            st.session_state.streak = 0
            st.rerun()


if __name__ == '__main__':
    main()
