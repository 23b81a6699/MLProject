import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import base64
import os
import plotly.express as px  # New import for the pie chart

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Music Classifier", page_icon="🎵", layout="wide")

# Helper function to load local image as background
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# Load background image
bin_str = get_base64_of_bin_file('backimg1.jpg') 
if bin_str:
    bg_img_style = f"url('data:image/jpg;base64,{bin_str}')"
else:
    bg_img_style = "linear-gradient(#f0f2f6, #e1e4e8)"

# --- 2. Custom CSS ---
st.markdown(f"""
<style>
    .stApp {{
        background: {bg_img_style};
        background-size: cover;
        background-attachment: fixed;
    }}

    .header-box {{
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 40px;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 20px auto 40px auto;
        max-width: 900px;
    }}

    .feature-card {{
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.3s ease;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }}
    .feature-card:hover {{ transform: translateY(-10px); background: rgba(255, 255, 255, 0.9); }}
    .feature-emoji {{ font-size: 40px; margin-bottom: 10px; }}
    .feature-label {{ color: #666; font-size: 14px; text-transform: uppercase; font-weight: 700; }}
    .feature-value {{ color: #000; font-size: 28px; font-weight: bold; }}

    h1, h2, h3, p, span {{ color: #222 !important; }}
    
    /* Center the chart */
    .chart-container {{
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        border-radius: 25px;
        padding: 20px;
        margin: 20px auto;
        max-width: 800px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }}
</style>

<div class="cd-container">
    <div class="rainbow-cd"></div>
</div>
""", unsafe_allow_html=True)

# --- 3. Logic ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/mood_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    mapping = joblib.load("models/mapping.pkl")
    return model, scaler, mapping

import io

def extract_features(audio_file):
    # Read the file into a buffer so we can load it twice if needed
    input_bytes = audio_file.read()
    
    # 1. First, load the full file (or just the header) to check duration
    with io.BytesIO(input_bytes) as b:
        # sr=None is faster as it avoids resampling just to check length
        y_temp, sr_temp = librosa.load(b, sr=None)
        duration = librosa.get_duration(y=y_temp, sr=sr_temp)

    # 2. Decide on offset: 
    # If song > 35s, start at 30s. Otherwise, start at 0s.
    start_offset = 30 if duration > 35 else 0
    
    # 3. Load the actual segment for analysis
    with io.BytesIO(input_bytes) as b:
        y, sr = librosa.load(b, duration=45, offset=start_offset)
    
    # --- Feature Extraction Logic ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = np.mean(librosa.feature.rms(y=y)) * 5 
    loudness = np.mean(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max))
    danceability = np.mean(librosa.feature.spectral_flatness(y=y)) * 12
    valence = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) / 5000 

    return {
        "danceability": min(float(np.array(danceability).item()), 1.0),
        "energy": min(float(np.array(energy).item()), 1.0),
        "loudness": float(np.array(loudness).item()),
        "valence": min(float(np.array(valence).item()), 1.0),
        "tempo": float(np.array(tempo).item())
    }

# --- 4. UI ---
st.markdown('<div class="header-box"><h1>🎵 AI Movie Song Classifier</h1><p>Technical Audio DNA & Mood Analysis</p></div>', unsafe_allow_html=True)

try:
    model, scaler, mapping = load_assets()
    uploaded_file = st.file_uploader("Upload your song here", type=['mp3', 'wav', 'm4a'])

    if uploaded_file:
        st.audio(uploaded_file)
        
        with st.spinner("🎧 Analyzing Sonic Patterns..."):
            data = extract_features(uploaded_file)
            input_scaled = scaler.transform(pd.DataFrame([data]))
            mood = mapping[model.predict(input_scaled)[0]]

            # A. Audio DNA Cards
            st.divider()
            st.subheader("📊 Extracted Audio DNA")
            
            icons = {"Danceability": "💃", "Energy": "⚡", "Loudness": "🔊", "Valence": "🌈", "Tempo": "🥁"}
            features_list = [
                ("Danceability", round(data['danceability'], 2)),
                ("Energy", round(data['energy'], 2)),
                ("Loudness", f"{round(data['loudness'], 1)} dB"),
                ("Valence", round(data['valence'], 2)),
                ("Tempo", f"{int(data['tempo'])} BPM")
            ]

            cols = st.columns(5)
            for i, (name, val) in enumerate(features_list):
                with cols[i]:
                    st.markdown(f'<div class="feature-card"><div class="feature-emoji">{icons[name]}</div><div class="feature-label">{name}</div><div class="feature-value">{val}</div></div>', unsafe_allow_html=True)
 
            # B. Colorful Pie Chart (The requested part)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.write("### 🧬 Feature Distribution")
            
            # Prepare data for Pie Chart (excluding Tempo and Loudness as they aren't 0-1 metrics)
            plot_data = {
                "Feature": ["Danceability", "Energy", "Valence"],
                "Value": [data['danceability'], data['energy'], data['valence']]
            }
            df_plot = pd.DataFrame(plot_data)
            
            fig = px.pie(
                df_plot, 
                values='Value', 
                names='Feature', 
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # C. Result Banner
            mood_styles = {"Energetic": ("#FF4B2B", "🔥"), "Happy": ("#FDC830", "😊"), "Sad": ("#00416A", "😢"), "Calm": ("#2ECC71", "🍃")}
            bg_color, emoji = mood_styles.get(mood, ("#999", "🎶"))
            
            st.markdown(f'<div style="background-color: {bg_color}; padding: 35px; border-radius: 25px; text-align: center; color: white; font-size: 40px; font-weight: bold; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">{emoji} Predicted Mood : {mood} {emoji}</div>', unsafe_allow_html=True)

except FileNotFoundError:
    st.error("🚨 Model files missing! Run 'python train.py' first.")