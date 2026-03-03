import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import base64
import os
import plotly.express as px

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
bin_str = get_base64_of_bin_file('backimg.jpg') 
bg_img_style = f"url('data:image/jpg;base64,{bin_str}')" if bin_str else "linear-gradient(#f0f2f6, #e1e4e8)"

# --- 2. Custom CSS (Unified Glass Container & CD Animation) ---
st.markdown(f"""
<style>
    .stApp {{
        background: {bg_img_style} no-repeat center center fixed;
        background-size: cover;
    }}

    /* Global Glass Wrapper - Forces all content to stay on top */
    .main-glass-wrapper {{
        background: rgba(255, 255, 255, 0.4); 
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 40px;
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
        max-width: 1000px;
        position: relative; 
        z-index: 10;         
    }}

    /* Rotating CD - Background Layer */
    @keyframes rotate_clockwise {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}

    .cd-container {{
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 0; 
        opacity: 0.3;
        pointer-events: none; 
    }}

    .rainbow-cd {{
        width: 600px;
        height: 600px;
        background-image: url('https://e7.pngegg.com/pngimages/145/233/png-clipart-cd-and-dvd-compact-disc-dvd-rom-computer-icons-cd-miscellaneous-blue.png');
        background-size: contain;
        background-repeat: no-repeat;
        animation: rotate_clockwise 20s linear infinite;
    }}

    .feature-card {{
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: transform 0.3s ease;
    }}
    .feature-card:hover {{ transform: translateY(-5px); }}

    h1, h2, h3, p {{ color: #1a1a1a !important; font-weight: 700 !important; }}
    
    .chart-container {{
        background: rgba(255, 255, 255, 0.6);
        border-radius: 25px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
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

def extract_features(audio_file):
    # Safety: Load full duration first to prevent seek errors on short files
    y_full, sr_native = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y_full, sr=sr_native)
    
    # Adaptive offset: start at 30s only if file is long enough
    offset_val = 30 if duration > 35 else 0
    y, sr = librosa.load(audio_file, duration=45, offset=offset_val)
    
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

# --- 4. Main UI ---
st.markdown('<div class="main-glass-wrapper">', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;"><h1>🎵 AI Movie Song Classifier</h1><p>Technical Audio DNA & Mood Analysis</p></div>', unsafe_allow_html=True)

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
                    st.markdown(f"""
                        <div class="feature-card">
                            <div style="font-size:35px;">{icons[name]}</div>
                            <div style="color:#666; font-size:12px; font-weight:700;">{name}</div>
                            <div style="font-size:22px; font-weight:bold;">{val}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # B. Colorful Pie Chart Glass Box
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.write("### 🧬 Feature Distribution")
            
            df_plot = pd.DataFrame({
                "Feature": ["Danceability", "Energy", "Valence"],
                "Value": [data['danceability'], data['energy'], data['valence']]
            })
            
            fig = px.pie(
                df_plot, values='Value', names='Feature', hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # C. Result Banner
            mood_styles = {"Energetic": ("#FF4B2B", "🔥"), "Happy": ("#FDC830", "😊"), "Sad": ("#00416A", "😢"), "Calm": ("#2ECC71", "🍃")}
            bg_color, emoji = mood_styles.get(mood, ("#999", "🎶"))
            
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 30px; border-radius: 20px; 
                text-align: center; color: white; font-size: 35px; font-weight: bold; 
                margin-top:20px; box-shadow: 0 10px 20px rgba(0,0,0,0.15);">
                    {emoji} Predicted Mood : {mood} {emoji}
                </div>
            """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("🚨 Model files missing! Run 'python train.py' first.")
except Exception as e:
    st.error(f"An error occurred: {e}")

st.markdown('</div>', unsafe_allow_html=True)