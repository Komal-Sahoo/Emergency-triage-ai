import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import librosa
from audio_processor import load_and_preprocess_audio
from triage_engine import run_triage_pipeline

# Configure Page
st.set_page_config(
    page_title="Emergency Acoustic Triage",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Theme CSS
st.markdown("""
<style>
    /* Global Background Gradient */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #201335 0%, #0d0818 100%);
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Hide default Streamlit header artifacts */ 
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Glass Card */
    section[data-testid="stSidebar"] {
        background-color: rgba(22, 14, 38, 0.7) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Custom Glass Panel Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px 24px;
        box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Header Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        margin-bottom: 25px;
    }

    .badge-user {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.06);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
    }

    /* Priority Badges */
    .priority-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.1));
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 12px 18px;
        border-radius: 14px;
        text-align: center;
    }

    .priority-moderate {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.25), rgba(161, 98, 7, 0.1));
        border: 1px solid #eab308;
        color: #fde047;
        padding: 12px 18px;
        border-radius: 14px;
        text-align: center;
    }

    .priority-low {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.25), rgba(21, 128, 61, 0.1));
        border: 1px solid #22c55e;
        color: #86efac;
        padding: 12px 18px;
        border-radius: 14px;
        text-align: center;
    }

    .threat-tag {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 4px 4px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
<div class="top-nav">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 24px;">🚨</span>
        <div>
            <h3 style="margin: 0; font-size: 18px; font-weight: 700; color: #ffffff;">Acoustic Intelligence Console</h3>
            <p style="margin: 0; font-size: 11px; color: #94a3b8;">112/911 Multimodal Emergency Response & Triage</p>
        </div>
    </div>
    <div class="badge-user">
        <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #22c55e;"></div>
        <span>Dispatcher Unit #04</span>
        <span style="color: #64748b;">|</span>
        <span style="color: #94a3b8; font-size: 12px;">CAD Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Input
st.sidebar.markdown("### 🎙️ Audio Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload incoming 911/112 call recording", type=["wav", "mp3"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_audio_path = f"temp_upload{file_extension}"
    
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.audio(uploaded_file, format=f"audio/{file_extension.replace('.', '')}")
    run_btn = st.sidebar.button("⚡ Run Full Triage Analysis", use_container_width=True, type="primary")

    if run_btn:
        with st.spinner("Processing acoustic biomarkers & neural sound detection..."):
            audio_signal, sample_rate = load_and_preprocess_audio(temp_audio_path)
            results = run_triage_pipeline(temp_audio_path, audio_signal, sample_rate)

        # Top Row: 3 Modular Cards
        col1, col2, col3 = st.columns([1, 1.4, 1.2])

        # CARD 1: Priority & Vocal Distress
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase;'>Triage Evaluation</p>", unsafe_allow_html=True)
            
            score = results["triage_score"]
            if score >= 4:
                st.markdown(f'<div class="priority-critical"><h2 style="margin:0;">PRIORITY {score} / 5</h2><p style="margin:4px 0 0 0; font-size:12px; font-weight:700;">CRITICAL EMERGENCY</p></div>', unsafe_allow_html=True)
            elif score == 3:
                st.markdown(f'<div class="priority-moderate"><h2 style="margin:0;">PRIORITY {score} / 5</h2><p style="margin:4px 0 0 0; font-size:12px; font-weight:700;">MODERATE THREAT</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="priority-low"><h2 style="margin:0;">PRIORITY {score} / 5</h2><p style="margin:4px 0 0 0; font-size:12px; font-weight:700;">ROUTINE / NON-URGENT</p></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1;'><span>Vocal Distress Index</span><span style='font-weight:700;'>{results['stress_index']*100:.0f}%</span></div>", unsafe_allow_html=True)
            st.progress(results["stress_index"])
            st.markdown('</div>', unsafe_allow_html=True)

        # CARD 2: Waveform & Acoustic Energy Dynamics
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase;'>Acoustic Signal Amplitude</p>", unsafe_allow_html=True)
            
            # Generate downsampled waveform for Plotly
            time_axis = np.linspace(0, len(audio_signal) / sample_rate, num=min(300, len(audio_signal)))
            step = max(1, len(audio_signal) // 300)
            downsampled_signal = audio_signal[::step][:len(time_axis)]

            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(
                x=time_axis, 
                y=downsampled_signal, 
                mode='lines',
                line=dict(color='#a855f7', width=2),
                fill='tozeroy',
                fillcolor='rgba(168, 85, 247, 0.15)'
            ))
            fig_wave.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=5, b=5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            st.plotly_chart(fig_wave, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # CARD 3: Acoustic Threat Distribution Donut Chart
        with col3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase;'>Acoustic Threat Scene</p>", unsafe_allow_html=True)
            
            threats = results["flagged_threats"]
            if threats:
                labels = [t.split('(')[0].strip() for t in threats]
                values = [float(t.split('(')[1].replace('%)', '')) for t in threats]
                
                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.65,
                    marker=dict(colors=['#ef4444', '#f97316', '#a855f7', '#06b6d4']),
                    textinfo='none'
                )])
                fig_donut.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=5, b=5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(font=dict(size=10, color='#94a3b8'), orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("No elevated background threat signatures detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Bottom Row: Speech Transcript & Detected Hazards
        b_col1, b_col2 = st.columns([1.8, 1.2])

        with b_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase;'>Live Speech Transcription</p>", unsafe_allow_html=True)
            transcript_text = results["transcript"] if results["transcript"] else "No intelligible speech detected in recording."
            st.markdown(f"<div style='background: rgba(0,0,0,0.25); padding: 14px; border-radius: 10px; border-left: 3px solid #a855f7; font-size: 14px; line-height: 1.6;'>\"{transcript_text}\"</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with b_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase;'>Detected Hazard Tags</p>", unsafe_allow_html=True)
            if threats:
                tags_html = "".join([f'<span class="threat-tag">⚠️ {t.upper()}</span>' for t in threats])
                st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 13px; color: #64748b;'>Ambient acoustic conditions within normal limits.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

else:
    # Empty State Dashboard Placeholder
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 48px 20px;">
        <div style="font-size: 42px; margin-bottom: 12px;">🎧</div>
        <h4 style="margin: 0; color: #f8fafc;">Waiting for Distress Call Ingestion</h4>
        <p style="margin: 6px 0 0 0; font-size: 13px; color: #94a3b8;">Upload an incoming audio stream or `.wav` recording from the left sidebar to start real-time acoustic scene triage.</p>
    </div>
    """, unsafe_allow_html=True)