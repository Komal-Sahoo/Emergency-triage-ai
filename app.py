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

# Light Mode Aesthetic & Custom Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..800;1,6..72,400..800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Newsreader', Georgia, serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    header { background: transparent !important; }
    div[data-testid="stDecoration"] { display: none; }
    footer { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.03);
    }

    .sidebar-header {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 25px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-top: 10px;
        margin-bottom: 4px;
        line-height: 1.2;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        margin-bottom: 6px;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, #F0F9FF 0%, #E0F2FE 100%) !important;
        border: 1.5px dashed #0284C7 !important;
        border-radius: 14px !important;
        padding: 16px 12px !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] div {
        color: #0369A1 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 6px 14px !important;
    }

    section[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.3) !important;
    }

    section[data-testid="stSidebar"] button[kind="primary"] p {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }

    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 28px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    }

    .main-title {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 30px !important;
        font-weight: 700 !important;
        line-height: 1.15;
        color: #0F172A;
        margin: 0;
    }

    .sub-title {
        font-family: 'Newsreader', serif !important;
        font-size: 15px !important;
        color: #64748B;
        margin-top: 4px;
    }

    .badge-user {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 8px 16px;
        border-radius: 30px;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 13px;
        font-weight: 600;
        color: #334155;
    }

    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
    }

    .card-label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }

    .p-badge {
        padding: 16px 20px;
        border-radius: 14px;
        text-align: center;
        color: #FFFFFF !important;
    }
    .p-badge h2 {
        font-family: 'Newsreader', serif !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: #FFFFFF !important;
    }
    .p-badge p {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        margin: 4px 0 0 0 !important;
        color: #FFFFFF !important;
    }

    .priority-1 { background-color: #2ECC71; box-shadow: 0 6px 18px rgba(46, 204, 113, 0.35); }
    .priority-2 { background-color: #F1C40F; box-shadow: 0 6px 18px rgba(241, 196, 15, 0.35); }
    .priority-3 { background-color: #E67E22; box-shadow: 0 6px 18px rgba(230, 126, 34, 0.35); }
    .priority-4 { background-color: #E74C3C; box-shadow: 0 6px 18px rgba(231, 76, 60, 0.35); }
    .priority-5 { background-color: #C0392B; box-shadow: 0 6px 18px rgba(192, 57, 43, 0.4); }

    .threat-tag {
        display: inline-block;
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        padding: 6px 12px;
        border-radius: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 4px 4px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
<div class="top-nav">
    <div style="display: flex; align-items: center; gap: 14px;">
        <span style="font-size: 32px;">🚨</span>
        <div>
            <h1 class="main-title">Acoustic Intelligence Console</h1>
            <p class="sub-title">Multimodal Emergency Response & Triage</p>
        </div>
    </div>
    <div class="badge-user">
        <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #2ECC71;"></div>
        <span>Dispatcher Unit #04</span>
        <span style="color: #CBD5E1;">|</span>
        <span style="color: #64748B;">CAD Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Input
st.sidebar.markdown('<p class="sidebar-header">🎙️ Audio Ingestion</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload incoming call recording", type=["wav", "mp3"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_audio_path = f"temp_upload{file_extension}"
    
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.audio(uploaded_file, format=f"audio/{file_extension.replace('.', '')}")
    run_btn = st.sidebar.button("⚡ Run Full Triage Analysis", use_container_width=True, type="primary")

    if run_btn:
        with st.spinner("Processing acoustic biomarkers, sliding-window threat classification & translation..."):
            audio_signal, sample_rate = load_and_preprocess_audio(temp_audio_path)
            results = run_triage_pipeline(temp_audio_path, audio_signal, sample_rate)

        # 3 Top Metric Cards
        col1, col2, col3 = st.columns([1, 1.4, 1.2])

        # CARD 1: Priority Badge
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="card-label">Triage Evaluation</p>', unsafe_allow_html=True)
            
            score = int(np.clip(results.get("triage_score", 1), 1, 5))
            badge_lookup = {
                1: ('priority-1', 'SAFE / NORMAL', 'LOW RISK & STABLE'),
                2: ('priority-2', 'ADVISORY ALERT', 'MILD WARNING / WATCH'),
                3: ('priority-3', 'WATCH ALERT', 'MODERATE INCIDENT'),
                4: ('priority-4', 'SEVERE ALERT', 'HIGH DANGER / THREAT'),
                5: ('priority-5', 'CRITICAL EMERGENCY', 'LIFE SAFETY ALERT')
            }
            css_class, title, subtitle = badge_lookup[score]
            
            st.markdown(f'''
                <div class="p-badge {css_class}">
                    <h2>PRIORITY {score} / 5</h2>
                    <p>{title} — {subtitle}</p>
                </div>
            ''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='display:flex; justify-content:space-between; font-family:Plus Jakarta Sans, sans-serif; font-size:13px; font-weight:600; color:#475569;'><span>Vocal Distress Index</span><span>{results['stress_index']*100:.0f}%</span></div>", unsafe_allow_html=True)
            st.progress(results["stress_index"])
            st.markdown('</div>', unsafe_allow_html=True)

        # CARD 2: Waveform Dynamics
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="card-label">Acoustic Signal Amplitude</p>', unsafe_allow_html=True)
            
            time_axis = np.linspace(0, len(audio_signal) / sample_rate, num=min(300, len(audio_signal)))
            step = max(1, len(audio_signal) // 300)
            downsampled_signal = audio_signal[::step][:len(time_axis)]

            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(
                x=time_axis, 
                y=downsampled_signal, 
                mode='lines',
                line=dict(color='#E67E22', width=2),
                fill='tozeroy',
                fillcolor='rgba(230, 126, 34, 0.12)'
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

        # CARD 3: Acoustic Threat Donut Chart
        with col3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="card-label">Acoustic Threat Scene</p>', unsafe_allow_html=True)
            
            threats = results["flagged_threats"]
            if threats:
                labels = []
                values = []
                for t in threats:
                    try:
                        # Split safely from the rightmost parenthesis
                        label_part, _, val_part = t.rpartition('(')
                        val_num = float(val_part.replace('%)', '').strip())
                        labels.append(label_part.strip())
                        values.append(val_num)
                    except Exception:
                        continue
                
                if values:
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=.65,
                        marker=dict(colors=['#C0392B', '#E74C3C', '#E67E22', '#F1C40F', '#0284C7']),
                        textinfo='none'
                    )])
                    fig_donut.update_layout(
                        height=150,
                        margin=dict(l=0, r=0, t=5, b=5),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=True,
                        legend=dict(font=dict(size=10, color='#475569', family='Plus Jakarta Sans'), orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.info("No elevated background threat signatures detected.")
            else:
                st.info("No elevated background threat signatures detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Bottom Row: Native Transcription vs English Translation
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="card-label">Original Spoken Audio ({results["language"]})</p>', unsafe_allow_html=True)
            native_text = results["native_transcript"] if results["native_transcript"] else "No intelligible speech detected."
            st.markdown(f"<div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 12px; border-left: 4px solid #0284C7; font-family: Newsreader, serif; font-size: 16px; line-height: 1.6; max-height: 180px; overflow-y: auto;'>\"{native_text}\"</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with t_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="card-label">Real-Time English Translation</p>', unsafe_allow_html=True)
            english_text = results["english_transcript"] if results["english_transcript"] else "Translation unavailable."
            st.markdown(f"<div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 12px; border-left: 4px solid #E67E22; font-family: Newsreader, serif; font-size: 16px; line-height: 1.6; max-height: 180px; overflow-y: auto;'>\"{english_text}\"</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Hazard Tags Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-label">Detected Hazard Tags (Across 5-Min Timeline)</p>', unsafe_allow_html=True)
        if threats:
            tags_html = "".join([f'<span class="threat-tag">⚠️ {t.upper()}</span>' for t in threats])
            st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-family: Plus Jakarta Sans, sans-serif; font-size: 13px; color: #64748B;'>Ambient acoustic conditions within normal limits.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)