import numpy as np
import librosa
import whisper
from transformers import pipeline
import streamlit as st

@st.cache_resource
def get_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def get_audio_classifier():
    return pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")

# Broad category keywords for AudioSet labels
CRITICAL_THREAT_KEYWORDS = [
    "gun", "gunshot", "gunfire", "explosion", "blast", "artillery",
    "scream", "screaming", "shout", "yell", "wail", 
    "fire alarm", "smoke detector", "alarm", "buzzer", "siren"
]

MODERATE_THREAT_KEYWORDS = [
    "glass", "shatter", "crash", "smash", "traffic", "vehicle",
    "crying", "sob", "whimper", "groan", "bang"
]

# Lexical trigger words for emergency dispatch
EMERGENCY_WORDS = {
    "help", "emergency", "fire", "bleeding", "shot", "gun", "shooting", 
    "dying", "attacker", "intruder", "crash", "knife", "smoke", "burning", "hurry"
}
NON_EMERGENCY_WORDS = {
    "lost", "inquiry", "routine", "minor", "water leak", "parking", 
    "wallet", "no emergency", "wrong number", "information"
}

def transcribe_audio(audio: np.ndarray) -> str:
    """Transcribes spoken dialogue directly from the normalized NumPy audio array."""
    model = get_whisper_model()
    audio_float32 = audio.astype(np.float32)
    result = model.transcribe(audio_float32, fp16=False)
    return result.get("text", "").strip()

def detect_threat_events(audio: np.ndarray, sr: int = 16000, top_k: int = 10) -> list[dict]:
    """Scans top 10 acoustic predictions to ensure layered background hazards are caught."""
    classifier = get_audio_classifier()
    audio_dict = {
        "raw": audio.astype(np.float32),
        "sampling_rate": sr
    }
    predictions = classifier(audio_dict, top_k=top_k)
    return predictions

def calculate_stress_index(audio: np.ndarray, sr: int = 16000, detected_threats_present: bool = False) -> float:
    """
    Evaluates vocal panic and total acoustic turmoil:
    - Pitch instability (F0 variance)
    - High-frequency energy concentration (Centroid)
    - Dynamic peak-to-average volume spikes
    """
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Active frame threshold
    max_rms = np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else 0.01
    active_frames = rms > (0.15 * max_rms)
    
    if np.sum(active_frames) == 0:
        return 0.10

    # 1. Pitch Tracking (F0)
    f0, voiced_flag, _ = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )
    valid_voiced = f0[voiced_flag & active_frames] if voiced_flag is not None else np.array([])
    
    if len(valid_voiced) > 4:
        pitch_std = float(np.std(valid_voiced))
        pitch_stress = np.clip((pitch_std - 30.0) / 75.0, 0.0, 1.0)
    else:
        pitch_stress = 0.20

    # 2. Spectral Centroid (Screaming, gunshots, and alarms produce sharp high frequencies > 2500 Hz)
    centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    avg_centroid = float(np.mean(centroids[active_frames]))
    centroid_stress = np.clip((avg_centroid - 1400.0) / 1800.0, 0.0, 1.0)

    # 3. Dynamic Crest Factor (Loud peak bursts like gunshots / screams vs quiet floor)
    peak_energy = np.percentile(rms, 95)
    mean_energy = np.mean(rms) + 1e-6
    crest_factor = peak_energy / mean_energy
    crest_stress = np.clip((crest_factor - 1.5) / 2.5, 0.0, 1.0)

    # 4. Synthesize Combined Acoustic Distress
    raw_distress = (0.35 * pitch_stress) + (0.35 * centroid_stress) + (0.30 * crest_stress)
    
    # If loud sirens, gunshots, or alarms are acoustically confirmed, elevate stress index floor
    if detected_threats_present:
        raw_distress = max(raw_distress, 0.85)

    return round(float(np.clip(raw_distress, 0.05, 0.98)), 2)

def compute_triage_score(detected_events: list[dict], transcript: str, audio: np.ndarray, sr: int = 16000) -> tuple[int, float, list[str]]:
    """
    Combines environmental hazards, vocal distress, and lexical cues into final priority (1-5).
    """
    flagged_threats = []
    critical_detected = False
    moderate_detected = False

    # 1. Evaluate Acoustic Threats from AudioSet (Confidence threshold >= 12% across top 10)
    for item in detected_events:
        label = item["label"].lower()
        score = item["score"]

        if score >= 0.12:
            if any(k in label for k in CRITICAL_THREAT_KEYWORDS):
                critical_detected = True
                flagged_threats.append(f"{item['label']} ({score*100:.0f}%)")
            elif any(k in label for k in MODERATE_THREAT_KEYWORDS):
                moderate_detected = True
                flagged_threats.append(f"{item['label']} ({score*100:.0f}%)")

    # Remove duplicate label tags while preserving order
    flagged_threats = list(dict.fromkeys(flagged_threats))

    # 2. Compute Vocal Distress with Threat Context
    stress_index = calculate_stress_index(audio, sr, detected_threats_present=critical_detected)

    # 3. Lexical Verification from Whisper Transcript
    words = set(transcript.lower().replace(",", "").replace(".", "").split())
    has_emergency_keyword = bool(words & EMERGENCY_WORDS)
    has_non_emergency_keyword = any(phrase in transcript.lower() for phrase in NON_EMERGENCY_WORDS)

    # 4. Deterministic Triage Rule Engine
    if critical_detected or (stress_index >= 0.80 and has_emergency_keyword):
        # Critical emergencies (gunshots, alarms, intense panic) always force Level 5
        final_score = 5
    elif moderate_detected or (stress_index >= 0.50) or has_emergency_keyword:
        # Moderate incidents (sirens, loud distress, traffic crash)
        final_score = 4 if (stress_index >= 0.65 or moderate_detected) else 3
    elif has_non_emergency_keyword or stress_index < 0.35:
        # Calm calls (lost wallet, minor routine reports)
        final_score = 1
    else:
        final_score = 2

    return final_score, stress_index, flagged_threats

def run_triage_pipeline(file_path: str, audio: np.ndarray, sr: int = 16000) -> dict:
    # 1. Transcribe dialogue
    transcript = transcribe_audio(audio)
    
    # 2. Extract environmental events
    events = detect_threat_events(audio, sr, top_k=10)
    
    # 3. Calculate unified score and distress index
    triage_score, stress_index, threats = compute_triage_score(events, transcript, audio, sr)
    
    return {
        "transcript": transcript,
        "stress_index": stress_index,
        "flagged_threats": threats,
        "triage_score": triage_score
    }