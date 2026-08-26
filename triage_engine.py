import numpy as np
import librosa
import torch
import whisper
from transformers import pipeline
import streamlit as st

# Automatic Hardware Acceleration Check
DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_whisper_model():
    return whisper.load_model("base", device=TORCH_DEVICE)

@st.cache_resource
def get_audio_classifier():
    return pipeline(
        "audio-classification", 
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
        device=DEVICE
    )

CRITICAL_THREAT_KEYWORDS = [
    "gun", "gunshot", "gunfire", "explosion", "blast", "artillery", "bomb",
    "scream", "screaming", "shout", "yell", "wail", "groan",
    "fire alarm", "smoke detector", "alarm", "buzzer", "siren", "emergency vehicle"
]

MODERATE_THREAT_KEYWORDS = [
    "glass", "shatter", "crash", "smash", "traffic", "vehicle",
    "crying", "sob", "whimper", "bang", "thud"
]

EMERGENCY_WORDS = {
    "help", "emergency", "fire", "bleeding", "shot", "gun", "shooting", 
    "dying", "attacker", "intruder", "crash", "knife", "smoke", "burning", "hurry", "war",
    "救命", "救", "死", "血", "枪", "火", "爆炸", "受伤", "快点", "救我", "危险", "救护车", "警车"
}

def prepare_audio_for_whisper(audio: np.ndarray) -> np.ndarray:
    """Ensures audio is 1D mono float32 normalized between -1.0 and 1.0."""
    audio_arr = np.asarray(audio, dtype=np.float32)
    
    # If stereo or multi-channel, convert to mono
    if audio_arr.ndim > 1:
        if audio_arr.shape[0] == 2:
            audio_arr = np.mean(audio_arr, axis=0)
        else:
            audio_arr = np.mean(audio_arr, axis=-1)
            
    # Flatten just in case
    audio_arr = audio_arr.flatten()
    
    # Normalize if integer or out of bounds
    max_val = np.max(np.abs(audio_arr))
    if max_val > 1.0:
        audio_arr = audio_arr / max_val
        
    return audio_arr

def transcribe_and_translate(audio: np.ndarray, file_path: str = None) -> tuple[str, str, str]:
    """Transcribes and translates using Whisper with fail-safes for tensor reshaping errors."""
    clean_audio = prepare_audio_for_whisper(audio)
    
    # Guard against 0-element or near-zero empty audio
    if len(clean_audio) < 1600:  # less than 0.1s
        return "", "", "EN"

    model = get_whisper_model()
    
    # Whisper can ingest the clean 1D numpy array directly
    try:
        native_result = model.transcribe(
            clean_audio, 
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=False,
            verbose=False
        )
        native_text = native_result.get("text", "").strip()
        detected_lang = native_result.get("language", "en")
    except Exception as e:
        # Fallback to loading file path directly if array decoding hit a shape error
        if file_path and os.path.exists(file_path):
            native_result = model.transcribe(
                file_path,
                fp16=torch.cuda.is_available(),
                condition_on_previous_text=False
            )
            native_text = native_result.get("text", "").strip()
            detected_lang = native_result.get("language", "en")
        else:
            native_text = ""
            detected_lang = "en"

    # English translation (if non-English)
    english_text = native_text
    if detected_lang != "en" and native_text:
        try:
            eng_result = model.transcribe(
                clean_audio,
                task="translate",
                fp16=torch.cuda.is_available(),
                condition_on_previous_text=False,
                verbose=False
            )
            english_text = eng_result.get("text", "").strip()
        except Exception:
            english_text = native_text
        
    return native_text, english_text, detected_lang

def detect_threat_events_fast(audio: np.ndarray, sr: int = 16000, chunk_sec: float = 10.0, hop_sec: float = 10.0) -> list[dict]:
    """Fast batched/sampled threat detection avoiding CPU pipeline stalls."""
    classifier = get_audio_classifier()
    chunk_samples = int(chunk_sec * sr)
    hop_samples = int(hop_sec * sr)
    total_samples = len(audio)
    
    label_scores = {}
    
    # Non-overlapping 10s steps for maximum speed
    for start_idx in range(0, total_samples, hop_samples):
        end_idx = min(start_idx + chunk_samples, total_samples)
        segment = audio[start_idx:end_idx].astype(np.float32)
        
        if len(segment) < (sr * 2):
            continue
            
        audio_dict = {"raw": segment, "sampling_rate": sr}
        try:
            preds = classifier(audio_dict, top_k=6)
            for p in preds:
                lbl = p["label"]
                scr = p["score"]
                label_scores[lbl] = max(label_scores.get(lbl, 0.0), scr)
        except Exception:
            continue

    sorted_events = [{"label": k, "score": v} for k, v in sorted(label_scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_events

def calculate_stress_index_fast(audio: np.ndarray, sr: int = 16000, detected_threats_present: bool = False) -> float:
    """
    High-speed vectorized distress calculation (~0.05s on CPU):
    Replaces slow pyin pitch tracking with Spectral Rolloff and Zero Crossing dynamics.
    """
    frame_length = 2048
    hop_length = 512
    
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else 0.01
    active_frames = rms > (0.08 * max_rms)
    
    if np.sum(active_frames) == 0:
        return 0.15

    # 1. High Frequency Energy Spikes (Spectral Centroid)
    centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    top_centroid = float(np.percentile(centroids[active_frames], 85)) if np.any(active_frames) else 1000.0
    centroid_stress = np.clip((top_centroid - 1200.0) / 1800.0, 0.0, 1.0)

    # 2. Vocal Turbulence / Screaming (Zero Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    top_zcr = float(np.percentile(zcr[active_frames], 85)) if np.any(active_frames) else 0.05
    zcr_stress = np.clip((top_zcr - 0.08) / 0.20, 0.0, 1.0)

    # 3. Crest Dynamic Spikes (Gunshots/Screams vs quiet baseline)
    peak_energy = np.percentile(rms, 95)
    mean_energy = np.mean(rms) + 1e-6
    crest_factor = peak_energy / mean_energy
    crest_stress = np.clip((crest_factor - 1.3) / 2.2, 0.0, 1.0)

    raw_distress = (0.35 * centroid_stress) + (0.35 * zcr_stress) + (0.30 * crest_stress)
    
    if detected_threats_present:
        raw_distress = max(raw_distress, 0.78)

    return round(float(np.clip(raw_distress, 0.10, 0.95)), 2)

def compute_triage_score(detected_events: list[dict], native_text: str, english_text: str, audio: np.ndarray, sr: int = 16000) -> tuple[int, float, list[str]]:
    flagged_threats = []
    critical_detected = False
    moderate_detected = False

    for item in detected_events:
        label = item["label"].lower()
        score = item["score"]

        if score >= 0.08:
            if any(k in label for k in CRITICAL_THREAT_KEYWORDS):
                critical_detected = True
                flagged_threats.append(f"{item['label']} ({score*100:.0f}%)")
            elif any(k in label for k in MODERATE_THREAT_KEYWORDS):
                moderate_detected = True
                flagged_threats.append(f"{item['label']} ({score*100:.0f}%)")

    flagged_threats = list(dict.fromkeys(flagged_threats))
    stress_index = calculate_stress_index_fast(audio, sr, detected_threats_present=critical_detected)

    combined_text = (native_text + " " + english_text).lower()
    has_emergency_keyword = any(word in combined_text for word in EMERGENCY_WORDS)

    if critical_detected or (stress_index >= 0.75 and has_emergency_keyword):
        final_score = 5 if (critical_detected and stress_index >= 0.70) else 4
    elif moderate_detected or (stress_index >= 0.50) or has_emergency_keyword:
        final_score = 4 if (stress_index >= 0.60 or moderate_detected) else 3
    elif stress_index < 0.30:
        final_score = 1
    else:
        final_score = 2

    return final_score, stress_index, flagged_threats

def run_triage_pipeline(file_path: str, audio: np.ndarray, sr: int = 16000) -> dict:
    # 1. Clean audio array to 1D mono
    audio = prepare_audio_for_whisper(audio)
    
    # 2. Transcribe & Translate with path fallback
    native_transcript, english_transcript, lang = transcribe_and_translate(audio, file_path=file_path)
    
    # 3. Detect acoustic threats
    events = detect_threat_events_fast(audio, sr=sr, chunk_sec=10.0, hop_sec=10.0)
    
    # 4. Compute Triage Score & Stress Index
    triage_score, stress_index, threats = compute_triage_score(events, native_transcript, english_transcript, audio, sr)
    
    return {
        "native_transcript": native_transcript,
        "english_transcript": english_transcript,
        "language": lang.upper(),
        "stress_index": stress_index,
        "flagged_threats": threats,
        "triage_score": triage_score
    }