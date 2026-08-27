import numpy as np
import os
import librosa
import torch
import whisper
from transformers import pipeline
import streamlit as st
import os

import librosa
import numpy as np
import streamlit as st
import torch
import whisper
from transformers import pipeline


DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_whisper_model():
	return whisper.load_model("small", device=TORCH_DEVICE)


@st.cache_resource
def get_audio_classifier():
	return pipeline(
		"audio-classification",
		model="MIT/ast-finetuned-audioset-10-10-0.4593",
		device=DEVICE,
	)


CRITICAL_THREAT_KEYWORDS = [
	"gun", "gunshot", "gunfire", "machine gun", "firearm", "shot",
	"explosion", "blast", "bomb", "artillery", "scream", "screaming",
	"shout", "yell", "wail", "groan", "crying", "sobbing", "whimper",
	"distress", "fire alarm", "smoke detector", "smoke alarm", "alarm",
	"buzzer", "warning signal", "siren", "police car", "ambulance",
	"emergency vehicle", "crash", "breaking glass", "shatter",
]
MODERATE_THREAT_KEYWORDS = [
	"glass", "shatter", "crash", "smash", "traffic", "vehicle", "crying",
	"sob", "whimper", "bang", "thud", "knock", "impact", "engine",
]
EMERGENCY_WORDS = {
	"help", "emergency", "fire", "bleeding", "shot", "gun", "shooting",
	"dying", "attacker", "intruder", "crash", "knife", "smoke", "burning",
	"hurry", "war", "救命", "救", "死", "血", "枪", "火", "爆炸", "受伤",
	"快点", "救我", "危险", "救护车", "警车",
}
EMERGENCY_PHRASES = [
	"right behind me", "behind me", "what's your location",
	"what is your location", "tell the unit", "multiple callers", "send help",
	"need help", "someone is shooting", "shots fired", "gun shots", "gunshot",
	"under attack", "being attacked", "someone is hurt", "someone is injured",
	"there is a fire", "people are screaming", "people are crying", "i am scared",
	"i'm scared", "get out", "run", "hurry", "help me",
]


def prepare_audio_for_whisper(audio: np.ndarray) -> np.ndarray:
	"""Return normalized, one-dimensional mono float32 audio."""
	audio_arr = np.asarray(audio, dtype=np.float32)
	if audio_arr.ndim > 1:
		audio_arr = np.mean(audio_arr, axis=0 if audio_arr.shape[0] == 2 else -1)
	audio_arr = audio_arr.flatten()
	if audio_arr.size and (max_val := np.max(np.abs(audio_arr))) > 1.0:
		audio_arr = audio_arr / max_val
	return audio_arr


def transcribe_and_translate(audio: np.ndarray, file_path: str = None) -> tuple[str, str, str]:
	clean_audio = prepare_audio_for_whisper(audio)
	if len(clean_audio) < 1600:
		return "", "", "EN"
	model = get_whisper_model()
	try:
		result = model.transcribe(clean_audio, fp16=torch.cuda.is_available(),
								  condition_on_previous_text=True, verbose=False)
	except Exception:
		if not file_path or not os.path.exists(file_path):
			return "", "", "EN"
		result = model.transcribe(file_path, fp16=torch.cuda.is_available(),
								  condition_on_previous_text=True)
	native_text = result.get("text", "").strip()
	detected_lang = result.get("language", "en")
	english_text = native_text
	if detected_lang != "en" and native_text:
		try:
			translated = model.transcribe(clean_audio, task="translate",
										  fp16=torch.cuda.is_available(), verbose=False)
			english_text = translated.get("text", "").strip()
		except Exception:
			pass
	return native_text, english_text, detected_lang


def detect_threat_events_fast(audio: np.ndarray, sr: int = 16000,
							  chunk_sec: float = 5.0,
							  hop_sec: float = 2.5) -> list[dict]:
	classifier = get_audio_classifier()
	chunk_samples, hop_samples = int(chunk_sec * sr), int(hop_sec * sr)
	label_scores, label_hits = {}, {}
	for start_idx in range(0, len(audio), hop_samples):
		segment = audio[start_idx:min(start_idx + chunk_samples, len(audio))].astype(np.float32)
		if len(segment) < int(sr):
			continue
		try:
			for prediction in classifier({"raw": segment, "sampling_rate": sr}, top_k=20):
				label, score = prediction["label"], float(prediction["score"])
				label_scores[label] = max(label_scores.get(label, 0.0), score)
				if score >= 0.03:
					label_hits[label] = label_hits.get(label, 0) + 1
		except Exception:
			continue
	events = [{"label": label, "score": score, "hits": label_hits.get(label, 0)}
			  for label, score in label_scores.items()
			  if score >= 0.025 or label_hits.get(label, 0) >= 2]
	return sorted(events, key=lambda x: (x["score"], x["hits"]), reverse=True)


def calculate_stress_index_fast(audio: np.ndarray, sr: int = 16000,
								detected_threats_present: bool = False) -> float:
	frame_length, hop_length = 2048, 512
	rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
	max_rms = np.max(rms) if len(rms) and np.max(rms) > 0 else 0.01
	active_frames = rms > 0.08 * max_rms
	if not np.any(active_frames):
		return 0.15
	centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
	zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length,
											 hop_length=hop_length)[0]
	centroid_stress = np.clip((np.percentile(centroids[active_frames], 85) - 900.0) / 1300.0, 0.0, 1.0)
	zcr_stress = np.clip((np.percentile(zcr[active_frames], 85) - 0.055) / 0.14, 0.0, 1.0)
	crest_stress = np.clip((np.percentile(rms, 95) / (np.mean(rms) + 1e-6) - 1.15) / 1.6, 0.0, 1.0)
	raw_distress = 0.30 * centroid_stress + 0.40 * zcr_stress + 0.30 * crest_stress
	if detected_threats_present:
		raw_distress = max(raw_distress, 0.75)
	return round(float(np.clip(raw_distress, 0.10, 0.95)), 2)


def compute_triage_score(detected_events: list[dict], native_text: str,
						 english_text: str, audio: np.ndarray,
						 sr: int = 16000) -> tuple[int, float, list[str]]:
	flagged, critical, moderate = [], False, False
	for item in detected_events:
		label, score = item["label"].lower(), item["score"]
		if score < 0.025:
			continue
		if any(k in label for k in CRITICAL_THREAT_KEYWORDS):
			critical = True
		elif any(k in label for k in MODERATE_THREAT_KEYWORDS):
			moderate = True
		else:
			continue
		flagged.append(f"{item['label']} ({score * 100:.0f}%)")
	flagged = list(dict.fromkeys(flagged))
	stress = calculate_stress_index_fast(audio, sr, critical)
	text = f"{native_text} {english_text}".lower()
	emergency = any(word in text for word in EMERGENCY_WORDS) or any(p in text for p in EMERGENCY_PHRASES)
	if critical:
		final = 5 if stress >= 0.70 or emergency else 4
	elif moderate:
		final = 4 if stress >= 0.60 or emergency else 3
	elif stress >= 0.70:
		final = 4
	elif stress >= 0.50 or emergency:
		final = 3
	elif stress < 0.30:
		final = 1
	else:
		final = 2
	return final, stress, flagged
# Automatic Hardware Acceleration Check
DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
@st.cache_resource
def get_whisper_model():
	return whisper.load_model("small", device=TORCH_DEVICE)
@st.cache_resource
def get_audio_classifier():
	return pipeline(
		"audio-classification",
		model="MIT/ast-finetuned-audioset-10-10-0.4593",
		device=DEVICE
	)
CRITICAL_THREAT_KEYWORDS = [
# Weapons / explosions
"gun",
"gunshot",
"gunfire",
"machine gun",
"firearm",
"shot",
"explosion",
"blast",
"bomb",
"artillery",
# Human distress
"scream",
"screaming",

"shout",
"yell",
"wail",
"groan",
"crying",
"sobbing",
"whimper",
"distress",
# Alarms
"fire alarm",
"smoke detector",
"smoke alarm",
"alarm",
"buzzer",
"warning signal",
# Emergency vehicles
"siren",
"police car",
"ambulance",
"emergency vehicle",
# Other dangerous acoustic events
"crash",
"breaking glass",
"shatter"
]
MODERATE_THREAT_KEYWORDS = [
"glass", "shatter", "crash", "smash", "traffic", "vehicle",
"crying", "sob", "whimper", "bang", "thud", "glass", "shatter",
"crash", "smash", "traffic", "vehicle",
"crying",
"cry",
"sob",
"whimper",
"bang",
"thud",
"knock",

"impact",
"engine"
]
EMERGENCY_WORDS = {
"help", "emergency", "fire", "bleeding", "shot", "gun", "shooting",
"dying", "attacker", "intruder", "crash", "knife", "smoke", "burning",
"hurry", "war",
"救命", "救", "死", "血", "枪", "火", "爆炸", "受伤", "快点", "救我", "危险",
"救护车", "警车",
}
EMERGENCY_PHRASES = [
"right behind me",
"behind me",
"what's your location",
"what is your location",
"tell the unit",
"multiple callers",
"send help",
"need help",
"someone is shooting",
"shots fired",
"gun shots",
"gunshot",
"under attack",
"being attacked",
"someone is hurt",
"someone is injured",
"there is a fire",
"people are screaming",
"people are crying",
"i am scared",
"i'm scared",
"get out",
"run",
"hurry",
"help me"
]

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
    """Transcribes and translates using Whisper with fail-safes for tensor
    reshaping errors."""
    clean_audio = prepare_audio_for_whisper(audio)
    # Guard against 0-element or near-zero empty audio
    if len(clean_audio) < 1600:  # less than 0.1s
        return "", "", "EN"

    model = get_whisper_model()
    native_text = ""
    detected_lang = "en"

    # Whisper can ingest the clean 1D numpy array directly
    try:
        native_result = model.transcribe(
            clean_audio,
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=True,
            verbose=False,
        )
        native_text = native_result.get("text", "").strip()
        detected_lang = native_result.get("language", "en")
    except Exception:
        # Fallback to loading file path directly if array decoding hit a
        # shape error
        if file_path and os.path.exists(file_path):
            try:
                native_result = model.transcribe(
                    file_path,
                    fp16=torch.cuda.is_available(),
                    condition_on_previous_text=True,
                    verbose=False,
                )
                native_text = native_result.get("text", "").strip()
                detected_lang = native_result.get("language", "en")
            except Exception:
                native_text = ""
                detected_lang = "en"
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
                condition_on_previous_text=True,
                verbose=False,
            )
            english_text = eng_result.get("text", "").strip()
        except Exception:
            english_text = native_text

    return native_text, english_text, detected_lang


def detect_threat_events_fast(
	audio: np.ndarray,
	sr: int = 16000,
	chunk_sec: float = 5.0,
	hop_sec: float = 2.5,
) -> list[dict]:
	classifier = get_audio_classifier()
	chunk_samples = int(chunk_sec * sr)
	hop_samples = int(hop_sec * sr)
	total_samples = len(audio)
	label_scores = {}
	label_hits = {}

	for start_idx in range(0, total_samples, hop_samples):
		end_idx = min(start_idx + chunk_samples, total_samples)
		segment = audio[start_idx:end_idx].astype(np.float32)
		if len(segment) < int(sr * 1.0):
			continue
		audio_dict = {"raw": segment, "sampling_rate": sr}
		try:
			preds = classifier(audio_dict, top_k=20)
			for prediction in preds:
				label = prediction["label"]
				score = float(prediction["score"])
				label_scores[label] = max(label_scores.get(label, 0.0), score)
				if score >= 0.03:
					label_hits[label] = label_hits.get(label, 0) + 1
		except Exception:
			continue

	sorted_events = []
	for label, score in label_scores.items():
		hits = label_hits.get(label, 0)
		if score >= 0.025 or hits >= 2:
			sorted_events.append({"label": label, "score": score, "hits": hits})
	sorted_events.sort(key=lambda event: (event["score"], event["hits"]), reverse=True)
	return sorted_events
def calculate_stress_index_fast(audio: np.ndarray, sr: int = 16000,
detected_threats_present: bool = False) -> float:
	"""
	High-speed vectorized distress calculation (~0.05s on CPU):
	Replaces slow pyin pitch tracking with Spectral Rolloff and Zero
	Crossing dynamics.
	"""
	frame_length = 2048
	hop_length = 512
	rms = librosa.feature.rms(y=audio, frame_length=frame_length,
							  hop_length=hop_length)[0]
	max_rms = np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else 0.01
	active_frames = rms > (0.08 * max_rms)
	if np.sum(active_frames) == 0:
		return 0.15
	# 1. High Frequency Energy Spikes (Spectral Centroid)
	centroids = librosa.feature.spectral_centroid(y=audio, sr=sr,
												  hop_length=hop_length)[0]
	top_centroid = float(np.percentile(centroids[active_frames], 85)) if np.any(active_frames) else 1000.0
	centroid_stress = np.clip((top_centroid - 900.0) / 1300.0, 0.0, 1.0)
	# 2. Vocal Turbulence / Screaming (Zero Crossing Rate)
	zcr = librosa.feature.zero_crossing_rate(y=audio,
											 frame_length=frame_length,
											 hop_length=hop_length)[0]
	top_zcr = float(np.percentile(zcr[active_frames], 85)) if np.any(active_frames) else 0.05
	zcr_stress = np.clip((top_zcr - 0.055) / 0.14, 0.0, 1.0)
	# 3. Crest Dynamic Spikes (Gunshots/Screams vs quiet baseline)
	peak_energy = np.percentile(rms, 95)
	mean_energy = np.mean(rms) + 1e-6
	crest_factor = peak_energy / mean_energy
	crest_stress = np.clip((crest_factor - 1.15) / 1.6, 0.0, 1.0)
	raw_distress = (0.30 * centroid_stress) + (0.40 * zcr_stress) + (0.30 * crest_stress)
	if detected_threats_present:
		raw_distress = max(raw_distress, 0.75)
	return round(float(np.clip(raw_distress, 0.10, 0.95)), 2)

def compute_triage_score(detected_events: list[dict], native_text: str,
english_text: str, audio: np.ndarray, sr: int = 16000) -> tuple[int,
float, list[str]]:
	flagged_threats = []
	critical_detected = False
	moderate_detected = False
	for item in detected_events:
		label = item["label"].lower()
		score = item["score"]
		if score >= 0.025:
			if any(k in label for k in CRITICAL_THREAT_KEYWORDS):
				critical_detected = True
				flagged_threats.append(f"{item['label']} ({score*100:.0f}%)")
			elif any(k in label for k in MODERATE_THREAT_KEYWORDS):
				moderate_detected = True
				flagged_threats.append(f"{item['label']} ({score*100:.0f}%)")
	flagged_threats = list(dict.fromkeys(flagged_threats))
	stress_index = calculate_stress_index_fast(
		audio, sr, detected_threats_present=critical_detected
	)
	combined_text = (native_text + " " + english_text).lower()
	has_emergency_keyword = (
		any(word in combined_text for word in EMERGENCY_WORDS)
		or any(phrase in combined_text for phrase in EMERGENCY_PHRASES)
	)
	if critical_detected:
		# Confirmed critical acoustic event
		if stress_index >= 0.70 or has_emergency_keyword:
			final_score = 5
		else:
			final_score = 4
	elif moderate_detected:
		if stress_index >= 0.60 or has_emergency_keyword:
			final_score = 4
		else:
			final_score = 3
	elif stress_index >= 0.70:
		final_score = 4
	elif stress_index >= 0.50 or has_emergency_keyword:
		final_score = 3
	elif stress_index < 0.30:
		final_score = 1
	else:
		final_score = 2
	return final_score, stress_index, flagged_threats
def run_triage_pipeline(file_path: str, audio: np.ndarray, sr: int =
16000) -> dict:
	# 1. Clean audio array to 1D mono
	audio = prepare_audio_for_whisper(audio)
	# 2. Transcribe & Translate with path fallback
	native_transcript, english_transcript, lang = transcribe_and_translate(
		audio,
		file_path=file_path
	)
	# 3. Detect acoustic threats
	events = detect_threat_events_fast(
		audio,
		sr=sr,
		chunk_sec=5.0,
		hop_sec=2.5
	)
	# DEBUG: See what AST is actually detecting
	print("\n========== AST EVENTS ==========")
	for event in events:
		print(event)
	print("================================\n")
	# 4. Compute Triage Score & Stress Index
	triage_score, stress_index, threats = compute_triage_score(
		events,
		native_transcript,
		english_transcript,
		audio,
		sr
	)
	print(f"Stress Index: {stress_index}")
	print(f"Flagged Threats: {threats}")
	print(f"Triage Score: {triage_score}/5")
	return {
		"native_transcript": native_transcript,
		"english_transcript": english_transcript,
		"language": lang.upper(),
		"stress_index": stress_index,
		"flagged_threats": threats,
		"triage_score": triage_score
	}