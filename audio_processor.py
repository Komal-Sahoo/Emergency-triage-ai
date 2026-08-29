import librosa
import numpy as np
import soundfile as sf
#change by shree
#change 2
#change by komal
#afreen
#hello by h
#hello me ananya
def load_and_preprocess_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono, resamples to target_sr, 
    and normalizes the amplitude.
    """
    try:
        # Fast load using soundfile
        data, sr = sf.read(file_path, dtype='float32')
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Convert to mono
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        audio_signal = data
    except Exception: 
        # Fallback to librosa standard loader
        audio_signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Normalize amplitude
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = audio_signal / max_val
        
    return audio_signal, sr
