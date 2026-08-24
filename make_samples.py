import numpy as np
import scipy.io.wavfile as wavfile

def generate_synthetic_samples():
    sr = 16000  # 16 kHz sample rate
    duration = 4.0  # 4 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Low Priority Sample: Gentle low-frequency hum (routine call)
    low_freq = 200  # 200 Hz tone
    low_audio = 0.3 * np.sin(2 * np.pi * low_freq * t)
    # Add subtle random background noise
    low_audio += 0.01 * np.random.normal(0, 1, len(t))
    wavfile.write("sample_low.wav", sr, (low_audio * 32767).astype(np.int16))
    print("Created: sample_low.wav")

    # 2. Critical Priority Sample: High-pitched pulsating siren + loud noise (alarm simulation)
    # Pulsing siren tone alternating between 800 Hz and 1200 Hz
    freq_mod = 1000 + 300 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    siren_audio = 0.6 * np.sin(phase)
    # Add loud erratic background noise bursts
    noise = 0.2 * np.random.normal(0, 1, len(t))
    critical_audio = np.clip(siren_audio + noise, -1.0, 1.0)
    wavfile.write("sample_critical.wav", sr, (critical_audio * 32767).astype(np.int16))
    print("Created: sample_critical.wav")

if __name__ == "__main__":
    generate_synthetic_samples()