import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
import io
import gc # <-- NEU: Für Memory Cleanup

st.set_page_config(page_title="DnB Master Limiter", layout="centered")
st.title("🔊 DnB Master Limiter - Liquid/Neuro Edition")

def design_highpass_sos(cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def apply_soft_clip(x, threshold=0.99):
    return np.tanh(x / threshold) * threshold

def master_process(audio, fs, cutoff_hz=32, ceiling_db=-0.3, target_lufs=-14.0):
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    sos = design_highpass_sos(cutoff_hz, fs)
    audio = sosfilt(sos, audio, axis=0)

    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_lufs/20.0)
    gain = target_rms / (rms + 1e-8)
    audio = audio * gain

    ceiling = 10**(ceiling_db/20.0)
    audio = np.clip(audio, -ceiling, ceiling)
    audio = apply_soft_clip(audio, threshold=ceiling*0.98)

    return audio.astype(np.float32)

uploaded_file = st.file_uploader("WAV 48kHz hier rein", type=["wav"])

if uploaded_file is not None:
    original_name = uploaded_file.name
    audio, fs = sf.read(uploaded_file, dtype='float32')

    st.audio(uploaded_file, format='audio/wav')
    st.write(f"Input: {fs}Hz | {audio.shape[1] if audio.ndim>1 else 1} Kanal | {original_name}")

    if st.button("Master jetzt"):
        with st.spinner("Mastere... 32Hz HPF + Limiter"):
            mastered = master_process(audio, fs, cutoff_hz=32)

        buf = io.BytesIO()
        sf.write(buf, mastered, fs, format='WAV', subtype='PCM_24')
        buf.seek(0) # <-- FIX 1: Buffer auf Anfang setzen

        data = buf.getvalue() # <-- FIX 2: Bytes rausholen bevor buf weg ist
        new_filename = f"Mastered {original_name}"

        st.download_button("Download WAV", data, new_filename)
        st.audio(data, format='audio/wav')
        st.success("Fertig! Kein Phasenschmutz durch Zero-Phase HPF.")

        # FIX 3: RAM freigeben
        del audio, mastered, buf, data
        gc.collect()
