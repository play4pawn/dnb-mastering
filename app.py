import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
import io
import gc

st.set_page_config(page_title="DnB Master Limiter", layout="centered")
st.title("🔊 DnB Master Limiter - Liquid/Neuro Edition")

def design_highpass_sos(cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def apply_soft_clip(x, threshold=0.99):
    return np.tanh(x / threshold) * threshold

def add_exciter(audio, fs, crossover_hz=8000, drive=2.2, mix=0.15):
    """Hybrid Exciter: Sättigt nur >8kHz für Sheen ohne Härte"""
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    # 1. Highpass: Nur Top-End behalten
    sos = butter(4, crossover_hz/(fs*0.5), btype='high', output='sos')
    highs = sosfilt(sos, audio, axis=0)

    # 2. Sättigen mit tanh = gerade Harmonische
    excited = np.tanh(highs * drive)

    # 3. Dezent zumischen
    return audio + excited * mix

def master_process(audio, fs, cutoff_hz=32, ceiling_db=-0.3, target_lufs=-14.0, use_exciter=True):
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    # 1. 32Hz HPF Zero-Phase
    sos = design_highpass_sos(cutoff_hz, fs)
    audio = sosfilt(sos, audio, axis=0)

    # 2. NEU: Exciter vor dem Limiting
    if use_exciter:
        audio = add_exciter(audio, fs, crossover_hz=8000, drive=2.2, mix=0.15)

    # 3. Loudness
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_lufs/20.0)
    gain = target_rms / (rms + 1e-8)
    audio = audio * gain

    # 4. Ceiling + Soft Clip
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

    # NEU: Exciter Toggle
    use_exciter = st.checkbox("✨ Hybrid Exciter >8kHz", value=True, help="Fügt seidige Höhen hinzu. Aus = Clean Master")

    if st.button("Master jetzt"):
        with st.spinner("Mastere... HPF + Exciter + Limiter"):
            mastered = master_process(audio, fs, cutoff_hz=32, use_exciter=use_exciter)

        buf = io.BytesIO()
        sf.write(buf, mastered, fs, format='WAV', subtype='PCM_24')
        buf.seek(0)

        data = buf.getvalue()
        tag = "Excited" if use_exciter else "Clean"
        new_filename = f"Mastered {tag} {original_name}"

        st.download_button("Download WAV", data, new_filename)
        st.audio(data, format='audio/wav')
        st.success(f"Fertig! {tag} Version erstellt.")

        del audio, mastered, buf, data
        gc.collect()
