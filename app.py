import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pedalboard import Pedalboard, Compressor, Gain, Limiter, HighshelfFilter
import io

st.title("🔊 DnB AI-Mastering Engine")

uploaded_file = st.file_uploader("Suno WAV-Datei auswählen", type=["wav"])

if uploaded_file is not None:
    st.info("Mastering läuft...")
    
    # Audio einlesen
    data, samplerate = sf.read(uploaded_file)
    
    # 35Hz Low-Cut Filter
    sos = butter(12, 35, 'hp', fs=samplerate, output='sos')
    clean_data = sosfilt(sos, data, axis=0)
    
    # Mastering-Kette
    board = Pedalboard([
        Compressor(threshold_db=-16, ratio=2.0, attack_ms=20, release_ms=150),
        HighshelfFilter(cutoff_frequency_hz=12000, gain_db=1.5),
        Gain(gain_db=4.5),
        Limiter(threshold_db=-0.1)
    ])
    mastered_data = board(clean_data, samplerate)
    
    # Datei für Download bereitmachen
    wav_io = io.BytesIO()
    sf.write(wav_io, mastered_data, samplerate, subtype='PCM_16', format='WAV')
    wav_io.seek(0)
    
    st.success("🎉 Fertig!")
    st.audio(wav_io, format="audio/wav")
    st.download_button(label="🚀 Download", data=wav_io, file_name="mastered_track.wav", mime="audio/wav")
