import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pedalboard import Pedalboard, Compressor, Gain, Limiter, HighshelfFilter
import io

# Seitenkonfiguration für mobile Ansicht optimieren
st.set_page_config(page_title="DnB Mastering Engine", page_icon="🔊", layout="centered")

st.title("🔊 DnB AI-Mastering Engine")
st.write("Lade deine Suno WAV-Datei hoch, um den Reese Bass zu bändigen und die Drums zu pushen.")

# 1. Datei-Upload
uploaded_file = st.file_uploader("Suno WAV-Datei auswählen", type=["wav"])

# Definiere den steilen 35Hz Low-Cut Filter
def apply_low_cut(data, samplerate):
    sos = butter(12, 35, 'hp', fs=samplerate, output='sos')
    return sosfilt(sos, data, axis=0)

if uploaded_file is not None:
    st.info("Datei erfolgreich hochgeladen. Starte Mastering...")
    
    # Audio einlesen
    data, samplerate = sf.read(uploaded_file)
    
    # Fortschrittsanzeige
    progress_bar = st.progress(0)
    
    # Schritt 1: Low-End aufräumen
    st.write("🧹 Schneide Sub-Matsch unter 35Hz ab...")
    clean_data = apply_low_cut(data, samplerate)
    progress_bar.progress(40)
    
    # Schritt 2: Mastering-Kette anwenden
    st.write("🎛️ Wende DnB-Mastering-Kette an (Glue Compression & Limiting)...")
    board = Pedalboard([
        Compressor(threshold_db=-16, ratio=2.0, attack_ms=20, release_ms=150),
        HighshelfFilter(cutoff_frequency_hz=12000, gain_db=1.5),
        Gain(gain_db=4.5), # Treibt das Signal sanft in den Limiter
        Limiter(threshold_db=-0.1)
    ])
    mastered_data = board(clean_data, samplerate)
    progress_bar.progress(80)
    
    # Schritt 3: In den Speicher schreiben für den Download
    st.write("💾 Bereite Download vor...")
    wav_io = io.BytesIO()
    sf.write(wav_io, mastered_data, samplerate, subtype='PCM_16', format='WAV')
    wav_io.seek(0)
    progress_bar.progress(100)
    
    st.success("🎉 Mastering abgeschlossen!")
    
    # Vorhör-Player (optional auf dem Handy)
    st.audio(wav_io, format="audio/wav")
    
    # Download-Button für das Handy
    st.download_button(
        label="🚀 Gemasterte WAV herunterladen",
        data=wav_io,
        file_name=f"mastered_{uploaded_file.name}",
        mime="audio/wav"
    )