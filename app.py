import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io

st.title("🔊 DnB AI-Mastering Engine (Stable)")
st.write("Bändige den Reese-Bass und maximiere den Druck deiner Suno WAV-Dateien.")

uploaded_file = st.file_uploader("Suno WAV-Datei auswählen", type=["wav"])

if uploaded_file is not None:
    st.info("Mastering läuft im Hintergrund...")
    
    # 1. Audio einlesen
    data, samplerate = sf.read(uploaded_file)
    
    # 2. 35Hz Low-Cut Filter (Entlastet die Anlage vom Sub-Matsch)
    sos = butter(12, 35, 'hp', fs=samplerate, output='sos')
    clean_data = sosfilt(sos, data, axis=0)
    
    # Zwischenspeichern für Pydub-Bearbeitung
    temp_io = io.BytesIO()
    sf.write(temp_io, clean_data, samplerate, subtype='PCM_16', format='WAV')
    temp_io.seek(0)
    
    # 3. Lautstärke-Maximierung und Peak-Limiting via Pydub
    sound = AudioSegment.from_wav(temp_io)
    
    # +4.5 dB Gain hinzufügen (Treibt die Lautstärke nach oben)
    louder_sound = sound + 4.5
    
    # Brickwall Limiter-Effekt (Verhindert Clipping bei Spitzen über -0.1 dB)
    mastered_sound = louder_sound.normalize(headroom=0.1)
    
    # 4. Datei für den Download bereitmachen
    wav_io = io.BytesIO()
    mastered_sound.export(wav_io, format="wav")
    wav_io.seek(0)
    
    st.success("🎉 Mastering erfolgreich abgeschlossen!")
    st.audio(wav_io, format="audio/wav")
    st.download_button(label="🚀 Gemasterte WAV herunterladen", data=wav_io, file_name=f"mastered_{uploaded_file.name}", mime="audio/wav")
