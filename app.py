import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io

st.title("🔊 DnB AI-Mastering Engine (with Exciter)")
st.write("Bändige den Reese-Bass und erzeuge frische Obertöne im Hochtonbereich.")

uploaded_file = st.file_uploader("Suno WAV-Datei auswählen", type=["wav"])

if uploaded_file is not None:
    st.info("Mastering & Spectral Recovery laufen...")
    
    # 1. Audio einlesen
    data, samplerate = sf.read(uploaded_file)
    
    # 2. 35Hz Low-Cut Filter
    sos = butter(12, 30, 'hp', fs=samplerate, output='sos')
    clean_data = sosfilt(sos, data, axis=0)
    
    # 3. DYNAMIC EXCITER / SPECTRAL RECOVERY (Pure Python)
    # Wir erzeugen ganz leichte, harmonische Obertöne durch sanfte Sättigung
    # Das simuliert die verlorenen Frequenzen über 15 kHz
    excitation_factor = 0.05  # Dezente Stärke, um Rauschen zu vermeiden
    excited_data = clean_data + (np.sin(clean_data * np.pi * 0.5) * excitation_factor)
    
    # Signal begrenzen, damit mathematisch nichts übersteuert
    excited_data = np.clip(excited_data, -1.0, 1.0)
    
    # Zwischenspeichern für Pydub-Bearbeitung
    temp_io = io.BytesIO()
    sf.write(temp_io, excited_data, samplerate, subtype='PCM_16', format='WAV')
    temp_io.seek(0)
    
    # 4. Lautstärke-Maximierung und Peak-Limiting via Pydub
    sound = AudioSegment.from_wav(temp_io)
    
    # +4.5 dB Gain für den kommerziellen DnB-Druck
    louder_sound = sound + 3.5
    
    # Brickwall Limiter-Effekt (Verhindert Clipping bei Spitzen über -0.1 dB)
    mastered_sound = louder_sound.normalize(headroom=0.1)
    
    # 5. Datei für den Download bereitmachen
    wav_io = io.BytesIO()
    mastered_sound.export(wav_io, format="wav")
    wav_io.seek(0)
    
    st.success("🎉 Mastering & Excitation erfolgreich abgeschlossen!")
    st.audio(wav_io, format="audio/wav")
    st.download_button(label="🚀 Gemasterte WAV mit Exciter herunterladen", data=wav_io, file_name=f"mastered_exciter_{uploaded_file.name}", mime="audio/wav")
