import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io

st.title("🔊 DnB AI-Mastering Engine (with Phase-Fix)")
st.write("Bändige den Reese-Bass, korrigiere Phasenprobleme und erzeuge frische Obertöne.")

uploaded_file = st.file_uploader("Suno WAV-Datei auswählen", type=["wav"])

if uploaded_file is not None:
    st.info("Phase wird korrigiert & Mastering läuft...")
    
    # 1. Audio einlesen
    data, samplerate = sf.read(uploaded_file)
    
    # Falls Datei Stereo ist (2 Kanäle), korrigieren wir die Phase im Bassbereich
    if len(data.shape) > 1 and data.shape[1] == 2:
        # Frequenzbänder für den Bass-Split definieren (120 Hz)
        sos_low = butter(12, 120, 'lp', fs=samplerate, output='sos')
        sos_high = butter(12, 120, 'hp', fs=samplerate, output='sos')
        
        # Audio in Bass und Höhen trennen
        bass_part = sosfilt(sos_low, data, axis=0)
        high_part = sosfilt(sos_high, data, axis=0)
        
        # Phasen-Fix: Den Bassbereich komplett Mono schalten (Mittelwert aus links & rechts)
        mono_bass = np.mean(bass_part, axis=1)
        bass_part[:, 0] = mono_bass
        bass_part[:, 1] = mono_bass
        
        # Bass (jetzt Phasen-korrigiert in Mono) und Stereo-Höhen wieder zusammenfügen
        data = bass_part + high_part
    
    # 2. 30Hz Low-Cut Filter (Jetzt auf dem phasen-sauberen Signal)
    sos_cut = butter(12, 30, 'hp', fs=samplerate, output='sos')
    clean_data = sosfilt(sos_cut, data, axis=0)
    
    # 3. Dynamic Exciter / Spectral Recovery
    excitation_factor = 0.05
    excited_data = clean_data + (np.sin(clean_data * np.pi * 0.5) * excitation_factor)
    excited_data = np.clip(excited_data, -1.0, 1.0)
    
    # Zwischenspeichern für Pydub-Bearbeitung
    temp_io = io.BytesIO()
    sf.write(temp_io, excited_data, samplerate, subtype='PCM_16', format='WAV')
    temp_io.seek(0)
    
    # 4. Lautstärke-Maximierung und Peak-Limiting (+3.5 dB für perfekte Dynamik)
    sound = AudioSegment.from_wav(temp_io)
    louder_sound = sound + 3.5
    mastered_sound = louder_sound.normalize(headroom=0.1)
    
    # 5. Datei für den Download bereitmachen
    wav_io = io.BytesIO()
    mastered_sound.export(wav_io, format="wav")
    wav_io.seek(0)
    
    st.success("🎉 Phase korrigiert & Mastering abgeschlossen!")
    st.audio(wav_io, format="audio/wav")
    st.download_button(label="🚀 Gemasterte WAV (Phasen-optimiert) herunterladen", data=wav_io, file_name=f"mastered_phase_fixed_{uploaded_file.name}", mime="audio/wav")
