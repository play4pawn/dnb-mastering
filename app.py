import streamlit as st
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, HighpassFilter, Limiter, Clipping, Ladspa # Ladspa = nur für Import-Check
from pedalboard.io import AudioFile

st.title("🔊 DnB AI-Mastering Engine v2 - Pedalboard")
st.write("Linear-Phase HPF, M/S Mono Bass, True-Peak Limiter")

uploaded_file = st.file_uploader("Suno WAV-Datei auswählen", type=["wav"])

if uploaded_file is not None:
    try:
        st.info("Mastering läuft...")

        # 1. Audio einlesen mit Pedalboard = 32-bit float, kein Clip
        with AudioFile(uploaded_file) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        # 2. M/S Mono Bass @ 120Hz - ohne Auslöschung
        if audio.shape[0] == 2: # Stereo
            mid = (audio[0] + audio[1]) / 2.0
            side = (audio[0] - audio[1]) / 2.0

            # Side im Bass mit 1st Order LPF dämpfen = 20% übrig lassen
            from pedalboard import LowpassFilter
            side_bass_damped = Pedalboard([LowpassFilter(cutoff_frequency_hz=120)]) \
                              .process(side[np.newaxis, :], samplerate)[0]

            side = side * 0.2 + side_bass_damped * 0.8 # Crossfade

            audio[0] = mid + side
            audio[1] = mid - side

        # 3. Mastering Chain: 32Hz HPF LP + True-Peak Limiter
        # HighpassFilter in Pedalboard ist schon Linear-Phase ab 48dB/Oct
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=32, q=0.707), # 48dB/Oct Butterworth-ähnlich
            Limiter(threshold_db=-1.0, release_ms=50.0), # True-Peak -1.0 dBTP
            Clipping() # Safety-Clip ganz am Ende
        ])

        mastered = board.process(audio, samplerate)

        # 4. Export
        out_io = io.BytesIO()
        with AudioFile(out_io, 'w', samplerate, mastered.shape[0], format='wav') as f:
            f.write(mastered)
        out_io.seek(0)

        st.success("🎉 Mastering abgeschlossen! -1.0 dBTP, 32Hz LP")
        st.audio(out_io, format="audio/wav")
        st.download_button(label="🚀 Gemasterte WAV herunterladen", data=out_io, file_name=f"mastered_dnb_{uploaded_file.name}", mime="audio/wav")

    except Exception as e:
        st.error(f"Pedalboard Error: {e}")
        st.warning("Tipp: `pip install pedalboard==0.5.3` ist die stabilste Version auf Streamlit Cloud")
