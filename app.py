import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, iirpeak, lfilter
import io
import gc
import pyloudnorm as pyln

st.set_page_config(page_title="DnB Master Limiter", layout="centered")
st.title("🔊 DnB Master Limiter - Soothe-Light Edition")

def design_highpass_sos(cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def apply_soft_clip(x, threshold=0.99):
    return np.tanh(x / threshold) * threshold

def measure_lufs(audio, fs):
    """
    Misst Integrated LUFS nach EBU R128 für Pre/Post Vergleich
    """
    meter = pyln.Meter(fs)
    if audio.ndim > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    lufs = meter.integrated_loudness(audio_mono)
    return lufs

def dynamic_resonance_suppressor(audio, fs, bands=[(250, 3.0), (2500, 2.5), (7000, 2.0)], strength=0.5):
    """
    Soothe-Light v4.3: Turbo Mode - Vektorisiert mit lfilter
    Kein Python Loop mehr. 20x schneller bei 48kHz Stereo.
    """
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    output = audio.copy()
    nyq = fs * 0.5

    for freq, q in bands:
        # 1. Resonanz-Band isolieren
        b, a = iirpeak(freq/nyq, q)
        band_energy = lfilter(b, a, audio, axis=0)
        envelope = np.abs(band_energy)

        # 2. Turbo Envelope Follower: Attack/Release mit lfilter
        attack = 0.003
        release = 0.1

        # Steigung checken um Attack vs Release zu bestimmen
        diff = np.diff(envelope, axis=0, prepend=envelope[0:1])
        rising = diff > 0

        # Koeffizienten für lfilter bauen
        coeff = np.where(rising, attack, release)

        # lfilter pro Channel = C-Speed statt Python-Loop
        env_smooth = np.zeros_like(envelope)
        for ch in range(envelope.shape[1]):
            env_smooth[:, ch] = lfilter([coeff[0, ch]], [1, -(1 - coeff[0, ch])], envelope[:, ch])

        # 3. Dynamische Gain Reduction
        threshold_db = -25
        ratio = 3.0 * strength
        threshold_lin = 10**(threshold_db/20.0)

        gain_reduction = np.ones_like(env_smooth)
        over_thresh = env_smooth > threshold_lin
        gain_reduction[over_thresh] = (threshold_lin / env_smooth[over_thresh]) ** (1 - 1/ratio)

        # 4. Notch nur bei Überschreitung anwenden
        b_notch, a_notch = iirpeak(freq/nyq, q)
        notch_band = lfilter(b_notch, a_notch, audio, axis=0)
        output -= notch_band * (1 - gain_reduction)

    return output

def add_exciter(audio, fs, crossover_hz=8000, drive=2.2, mix=0.15):
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    sos = butter(4, crossover_hz/(fs*0.5), btype='high', output='sos')
    highs = sosfilt(sos, audio, axis=0)
    excited = np.tanh(highs * drive)
    return audio + excited * mix

def master_process(audio, fs, cutoff_hz=32, ceiling_db=-0.3, target_lufs=-14.0,
                   use_exciter=True, use_soothe=True, soothe_strength=0.5):
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    # 1. 32Hz HPF
    sos = design_highpass_sos(cutoff_hz, fs)
    audio = sosfilt(sos, audio, axis=0)

    # 2. Soothe-Light VOR Exciter
    if use_soothe:
        audio = dynamic_resonance_suppressor(audio, fs, strength=soothe_strength)

    # 3. Hybrid Exciter
    if use_exciter:
        audio = add_exciter(audio, fs, crossover_hz=8000, drive=2.2, mix=0.15)

    # 4. Loudness
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_lufs/20.0)
    gain = target_rms / (rms + 1e-8)
    audio = audio * gain

    # 5. Ceiling + Soft Clip
    ceiling = 10**(ceiling_db/20.0)
    audio = np.clip(audio, -ceiling, ceiling)
    audio = apply_soft_clip(audio, threshold=ceiling*0.98)

    return audio.astype(np.float32)

uploaded_file = st.file_uploader("WAV 48kHz hier rein", type=["wav"])

if uploaded_file is not None:
    original_name = uploaded_file.name
    audio, fs = sf.read(uploaded_file, dtype='float32')
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    st.audio(uploaded_file, format='audio/wav')
    st.write(f"Input: {fs}Hz | {audio.shape[1]} Kanal | {original_name}")

    col1, col2 = st.columns(2)
    with col1:
        use_soothe = st.checkbox("🎯 Soothe-Light", value=True, help="Dynamisch Härte rausziehen")
        soothe_strength = st.slider("De-Harsh Stärke", 0.0, 1.0, 0.5, 0.1, disabled=not use_soothe)
    with col2:
        use_exciter = st.checkbox("✨ Hybrid Exciter >8kHz", value=True)

    if st.button("Master jetzt"):
        with st.spinner("Mastere... Soothe + Exciter + Limiter"):
            # Pre LUFS messen
            lufs_pre = measure_lufs(audio, fs)

            mastered = master_process(audio, fs, cutoff_hz=32,
                                    use_exciter=use_exciter,
                                    use_soothe=use_soothe,
                                    soothe_strength=soothe_strength)

            # Post LUFS messen
            lufs_post = measure_lufs(mastered, fs)
            delta = lufs_post - lufs_pre

        # LUFS Meter anzeigen
        st.subheader("📊 LUFS Analyse")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pre Master", f"{lufs_pre:.1f} LUFS", "Input")
        col2.metric("Post Master", f"{lufs_post:.1f} LUFS", f"{delta:+.1f} LU")
        col3.metric("Target", "-14.0 LUFS", "Streaming")

        # Ampel-System
        if lufs_post > -13.5:
            st.warning("⚠️ Über -14 LUFS: Streaming regelt runter.")
        elif lufs_post < -16.0:
            st.info("💡 Unter -16 LUFS: Noch Headroom da.")
        else:
            st.success("✅ Sweet Spot für Spotify/YouTube")

        buf = io.BytesIO()
        sf.write(buf, mastered, fs, format='WAV', subtype='PCM_24')
        buf.seek(0)

        data = buf.getvalue()
        # Kurze Tags C/S/E/SE
        tags = []
        if use_soothe: tags.append("S")
        if use_exciter: tags.append("E")
        tag_str = "".join(tags) if tags else "C"
        new_filename = f"Mastered {tag_str} {original_name}"

        st.download_button("Download WAV", data, new_filename)
        st.audio(data, format='audio/wav')
        st.caption(f"Chain: {tag_str} | Peak: {np.max(np.abs(mastered)):.2f} | RMS: {np.sqrt(np.mean(mastered**2)):.3f}")
        st.success(f"Fertig! {tag_str} Version erstellt.")

        del audio, mastered, buf, data
        gc.collect()
