import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, iirpeak, lfilter
import io
import gc

st.set_page_config(page_title="DnB Master Limiter", layout="centered")
st.title("🔊 DnB Master Limiter - Soothe-Light Edition")

def design_highpass_sos(cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def apply_soft_clip(x, threshold=0.99):
    return np.tanh(x / threshold) * threshold

def dynamic_resonance_suppressor(audio, fs, bands=[(250, 3.0), (2500, 2.5), (7000, 2.0)], strength=0.5):
    """
    Soothe-Light: Dynamischer Multiband Cut - Stereo Safe v4.2
    bands: [(freq, Q),...] - Die Problemzonen
    strength: 0.0 = aus, 1.0 = aggressiv
    """
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    output = audio.copy()
    nyq = fs * 0.5

    for freq, q in bands:
        # 1. Bandpass um die Resonanz zu messen
        b, a = iirpeak(freq/nyq, q)
        band_energy = lfilter(b, a, audio, axis=0)

        # 2. Envelope Follower: Wie laut ist das Band gerade?
        envelope = np.abs(band_energy)

        # Attack/Release für jeden Kanal einzeln - Stereo Safe
        attack = 0.003
        release = 0.1
        env_smooth = np.zeros_like(envelope)
        env_smooth[0] = envelope[0] # Init ersten Wert gegen IndexError

        for i in range(1, len(envelope)):
            # Kanal-weise vergleichen mit np.where
            env_smooth[i] = np.where(
                envelope[i] > env_smooth[i-1],
                attack * envelope[i] + (1-attack) * env_smooth[i-1], # Attack
                release * envelope[i] + (1-release) * env_smooth[i-1] # Release
            )

        # 3. Gain Reduction: Je lauter das Band, desto mehr Cut
        threshold_db = -25
        ratio = 3.0 * strength
        threshold_lin = 10**(threshold_db/20.0)

        # Gain Reduction nur da wo nötig, sonst 1.0
        gain_reduction = np.ones_like(env_smooth)
        mask = env_smooth > threshold_lin
        gain_reduction[mask] = (threshold_lin / env_smooth[mask]) ** (1 - 1/ratio)

        # 4. Cut nur auf das Problemband anwenden
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

    st.audio(uploaded_file, format='audio/wav')
    st.write(f"Input: {fs}Hz | {audio.shape[1] if audio.ndim>1 else 1} Kanal | {original_name}")

    col1, col2 = st.columns(2)
    with col1:
        use_soothe = st.checkbox("🎯 Soothe-Light", value=True, help="Dynamisch Härte rausziehen")
        soothe_strength = st.slider("De-Harsh Stärke", 0.0, 1.0, 0.5, 0.1, disabled=not use_soothe)
    with col2:
        use_exciter = st.checkbox("✨ Hybrid Exciter >8kHz", value=True)

    if st.button("Master jetzt"):
        with st.spinner("Mastere... Soothe + Exciter + Limiter"):
            mastered = master_process(audio, fs, cutoff_hz=32,
                                    use_exciter=use_exciter,
                                    use_soothe=use_soothe,
                                    soothe_strength=soothe_strength)

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
