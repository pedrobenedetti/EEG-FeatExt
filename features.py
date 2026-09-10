"""Estimadores EEG derivados de featureExtraction_1.py de Pedro Benedetti.

La ejecucion y configuracion estan en ejecutar_pipeline.py y config_*.py.
Las formulas originales se conservan, salvo la correccion de la matriz wPLI.
La segmentacion y el alcance temporal se proporcionan explicitamente.
"""

import time
from itertools import permutations
from pathlib import Path
#from tkinter import messagebox

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.special as sp_special
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity_epochs
from scipy.integrate import simpson
from scipy.signal import detrend
from config_regiones import get_regions

mne.set_log_level("WARNING")


def preprocessing_mne(
    path: str,
    file: str,
    excluded: list[str],
    bads: list[str],
    lowpass_cut: int,
    highpass_cut: int,
    raw_plot: bool,
    filtered_plot: bool,
    psd_plot: bool,
    edit_marks: bool,
    interpolate: bool = True,
):
    """Preprocessing of biosemi signals."""
    misc = ["EXG1", "EXG2"]
    eog = []
    base = Path(path) / file
    if base.with_suffix(".bdf").exists():
        raw = mne.io.read_raw_bdf(base.with_suffix(".bdf"), preload=True, verbose=False,
                                  eog=eog, misc=misc, exclude=excluded)
    elif base.with_suffix(".fif").exists():
        raw = mne.io.read_raw_fif(base.with_suffix(".fif"), preload=True, verbose=False)
    else:
        raise FileNotFoundError(f"No existe BDF/FIF: {base}")

    # Mark bad channels
    raw.info["bads"].extend(bads)
    sfreq = raw.info["sfreq"]

    # Apply notch filter (50 Hz for power line noise)
    raw.notch_filter(50)

    # Set montage and reference
    raw.set_montage("biosemi128", on_missing="ignore")
    raw.set_eeg_reference(ref_channels=["EXG1", "EXG2"])

    # Plot raw data if requested
    if raw_plot:
        raw.plot(block=True, title="Carefully identify wrong channels", theme="light")
        plt.show()
        print("Bads Pre-Interpolation:", raw.info["bads"])

    # Interpolate bad channels
    if interpolate and len(raw.info["bads"]) > 0:
        print(f"Interpolating {len(raw.info['bads'])} bad channel(s): {raw.info['bads']}")
        raw.interpolate_bads(reset_bads=True)
        # print("Interpolation completed!")

    # Apply bandpass filter
    raw.filter(lowpass_cut, highpass_cut, l_trans_bandwidth=1, h_trans_bandwidth=1)

    # Resample
    nfreq = 500
    raw.resample(nfreq)

    # Plot filtered data if requested
    if filtered_plot:
        raw.plot(block=True, title="Filtered Signal", theme="light")
        plt.show()

    # Plot PSD if requested
    if psd_plot:
        fig = raw.compute_psd().plot(average=True)
        plt.show()

    # La segmentacion se realiza despues de ICA en una tabla de intervalos.
    if edit_marks:
        raise ValueError("Usar edit_marks=False y segmentacion.py despues de ICA.")
    return raw, raw.copy()


def make_ICA(
    raw,
    method: str,
    n_components: int,
    decim: int,
    random_state: int,
    reject_limit,
    bad_ica_channels,
    plot_ica_topo: bool,
    plot_ica_time: bool,
    plot_raw: bool,
):
    """
    Ajusta ICA sobre canales EEG y aplica la exclusión manual de componentes.
    """

    raw_clean = raw.copy()

    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")

    if method == "infomax":
        ica = ICA(
            n_components=n_components,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=random_state,
            max_iter="auto",
        )
    else:
        ica = ICA(n_components=n_components, method=method, random_state=random_state, max_iter="auto")

    reject = dict(eeg=reject_limit) if reject_limit is not None else None

    ica.fit(raw_clean, picks=picks_eeg, decim=decim, reject=reject)

    if plot_ica_topo:
        ica.plot_components(title="ICA - Componentes Topográficos", cmap="coolwarm")
        plt.show()

    if plot_ica_time:
        ica.plot_sources(raw_clean, title="ICA - Componentes en el Tiempo")
        plt.show()

    if bad_ica_channels is not None:
        ica.exclude = bad_ica_channels

    raw_clean = ica.apply(raw_clean)

    if plot_raw:
        raw.plot(title="Señal cruda antes de ICA")
        raw_clean.plot(title="Señal limpia después de ICA")
        plt.show()

    return ica, raw_clean


# ============================================================================
# POWER SPECTRA - PERIODIC AND APERIODIC
# ============================================================================


def spectral_parametrization(
    raw,
    band_range=(4, 8),
    freq_range=(1, 30),
    status_channel: str | None = None,
    status_start_code: int | None = None,
    status_end_code: int = -1,
    trial_mode: str = "continuous",
    trial_info: dict | None = None,
):
    """Ajusta specparam y conserva la medida espectral del script original.

    spec_period = integral_banda 10**(log10(PSD) - fondo_log10) df
                = integral_banda PSD / fondo_lineal df.
    Integra el espectro observado normalizado por el fondo aperiodico;
    incluye desviaciones residuales aunque no se detecten picos gaussianos.
    No es potencia absoluta en uV**2, ni el area de los picos ajustados.
    El cociente es adimensional y su integral tiene unidades Hz.
    Sin exceso sobre el fondo, la integral se aproxima al ancho integrado
    de la banda, no a cero. Conservamos la mascara inclusiva original.

    Los nombres internos theta_power_* se conservan para compatibilidad;
    el exportador los nombra spec_period_* para cualquier banda.
    El exponente es el parametro positivo chi en offset-chi*log10(f),
    no una medida directa de excitacion/inhibicion.
    """

    print("\n" + "=" * 60)
    print("SPECTRAL PARAMETRIZATION")
    print("=" * 60)
    print(f"Band range: {band_range[0]}-{band_range[1]} Hz")
    print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")

    from specparam import SpectralGroupModel

    def _fit_specparam_on_psd(freqs_arr, psd_arr, ch_names):
        n_channels_local = psd_arr.shape[0]
        theta_powers_local = np.zeros(n_channels_local)
        aperiodic_exponents_local = np.zeros(n_channels_local)
        aperiodic_offsets_local = np.zeros(n_channels_local)

        fg = SpectralGroupModel(peak_width_limits=[2, 8], min_peak_height=0.05, max_n_peaks=6, aperiodic_mode="fixed")
        fg.fit(freqs_arr, psd_arr)
        all_aperiodic = fg.get_params("aperiodic")

        for i in range(n_channels_local):
            try:
                aperiodic_offsets_local[i] = all_aperiodic[i, 0]  # se queda con los offset
                aperiodic_exponents_local[i] = all_aperiodic[i, 1]  # se queda con los exponentes

                aperiodic_log = all_aperiodic[i, 0] - all_aperiodic[i, 1] * np.log10(freqs_arr)  # Calcula aperiodica
                data_log = np.log10(psd_arr[i])
                periodic_log = data_log - aperiodic_log
                periodic_linear = 10**periodic_log
                theta_mask = (freqs_arr >= band_range[0]) & (freqs_arr <= band_range[1])
                theta_powers_local[i] = simpson(periodic_linear[theta_mask], x=freqs_arr[theta_mask])
            except Exception as e:
                print(f"  WARNING: Channel {ch_names[i]} failed: {e}")
                aperiodic_offsets_local[i] = np.nan
                aperiodic_exponents_local[i] = np.nan
                theta_powers_local[i] = np.nan

        return theta_powers_local, aperiodic_exponents_local, aperiodic_offsets_local

    # ---------------------------------------------------------------------
    # Continuous mode (backward compatible)
    # ---------------------------------------------------------------------
    if trial_mode in ("continuous", "all") or status_channel is None:
        spectrum = raw.compute_psd(fmin=freq_range[0], fmax=freq_range[1], verbose=False)
        freqs = spectrum.freqs
        psd_data = spectrum.get_data()

        # print(f"\nFitting {psd_data.shape[0]} channels...")
        theta_powers, aperiodic_exponents, aperiodic_offsets = _fit_specparam_on_psd(freqs, psd_data, raw.ch_names)

        print("Spectral parametrization completed!")
        return {
            "theta_power": theta_powers,
            "aperiodic_exponent": aperiodic_exponents,
            "aperiodic_offset": aperiodic_offsets,
            "channel_names": raw.ch_names,
            "mode": "continuous",
        }

    # ---------------------------------------------------------------------
    # Trial-by-trial mode: apply SAME algorithm on each trial
    # ---------------------------------------------------------------------
    if trial_info is None:
        trial_info = _extract_trials_from_status(
            raw,
            status_channel,
            trial_mode="average",
            status_start_code=status_start_code,
            status_end_code=status_end_code,
        )
    if trial_info is None:
        raise RuntimeError("No trials found for spectral parametrization. Check Status channel and codes.")

    sfreq = float(raw.info["sfreq"])
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False, exclude=[])
    eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

    # Zone aggregation (same zones as connectivity)
    zones, zone_names = _get_zone_definitions()
    zone_indices = _map_zone_indices(eeg_ch_names, zones)

    trial_starts = trial_info["trial_starts"]
    trial_ends = trial_info["trial_ends"]
    trial_vals = trial_info["trial_values"]
    n_trials = trial_info["n_trials"]

    # Allocate
    theta_power_ch = np.full((n_trials, len(eeg_ch_names)), np.nan)
    exponent_ch = np.full((n_trials, len(eeg_ch_names)), np.nan)
    offset_ch = np.full((n_trials, len(eeg_ch_names)), np.nan)

    theta_power_z = np.full((n_trials, len(zone_names)), np.nan)
    exponent_z = np.full((n_trials, len(zone_names)), np.nan)
    offset_z = np.full((n_trials, len(zone_names)), np.nan)

    print(f"\nFitting specparam trial-by-trial: {n_trials} trials")

    for t, (s, e) in enumerate(zip(trial_starts, trial_ends)):
        seg = raw.get_data(picks=eeg_picks, start=int(s), stop=int(e))
        # seg [nchan x nsamples] = [nchan x duration*fs]
        if seg.shape[1] < 10:
            continue

        psd_arr, freqs = mne.time_frequency.psd_array_welch(
            seg, sfreq=sfreq, fmin=freq_range[0], fmax=freq_range[1], verbose=False
        )

        th, ex, off = _fit_specparam_on_psd(freqs, psd_arr, eeg_ch_names)
        theta_power_ch[t, :] = th
        exponent_ch[t, :] = ex
        offset_ch[t, :] = off

        # Aggregate channels -> zones (mean across channels)
        for zi, zn in enumerate(zone_names):
            idxs = zone_indices.get(zn, [])
            if len(idxs) == 0:
                continue
            theta_power_z[t, zi] = np.nanmean(th[idxs])
            exponent_z[t, zi] = np.nanmean(ex[idxs])
            offset_z[t, zi] = np.nanmean(off[idxs])

    print("Spectral parametrization (trial-by-trial) completed!")
    return {
        "theta_power_channels": theta_power_ch,
        "aperiodic_exponent_channels": exponent_ch,
        "aperiodic_offset_channels": offset_ch,
        "theta_power_zones": theta_power_z,
        "aperiodic_exponent_zones": exponent_z,
        "aperiodic_offset_zones": offset_z,
        "zone_names": zone_names,
        "zone_indices": zone_indices,
        "channel_names": eeg_ch_names,
        "mode": "trials",
        "trial_starts": trial_starts,
        "trial_ends": trial_ends,
        "trial_values": trial_vals,
        "sfreq": sfreq,
    }


# ============================================================================
# PHASE CONNECTIVITY OR SYNCHRONY - WEIGHTED PHASE LAG INDEX (wPLI)
# ============================================================================


def phase_connectivity_wpli(
    raw,
    band_range=(4, 8),
    status_channel: str | None = None,
    status_start_code: int | None = None,
    status_end_code: int = -1,
    trial_mode: str = "continuous",
    trial_info: dict | None = None,
    epoch_seconds=5.0,
):
    """
    Calculate weighted Phase Lag Index (wPLI) connectivity between brain zones.

    WHAT IT MEASURES:
    -----------------
    wPLI quantifies phase synchronization between brain regions while being
    robust to volume conduction artifacts.

    Phase synchronization = two regions oscillating at the same frequency
    with a consistent phase relationship (e.g., one leads the other by 45°)

    Values range: 0 to 1
    - 0 = no consistent phase relationship (independent)
    - 1 = perfect phase locking (highly synchronized)
    - Typical values: 0.05-0.3 for EEG
    """

    zones = get_regions()

    zone_names = list(zones.keys())

    print("\n" + "=" * 60)
    print("PHASE CONNECTIVITY ANALYSIS (wPLI)")
    print("=" * 60)
    print(f"Band range: {band_range[0]}-{band_range[1]} Hz")
    # print(f"Zones: {zone_names}")

    # print("\nFiltering signal in band...")
    raw_theta = raw.copy()
    raw_theta.filter(l_freq=band_range[0], h_freq=band_range[1], picks="eeg", method="iir", verbose=False)

    # ---------------------------------------------------------------------
    # Continuous mode (backward compatible): fixed-length epochs
    # ---------------------------------------------------------------------
    if trial_mode in ("continuous", "all") or status_channel is None:
        # print("Creating epochs...")
        events = mne.make_fixed_length_events(raw_theta, duration=2.0)
        epochs = mne.Epochs(raw_theta, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False)
        print(f"  Created {len(epochs)} epochs")

        print("Computing wPLI connectivity...")
        conn = spectral_connectivity_epochs(
            epochs, method="wpli", mode="fourier", fmin=band_range[0], fmax=band_range[1], faverage=True, verbose=False
        )

        wpli_all = _conn_to_square_matrix(conn, n_channels=len(raw_theta.ch_names))

        all_channels = raw_theta.ch_names
        zone_indices = {}
        for zone_name, channels in zones.items():
            indices = [all_channels.index(ch) for ch in channels if ch in all_channels]
            zone_indices[zone_name] = indices

        n_zones = len(zone_names)
        wpli_matrix = np.zeros((n_zones, n_zones))

        print("Calculating zone-to-zone connectivity...")
        for i, zone_i in enumerate(zone_names):
            for j, zone_j in enumerate(zone_names):
                indices_i = zone_indices[zone_i]
                indices_j = zone_indices[zone_j]
                zone_wpli_values = []

                if i == j:
                    for ii, idx_i in enumerate(indices_i):
                        for jj, idx_j in enumerate(indices_j):
                            if ii < jj:
                                zone_wpli_values.append(wpli_all[idx_i, idx_j])
                else:
                    for idx_i in indices_i:
                        for idx_j in indices_j:
                            zone_wpli_values.append(wpli_all[idx_i, idx_j])

                if len(zone_wpli_values) > 0:
                    wpli_matrix[i, j] = np.mean(zone_wpli_values)

        print("Phase connectivity analysis completed!")
        return {
            "wpli_matrix": wpli_matrix,
            "zone_names": zone_names,
            "zone_indices": zone_indices,
            "mode": "continuous",
        }

    # ---------------------------------------------------------------------
    # Trial-by-trial mode: use 5-second subepochs inside each trial
    # ---------------------------------------------------------------------
    if trial_info is None:
        trial_info = _extract_trials_from_status(
            raw_theta,
            status_channel,
            trial_mode="average",
            status_start_code=status_start_code,
            status_end_code=status_end_code,
        )

    if trial_info is None:
        raise RuntimeError("No trials found for wPLI. Check Status channel and codes.")

    sfreq = float(raw_theta.info["sfreq"])
    eeg_picks = mne.pick_types(
        raw_theta.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False, exclude=[]
    )
    eeg_ch_names = [raw_theta.ch_names[i] for i in eeg_picks]
    info_eeg = mne.create_info(eeg_ch_names, sfreq=sfreq, ch_types="eeg")

    zone_indices = _map_zone_indices(eeg_ch_names, zones)
    n_zones = len(zone_names)

    trial_starts = trial_info["trial_starts"]
    trial_ends = trial_info["trial_ends"]
    trial_vals = trial_info["trial_values"]
    n_trials = trial_info["n_trials"]

    wpli_trials = np.full((n_trials, n_zones, n_zones), np.nan)
    wpli_channels = np.full((n_trials, len(eeg_ch_names), len(eeg_ch_names)), np.nan)
    epoch_counts = []

    subepoch_duration_s = float(epoch_seconds)
    subepoch_len = int(round(subepoch_duration_s * sfreq))

    print(f"Computing wPLI trial-by-trial: {n_trials} trials")
    print(f"Using subepochs of {subepoch_duration_s:.1f} s ({subepoch_len} samples)")

    for t, (s, e) in enumerate(zip(trial_starts, trial_ends)):
        seg = raw_theta.get_data(picks=eeg_picks, start=int(s), stop=int(e))
        n_times = seg.shape[1]

        if n_times < subepoch_len:
            print(
                f"  Trial {t + 1}/{n_trials}: too short for 5 s subepochs "
                f"({n_times} samples < {subepoch_len}). Filling with NaN."
            )
            continue

        # Subepochs no solapadas de 5 s dentro del trial original
        starts_sub = np.arange(0, n_times - subepoch_len + 1, subepoch_len, dtype=int)

        if len(starts_sub) == 0:
            print(f"  Trial {t + 1}/{n_trials}: no valid subepochs. Filling with NaN.")
            continue

        ep_list = [seg[:, st : st + subepoch_len] for st in starts_sub]
        ep_data = np.stack(ep_list, axis=0)  # (n_subepochs, n_ch, subepoch_len)

        print(f"  Trial {t + 1}/{n_trials}: {len(starts_sub)} subepochs of {subepoch_duration_s:.1f} s")

        epochs = mne.EpochsArray(ep_data, info_eeg, tmin=0.0, baseline=None, verbose=False)

        conn = spectral_connectivity_epochs(
            epochs, method="wpli", mode="fourier", fmin=band_range[0], fmax=band_range[1], faverage=True, verbose=False
        )

        wpli_all = _conn_to_square_matrix(conn, n_channels=len(eeg_ch_names))
        wpli_channels[t] = wpli_all
        epoch_counts.append(len(starts_sub))

        for i, zone_i in enumerate(zone_names):
            for j, zone_j in enumerate(zone_names):
                indices_i = zone_indices.get(zone_i, [])
                indices_j = zone_indices.get(zone_j, [])
                zone_vals = []

                if i == j:
                    for ii, idx_i in enumerate(indices_i):
                        for jj, idx_j in enumerate(indices_j):
                            if ii < jj:
                                zone_vals.append(wpli_all[idx_i, idx_j])
                else:
                    for idx_i in indices_i:
                        for idx_j in indices_j:
                            zone_vals.append(wpli_all[idx_i, idx_j])

                if len(zone_vals) > 0:
                    wpli_trials[t, i, j] = float(np.nanmean(zone_vals))

    print("Phase connectivity (trial-by-trial) completed!")
    return {
        "wpli_trials": wpli_trials,
        "wpli_channels": wpli_channels,
        "channel_names": eeg_ch_names,
        "epoch_counts": epoch_counts,
        "zone_names": zone_names,
        "zone_indices": zone_indices,
        "mode": "trials",
        "trial_starts": trial_starts,
        "trial_ends": trial_ends,
        "trial_values": trial_vals,
        "sfreq": sfreq,
    }


# ============================================================================
# PATTERNS CONNECTIVITY - WEIGHTED SYMBOLIC MUTUAL INFORMATION (wSMI)
# ============================================================================


def patterns_connectivity_wsmi(
    raw,
    band_range=(4, 8),
    embedding_dim=3,
    tau=None,  # None = auto-calculate
    n_channels_per_zone=None,
    status_channel="Status",  # Canal de status
    status_start_code=1,  # Código de inicio (stim)
    status_end_code=0,  # Código de fin (resp o 0)
    # 'all' para continuo; en modo trials, filtra por valor si es numérico
    trial_mode="average",
    debug_first_pair=False,
    trial_info=None,  # Limites [inicio, fin) validados por segmentacion.py
):
    """
    Calculate weighted Symbolic Mutual Information (wSMI) connectivity.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data

    band_range : tuple of (float, float), default=(4, 8)
        Frequency range for  band in Hz

    embedding_dim : int, default=3
        Embedding dimension (pattern length)
        embedding_dim=3 → 6 possible patterns (3! = 6)

    tau : int or None, default=None
        Time delay between samples in pattern
        If None, calculated as: round(sfreq / embedding_dim / theta_max)
        MATLAB formula: round(fs / kernel / freq_max)

    n_channels_per_zone : int, default=5
        Number of channels to subsample per zone
        Lower = faster but less representative

    status_channel : str, default='Status'
        Name of the status/trigger channel that marks trial events
        Set to None if no trials (continuous recording)

    trial_mode : str or int, default='all'
        How to handle trials:
        - 'all': Calculate wSMI over entire continuous signal (fast)
        - 'average': Calculate per trial, then average (slower, preserves variability)
        - int (e.g., 1, 2, 3): Calculate only for trials with this status value

    debug_first_pair : bool, default=False
        Print detailed debug for first channel pair

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'wsmi_matrix': array (n_zones, n_zones) or (n_zones, n_zones, n_trials)
            If trial_mode='all': (n_zones, n_zones)
            If trial_mode='average' or int: (n_zones, n_zones, n_trials)
        - 'zone_names': list of str
        - 'trial_mode': str or int (what was used)
        - 'n_trials': int (number of trials, or 0 if continuous)
        - 'tau': int (tau value used)

    Examples
    --------
    # Continuous (no trials)
    >>> results = patterns_connectivity_wsmi(raw, trial_mode='all')
    >>> wsmi = results['wsmi_matrix']  # Shape: (5, 5)

    # Average over all trials
    >>> results = patterns_connectivity_wsmi(raw, trial_mode='average')
    >>> wsmi_avg = results['wsmi_matrix'].mean(axis=2)  # Shape: (5, 5)
    >>> wsmi_std = results['wsmi_matrix'].std(axis=2)   # Variability

    # Only correct trials (status = 1)
    >>> results = patterns_connectivity_wsmi(raw, trial_mode=1)
    >>> wsmi_correct = results['wsmi_matrix']  # Shape: (5, 5, n_correct_trials)

    Notes
    -----
    - Based on King et al. (2013) implementation
    - Tau calculation follows MATLAB: tau = round(sfreq / embedding_dim / freq_max)
    - Weight matrix: diagonal and anti-diagonal set to 0 (MATLAB style)
    - Computation time: ~5-10 minutes per trial, 30-60 minutes total for 'average' mode
    """

    print("\n" + "=" * 60)
    print("PATTERNS CONNECTIVITY ANALYSIS (wSMI)")
    print("=" * 60)

    # Define zones (same as wPLI)
    zones = get_regions()

    zone_names = list(zones.keys())

    # Calculate tau dynamically (MATLAB formula)
    sfreq = raw.info["sfreq"]
    if tau is None:
        tau = round(sfreq / embedding_dim / band_range[1])
        print(f"Auto-calculated tau: {tau} samples")
        print("  (formula: round(sfreq / embedding_dim / theta_max))")
        # print(
        #   f"  (sfreq={sfreq}, embedding_dim={embedding_dim}, theta_max={band_range[1]})")
    else:
        print(f"Using provided tau: {tau} samples")

    print(f"Band range: {band_range[0]}-{band_range[1]} Hz")
    # print(f"Embedding dimension: {embedding_dim}")
    chpz = "ALL" if (n_channels_per_zone is None or n_channels_per_zone <= 0) else str(n_channels_per_zone)
    # print(f"Channels per zone: {chpz}")
    # print(f"Trial mode: {trial_mode}")

    # Filter signal
    # print("\nFiltering signal in band...")
    # -------------------------------------------------------------------------
    # Channel selection (EEG only) - keep indices consistent throughout
    # -------------------------------------------------------------------------
    sfreq = float(raw.info["sfreq"])
    try:
        eeg_picks = mne.pick_types(
            raw.info,
            eeg=True,
            meg=False,
            eog=False,
            stim=False,
            misc=False,
            ecg=False,
            emg=False,
            seeg=False,
            exclude="bads",
        )
    except Exception:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels were found after applying picks='eeg'.")

    eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

    # Auto-calculate tau if not provided (MATLAB-style)
    if tau is None:
        tau = int(round(sfreq / embedding_dim / float(band_range[1])))

    print(f"  Using tau={tau} (sfreq={sfreq:.2f} Hz, m={embedding_dim}, fmax={band_range[1]})")

    # EEG-only data (filtering strategy depends on trial_mode)
    data_eeg = raw.get_data(picks=eeg_picks)

    # Extract trials if status channel exists

    trials_info = trial_info
    if trial_info is not None:
        pass
    elif status_channel is not None and status_channel in raw.ch_names:
        print(f"\nExtracting trials from '{status_channel}' channel...")
        trials_info = _extract_trials_from_status(raw, status_channel, trial_mode, status_start_code, status_end_code)

        if trials_info is not None:
            print(f"  Total trials found: {trials_info['n_trials']}")
            if trial_mode != "all":
                print(f"  Trials matching condition '{trial_mode}': {len(trials_info['trial_indices'])}")
        else:
            print("  WARNING: No trials found or extracted. Using continuous mode.")
            trial_mode = "all"
    else:
        if status_channel is not None:
            print(f"\nWARNING: Status channel '{status_channel}' not found.")
        print("Using continuous signal mode (trial_mode='all')")
        trial_mode = "all"

    # EEG channel universe (MATLAB-like)
    data = data_eeg
    all_channels = eeg_ch_names

    # Subsample channels per zone
    zone_indices = {}
    zone_channels_selected = {}

    for zone_name, channels in zones.items():
        available_channels = [ch for ch in channels if ch in all_channels]

        # Channel selection strategy
        n_available = len(available_channels)
        if (n_channels_per_zone is None) or (n_channels_per_zone <= 0) or (n_available <= n_channels_per_zone):
            # MATLAB-like: use ALL available channels in the zone
            selected = available_channels
        else:
            # Optional speed-up: evenly subsample a fixed number per zone (legacy behavior)
            step = n_available / n_channels_per_zone
            selected_idx = [int(i * step) for i in range(n_channels_per_zone)]
            selected = [available_channels[i] for i in selected_idx]

        indices = [all_channels.index(ch) for ch in selected]
        zone_indices[zone_name] = indices
        zone_channels_selected[zone_name] = selected

        # print(
        #     f"  {zone_name}: {len(selected)}/{len(available_channels)} channels selected")

    # Create index mapping
    channel_to_zone = {}
    flat_indices = []
    for zone_name, indices in zone_indices.items():
        for idx in indices:
            channel_to_zone[idx] = zone_name
            flat_indices.append(idx)

    n_selected = len(flat_indices)
    total_pairs = n_selected * (n_selected - 1) // 2

    # print(f"\nTotal channels to analyze: {n_selected}")
    # print(f"Total pairs to compute: {total_pairs}")

    # Calculate wSMI based on trial_mode
    if trial_mode == "all" or trials_info is None:
        # Continuous mode: single calculation
        # print("\n[MODE: CONTINUOUS] Calculating wSMI over entire signal...")
        # print("This may take 5-15 minutes...")

        # FIR, zero-phase filtering (EEGLAB pop_eegfiltnew-like)
        data_filt = mne.filter.filter_data(
            data,
            sfreq=sfreq,
            l_freq=band_range[0],
            h_freq=band_range[1],
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=False,
        )

        wsmi_all = _calculate_wsmi_matrix(data_filt, flat_indices, embedding_dim, tau, debug_first_pair, total_pairs)

        # Create zone-to-zone matrix
        wsmi_matrix = _aggregate_to_zones(wsmi_all, flat_indices, channel_to_zone, zone_names)

        n_trials = 0

    else:
        # Trial mode: calculate per trial
        trial_indices = trials_info["trial_indices"]
        trial_starts = trials_info["trial_starts"]
        trial_ends = trials_info["trial_ends"]
        n_trials = len(trial_indices)

        print(f"\n[MODE: TRIAL-BY-TRIAL] Calculating wSMI for {n_trials} trials...")
        # print(f"Estimated time: {n_trials * 5}-{n_trials * 10} minutes...")

        # Initialize 3D matrix
        n_zones = len(zone_names)
        wsmi_matrix = np.zeros((n_zones, n_zones, n_trials))
        wsmi_channels = np.full((n_trials, n_selected, n_selected), np.nan)

        for trial_idx in range(n_trials):
            start_sample = trial_starts[trial_idx]
            end_sample = trial_ends[trial_idx]
            trial_data = data[:, start_sample:end_sample]

            # FIR, zero-phase filtering per trial
            # Skip trials that are too short for embedding
            min_needed = tau * (embedding_dim - 1) + 1
            if trial_data.shape[1] < (min_needed + 1):
                print(
                    f"    WARNING: Trial too short for wSMI (n={trial_data.shape[1]}, needed>{min_needed}). Filling with NaN."
                )
                wsmi_matrix[:, :, trial_idx] = np.nan
                continue

            try:
                trial_data_filt = mne.filter.filter_data(
                    trial_data,
                    sfreq=sfreq,
                    l_freq=band_range[0],
                    h_freq=band_range[1],
                    method="fir",
                    phase="zero",
                    fir_design="firwin",
                    verbose=False,
                )
            except Exception as e:
                print(f"    WARNING: FIR filtering failed for trial {trial_idx + 1}: {e}. Filling with NaN.")
                wsmi_matrix[:, :, trial_idx] = np.nan
                continue

            # print(f"\n  Trial {trial_idx + 1}/{n_trials} (samples {start_sample}-{end_sample})...")

            # Calculate for this trial
            wsmi_all_trial = _calculate_wsmi_matrix(
                trial_data_filt,
                flat_indices,
                embedding_dim,
                tau,
                debug=(debug_first_pair and trial_idx == 0),
                total_pairs=total_pairs,
            )

            wsmi_channels[trial_idx] = wsmi_all_trial
            # Aggregate to zones
            wsmi_zones_trial = _aggregate_to_zones(wsmi_all_trial, flat_indices, channel_to_zone, zone_names)

            wsmi_matrix[:, :, trial_idx] = wsmi_zones_trial

    # Print results
    if trial_mode == "all":
        print("\nwSMI Connectivity Matrix (Zone-to-Zone, Continuous):")
        print("=" * 60)
        _print_matrix(wsmi_matrix, zone_names)
    else:
        # print(f"\nwSMI Connectivity Matrix (Zone-to-Zone, Averaged over {n_trials} trials):")
        # print("=" * 60)
        wsmi_avg = wsmi_matrix.mean(axis=2)
        _print_matrix(wsmi_avg, zone_names)

        # print("\nStandard deviation across trials:")
        wsmi_std = wsmi_matrix.std(axis=2)
        _print_matrix(wsmi_std, zone_names)

    print("\nPattern connectivity analysis completed!")

    results = {
        "wsmi_matrix": wsmi_matrix,
        "wsmi_channels": wsmi_all if trial_mode == "all" else wsmi_channels,
        "channel_names": [all_channels[i] for i in flat_indices],
        "wsmi_avg": (wsmi_matrix if trial_mode == "all" else np.nanmean(wsmi_matrix, axis=2)),
        "wsmi_std": (np.zeros_like(wsmi_matrix) if trial_mode == "all" else np.nanstd(wsmi_matrix, axis=2)),
        "zone_names": zone_names,
        "zone_indices": zone_indices,
        "trial_mode": trial_mode,
        "n_trials": n_trials,
        "tau": tau,
        "embedding_dim": embedding_dim,
    }

    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_zone_definitions():
    """Return the canonical 5-zone definition used across all features."""
    zone_names = list(get_regions())
    zones = get_regions()
    return zones, zone_names


def _map_zone_indices(ch_names, zones):
    """Map a list of channel names to zone indices."""
    zone_indices = {}
    for zone_name, channels in zones.items():
        zone_indices[zone_name] = [ch_names.index(ch) for ch in channels if ch in ch_names]
    return zone_indices


def _conn_to_square_matrix(conn, n_channels: int):
    """wPLI todos los pares, una banda. Conserva indices y magnitudes MNE."""
    if conn.indices is not None and not (isinstance(conn.indices, str) and conn.indices == "all"):
        raise ValueError("Se requieren todos los pares (indices=None).")
    data = np.asarray(conn.get_data(output="dense"))
    if data.shape != (n_channels, n_channels, 1):
        raise ValueError(f"Forma wPLI inesperada: {data.shape}")
    matrix = data[:, :, 0]
    if not np.all(np.isfinite(matrix)):
        raise ValueError("wPLI contiene valores no finitos.")
    if np.any(np.triu(matrix, 1) != 0):
        raise ValueError("Formato MNE inesperado: triangulo superior no vacio.")
    lower = np.tril(matrix, -1)
    if np.any((lower < 0) | (lower > 1)):
        raise ValueError("wPLI fuera de [0, 1].")
    return lower + lower.T


def _flatten_symmetric_zone_matrix(mat, zone_names, prefix):
    """Flatten upper triangle (i<j) of a symmetric zone matrix into columns."""
    out = {}
    n = len(zone_names)
    for i in range(n):
        for j in range(i + 1, n):
            out[f"{prefix}_{zone_names[i]}__{zone_names[j]}"] = float(mat[i, j])
    return out


def _flatten_directed_zone_matrix(mat, zone_names, prefix, include_self=False):
    """Flatten a directed zone matrix into columns (i->j)."""
    out = {}
    n = len(zone_names)
    for i in range(n):
        for j in range(n):
            if (not include_self) and i == j:
                continue
            out[f"{prefix}_{zone_names[i]}__{zone_names[j]}"] = float(mat[i, j])
    return out


def _build_trialwise_rows(
    subject: str,
    band: str,
    trial_info: dict,
    spectral_results: dict | None = None,
    wpli_results: dict | None = None,
    pe_results: dict | None = None,
    lzc_results: dict | None = None,
    te_results: dict | None = None,
    wsmi_results: dict | None = None,
):
    """Create one dict row per trial, with audit columns + feature columns."""
    sfreq = float(trial_info.get("sfreq", np.nan))
    starts = trial_info["trial_starts"]
    ends = trial_info["trial_ends"]
    conds = trial_info["trial_values"]
    n_trials = trial_info["n_trials"]

    rows = []
    for t in range(n_trials):
        s = int(starts[t])
        e = int(ends[t])
        cond = int(conds[t])
        row = {
            "subject": subject,
            "band": band,
            "condition": cond,
            "trial_index": int(t),
            "start_sample": s,
            "end_sample": e,
            "n_samples": int(max(e - s, 0)),
            "start_s": (s / sfreq) if np.isfinite(sfreq) else np.nan,
            "end_s": (e / sfreq) if np.isfinite(sfreq) else np.nan,
            "sfreq": sfreq,
        }

        # Spectral parametrization (zones)
        if spectral_results is not None and spectral_results.get("mode") == "trials":
            z = spectral_results["zone_names"]
            theta = spectral_results["theta_power_zones"][t]
            exp = spectral_results["aperiodic_exponent_zones"][t]
            off = spectral_results["aperiodic_offset_zones"][t]
            for zi, zn in enumerate(z):
                row[f"spec_period_{zn}"] = float(theta[zi])
                row[f"spec_exp_{zn}"] = float(exp[zi])
                row[f"spec_off_{zn}"] = float(off[zi])

        # wPLI (zones, symmetric)
        if wpli_results is not None and wpli_results.get("mode") == "trials":
            zone_names = wpli_results["zone_names"]
            mat = wpli_results["wpli_trials"][t]
            row.update(_flatten_symmetric_zone_matrix(mat, zone_names, prefix="wpli"))

        # Permutation entropy (zones x trials)
        if pe_results is not None and "pe_matrix_zones" in pe_results:
            zone_names = pe_results["zone_names"]
            pe_vals = pe_results["pe_matrix_zones"][:, t]
            for zi, zn in enumerate(zone_names):
                row[f"pe_{zn}"] = float(pe_vals[zi])

        # LZC (zones x trials)
        if lzc_results is not None and "lzc_matrix_zones" in lzc_results:
            zone_names = lzc_results["zone_names"]
            lzc_vals = lzc_results["lzc_matrix_zones"][:, t]
            for zi, zn in enumerate(zone_names):
                row[f"lzc_{zn}"] = float(lzc_vals[zi])

        # Transfer entropy (trials x zones x zones)
        if te_results is not None and "te_mean_lag" in te_results:
            zone_names = te_results["zone_names"]
            te_mat = te_results["te_mean_lag"][t]
            row.update(_flatten_directed_zone_matrix(te_mat, zone_names, prefix="te", include_self=False))

        # wSMI (zones x zones x trials) if available
        if wsmi_results is not None and "wsmi_matrix" in wsmi_results and wsmi_results.get("trial_mode") != "all":
            zone_names = wsmi_results["zone_names"]
            wsmi_mat = wsmi_results["wsmi_matrix"][:, :, t]
            row.update(_flatten_symmetric_zone_matrix(wsmi_mat, zone_names, prefix="wsmi"))

        rows.append(row)

    return rows




def _extract_trials_from_status(raw, status_channel, trial_mode, status_start_code=1, status_end_code=0):
    """
    Extract trial boundaries from a status/trigger channel.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object containing the status channel.
    status_channel : str
        Name of the status/trigger channel.
    trial_mode : str or int
        'average', 'all', or a numeric condition selector (kept for backward compatibility).
    status_start_code : int, default=1
        Code that marks the START of a trial (e.g., stimulus marker).
    status_end_code : int, default=0
        How to find the END of a trial:
          - 0  : end at transition into 0 from non-zero (classic 0->nonzero->0 blocks)
          - >0 : end at transition into this explicit code
          - -1 : end at the next non-zero impulse after the start

    Returns
    -------
    trials_info : dict or None
        Dictionary with:
          - 'trial_indices': indices of selected trials
          - 'trial_starts' : sample indices for trial starts
          - 'trial_ends'   : sample indices for trial ends
          - 'trial_values' : values associated with each trial (start code by default)
          - 'trial_end_values': end marker values (useful when status_end_code == -1)
          - 'n_trials'     : total number of selected trials
    """
    # Get status channel data
    status_idx = raw.ch_names.index(status_channel)
    status_data = raw.get_data(picks=[status_idx])[0]
    status_int = np.round(status_data).astype(int)

    n = status_int.shape[0]
    if n < 2:
        return None

    # Impulses are detected as transitions from 0 -> non-zero
    impulse_mask = (status_int[1:] != 0) & (status_int[:-1] == 0)
    impulse_idxs = np.where(impulse_mask)[0] + 1
    if impulse_idxs.size == 0:
        return None
    impulse_vals = status_int[impulse_idxs]

    # Start impulses
    # - If status_start_code is None: use *all* non-zero impulses as trial starts
    # - Else: use impulses matching the provided code
    if status_start_code is None:
        start_positions = np.arange(impulse_vals.size)
    else:
        start_positions = np.where(impulse_vals == int(status_start_code))[0]

    if start_positions.size == 0:
        return None

    events = []
    start_values = []
    end_values = []

    if int(status_end_code) == -1:
        # End at the NEXT non-zero impulse after the start impulse
        for p in start_positions:
            if p + 1 >= impulse_idxs.size:
                continue
            s = int(impulse_idxs[p])
            e = int(impulse_idxs[p + 1])
            if e > s:
                events.append((s, e))
                start_values.append(int(impulse_vals[p]))
                end_values.append(int(impulse_vals[p + 1]))
    elif int(status_end_code) == 0:
        # End at transition into 0 from non-zero (classic block end)
        end_mask = (status_int[1:] == 0) & (status_int[:-1] != 0)
        end_idxs = np.where(end_mask)[0] + 1
        if end_idxs.size == 0:
            return None

        # Detect STARTS as transitions into status_start_code (robust to pulses)
        start_mask = (status_int[1:] == int(status_start_code)) & (status_int[:-1] != int(status_start_code))
        start_idxs = np.where(start_mask)[0] + 1
        if start_idxs.size == 0:
            return None

        end_ptr = 0
        for s in start_idxs:
            while end_ptr < len(end_idxs) and end_idxs[end_ptr] <= s:
                end_ptr += 1
            if end_ptr >= len(end_idxs):
                break
            e = int(end_idxs[end_ptr])
            if e > s:
                events.append((int(s), e))
                start_values.append(int(status_int[s]))
                end_values.append(int(status_int[e - 1]) if e - 1 >= 0 else 0)
            end_ptr += 1
    else:
        # End at an explicit marker code (use impulses: 0->nonzero transitions)
        end_positions = np.where(impulse_vals == int(status_end_code))[0]
        if end_positions.size == 0:
            return None

        # Pair each start impulse with the first subsequent end impulse
        end_ptr = 0
        end_positions_sorted = end_positions  # already sorted by time
        for p in start_positions:
            # advance end_ptr until we find an end after this start
            while (
                end_ptr < end_positions_sorted.size and impulse_idxs[end_positions_sorted[end_ptr]] <= impulse_idxs[p]
            ):
                end_ptr += 1
            if end_ptr >= end_positions_sorted.size:
                break
            s = int(impulse_idxs[p])
            e_pos = int(end_positions_sorted[end_ptr])
            e = int(impulse_idxs[e_pos])
            if e > s:
                events.append((s, e))
                start_values.append(int(impulse_vals[p]))
                end_values.append(int(impulse_vals[e_pos]))
            end_ptr += 1

    if len(events) == 0:
        return None

    # Trial selection (kept for backward compatibility):
    # - 'average' and 'all' keep all events
    # - numeric selects trials with matching start value
    if trial_mode in ("average", "all"):
        selected_indices = list(range(len(events)))
    else:
        try:
            cond = int(trial_mode)
        except Exception:
            cond = None

        if cond is None:
            selected_indices = list(range(len(events)))
        else:
            selected_indices = [i for i, v in enumerate(start_values) if v == cond]

        if len(selected_indices) == 0:
            print(f"  WARNING: No trials found with start value {trial_mode}")
            return None

    trial_starts = [events[i][0] for i in selected_indices]
    trial_ends = [events[i][1] for i in selected_indices]
    trial_vals = [start_values[i] for i in selected_indices]
    trial_end_vals = [end_values[i] for i in selected_indices]

    return {
        "trial_indices": selected_indices,
        "trial_starts": trial_starts,
        "trial_ends": trial_ends,
        "trial_values": trial_vals,
        "trial_end_values": trial_end_vals,
        "n_trials": len(selected_indices),
    }


# -----------------------------------------------------------------------------
# Efficient wSMI implementation (symbolize once per channel per trial)
# -----------------------------------------------------------------------------
_WSMI_CACHE = {}


def _get_wsmi_lookup_and_weights(embedding_dim: int):
    """Precompute symbol lookup (MATLAB-style) and wSMI weights for a given embedding_dim."""
    if embedding_dim in _WSMI_CACHE:
        return _WSMI_CACHE[embedding_dim]

    # Generate all permutations (1-based, sorted like MATLAB)
    all_perms = list(permutations(range(1, embedding_dim + 1)))
    all_perms = sorted(all_perms)

    # Flip second half (mirror structure like MATLAB code in this script)
    n = len(all_perms)
    for i in range(n // 2):
        all_perms[n - i - 1] = tuple(reversed(all_perms[i]))

    # Decimal encoding: [1,2,3] -> 123
    adjust = np.array([10 ** (embedding_dim - 1 - i) for i in range(embedding_dim)], dtype=int)
    codes = np.array([int(np.sum(np.array(perm) * adjust)) for perm in all_perms], dtype=int)

    n_symbols = int(np.prod(np.arange(1, embedding_dim + 1)))  # factorial(m)

    # Vectorized code->index lookup table
    max_code = int("".join(str(i) for i in range(embedding_dim, 0, -1)))  # e.g., 321 for m=3
    lookup = np.full(max_code + 1, -1, dtype=int)
    for idx, c in enumerate(codes):
        if c <= max_code:
            lookup[c] = idx

    # weights: ones with diagonal and anti-diagonal set to 0
    weights = np.ones((n_symbols, n_symbols), dtype=float)
    np.fill_diagonal(weights, 0.0)
    for i in range(n_symbols):
        weights[i, n_symbols - 1 - i] = 0.0

    _WSMI_CACHE[embedding_dim] = (lookup, adjust, n_symbols, weights)
    return _WSMI_CACHE[embedding_dim]


def _symbolize_all_channels(data_sel: np.ndarray, embedding_dim: int, tau: int, lookup: np.ndarray, adjust: np.ndarray):
    """Symbolize all channels (rows) once, MATLAB-compatible ordinal patterns."""
    n_ch, n_t = data_sel.shape
    n_samples = n_t - tau * (embedding_dim - 1)
    if n_samples <= 0:
        return None

    symbols = np.zeros((n_ch, n_samples), dtype=np.int16)

    # Build delayed embedding per channel and argsort along pattern dimension
    # embedding_dim is small (typically 3), so a Python loop over channels is OK.
    for ch in range(n_ch):
        x = data_sel[ch]
        emb = np.stack([x[i * tau : i * tau + n_samples] for i in range(embedding_dim)], axis=1)
        perm = np.argsort(emb, axis=1, kind="mergesort") + 1  # 1..m
        code = (perm * adjust).sum(axis=1).astype(int)

        # Map decimal code -> symbol index
        # If for some reason a code is out of range, default to 0.
        sym = np.zeros_like(code, dtype=np.int16)
        in_range = code < lookup.shape[0]
        sym[in_range] = lookup[code[in_range]].astype(np.int16)
        sym[~in_range] = 0
        # Replace any unmapped (-1) with 0 (shouldn't happen)
        sym[sym < 0] = 0
        symbols[ch, :] = sym

    return symbols


def _marginals_from_symbols(symbols: np.ndarray, n_symbols: int):
    """Compute marginal probabilities per channel from symbol sequences."""
    n_ch, n_samp = symbols.shape
    marg = np.zeros((n_ch, n_symbols), dtype=float)
    for ch in range(n_ch):
        marg[ch] = np.bincount(symbols[ch].astype(int), minlength=n_symbols) / float(n_samp)
    return marg


def _wsmi_from_symbols(
    sym_x: np.ndarray, sym_y: np.ndarray, p_x: np.ndarray, p_y: np.ndarray, weights: np.ndarray, n_symbols: int
):
    """Compute wSMI between two channels given their symbol sequences and marginals."""
    L = min(len(sym_x), len(sym_y))
    if L <= 0:
        return np.nan

    # Joint probability via 1D bincount
    joint = np.bincount(
        (sym_x[:L].astype(int) * n_symbols + sym_y[:L].astype(int)), minlength=n_symbols * n_symbols
    ).reshape(n_symbols, n_symbols) / float(L)

    denom = np.outer(p_x, p_y)
    mask = (joint > 0) & (denom > 0) & (weights > 0)

    if not np.any(mask):
        return 0.0

    wsmi = np.sum(weights[mask] * joint[mask] * np.log(joint[mask] / denom[mask]))
    wsmi = wsmi / np.log(n_symbols)
    return float(wsmi)


def _calculate_wsmi_matrix(data, flat_indices, embedding_dim, tau, debug, total_pairs):
    """Calculate wSMI for all channel pairs (symbolize once per channel)."""
    # Select only the channels that belong to the defined zones (flat_indices)
    data_sel = data[np.array(flat_indices), :]

    lookup, adjust, n_symbols, weights = _get_wsmi_lookup_and_weights(embedding_dim)

    symbols = _symbolize_all_channels(data_sel, embedding_dim, tau, lookup, adjust)
    if symbols is None:
        return np.full((len(flat_indices), len(flat_indices)), np.nan)

    marginals = _marginals_from_symbols(symbols, n_symbols)

    n_selected = len(flat_indices)
    wsmi_all = np.zeros((n_selected, n_selected), dtype=float)

    pair_count = 0
    for i in range(n_selected):
        for j in range(i + 1, n_selected):
            debug_this = pair_count == 0 and debug
            if debug_this:
                print("\n>>> wSMI DEBUG (first pair) <<<")
                print(f"  embedding_dim={embedding_dim} | tau={tau} | n_symbols={n_symbols}")
                print(f"  samples used for symbols: {symbols.shape[1]}")
                # Simple sanity checks
                same = _wsmi_from_symbols(symbols[i], symbols[i], marginals[i], marginals[i], weights, n_symbols)
                shuf_sym = np.random.permutation(symbols[j])
                p_shuf = np.bincount(shuf_sym.astype(int), minlength=n_symbols) / float(symbols.shape[1])
                shuf = _wsmi_from_symbols(symbols[i], shuf_sym, marginals[i], p_shuf, weights, n_symbols)
                print(f"  wSMI(x,x)          = {same:.6f}")
                print(f"  wSMI(x,y_shuffled) = {shuf:.6f}")

            wsmi_val = _wsmi_from_symbols(symbols[i], symbols[j], marginals[i], marginals[j], weights, n_symbols)

            wsmi_all[i, j] = wsmi_val
            wsmi_all[j, i] = wsmi_val

            pair_count += 1
            # if pair_count % 200 == 0:
            #     progress = (pair_count / total_pairs) * 100
            #     print(f"    Progress: {pair_count}/{total_pairs} pairs ({progress:.1f}%)")

    return wsmi_all


def _aggregate_to_zones(wsmi_all, flat_indices, channel_to_zone, zone_names):
    """Aggregate channel-level wSMI to zone-level."""
    n_zones = len(zone_names)
    wsmi_matrix = np.zeros((n_zones, n_zones))

    for i, zone_i in enumerate(zone_names):
        for j, zone_j in enumerate(zone_names):
            local_indices_i = [k for k, idx in enumerate(flat_indices) if channel_to_zone[idx] == zone_i]
            local_indices_j = [k for k, idx in enumerate(flat_indices) if channel_to_zone[idx] == zone_j]

            zone_wsmi_values = []

            if i == j:
                # Intra-zone: use upper triangle only (like MATLAB)
                for ii in local_indices_i:
                    for jj in local_indices_j:
                        if ii < jj:
                            zone_wsmi_values.append(wsmi_all[ii, jj])
            else:
                # Inter-zone: use all pairs
                for ii in local_indices_i:
                    for jj in local_indices_j:
                        zone_wsmi_values.append(wsmi_all[ii, jj])

            if len(zone_wsmi_values) > 0:
                wsmi_matrix[i, j] = np.mean(zone_wsmi_values)

    return wsmi_matrix


def _print_matrix(matrix, zone_names):
    """Pretty print connectivity matrix."""
    n_zones = len(zone_names)
    # print(f"{'':15s}", end='')
    for name in zone_names:
        # print(f"{name:>12s}", end='')
        continue
    # print()

    # for i, name_i in enumerate(zone_names):
    #     print(f"{name_i:15s}", end='')
    #     for j in range(n_zones):
    #         print(f"{matrix[i, j]:12.4f}", end='')
    #     print()


# ============================================================================
# COMPLEXITY/ENTROPY - LEMPEL-ZIV COMPLEXITY (LZC)
# ============================================================================


def _lz78_complexity(binary_seq):
    """
    Compute LZ78 complexity using dictionary algorithm.

    WHAT IT DOES:
    -------------
    Counts the number of unique patterns (substrings) in a binary sequence.
    More unique patterns = higher complexity = less predictable signal.

    ALGORITHM:
    ----------
    1. Start with empty dictionary and empty current word
    2. Read sequence character by character
    3. Add character to current word
    4. If current word NOT in dictionary:
       - Add it to dictionary
       - Reset current word to empty
    5. Repeat until end of sequence
    6. Dictionary size = complexity measure

    Example:
    --------
    Sequence: "010101010101" (repeating pattern)
    Dictionary: {'0', '01', '010'} → size = 3 (low complexity)

    Sequence: "011010001101" (random-looking)
    Dictionary: {'0', '01', '1', '10', '00', '011'...} → size = 8 (high complexity)

    Parameters
    ----------
    binary_seq : str or array-like
        Binary sequence (string of '0' and '1')

    Returns
    -------
    c : int
        Dictionary length (raw LZ78 complexity)
        Higher values = more complex/unpredictable signal

    Notes
    -----
    Based on LZ78 algorithm by Lempel & Ziv (1978)
    Used in data compression (zip, gzip)
    """

    # Convert to string if array
    if not isinstance(binary_seq, str):
        binary_seq = "".join(str(int(b)) for b in binary_seq)

    # LZ78 algorithm: build dictionary of unique substrings
    dictionary = set()  # Set of all unique patterns seen
    w = ""  # Current word being built

    # Scan through sequence character by character
    for ch in binary_seq:
        w = w + ch  # Append character to current word
        if w not in dictionary:  # New pattern found
            dictionary.add(w)  # Add to dictionary
            w = ""  # Reset current word

    # Return dictionary size = number of unique patterns
    return len(dictionary)


def lempel_ziv_complexity(
    raw,
    freq_range=(1, 30),
    status_channel="Status",
    status_start_code=None,
    status_end_code=-1,
    trial_mode="average",
    min_samples=200,
    trial_info=None,  # Limites [inicio, fin) validados por segmentacion.py
):
    """
    Calculate Lempel-Ziv Complexity (LZC; LZ78) for EEG, optionally trial-by-trial (MATLAB-like).

    MATLAB reference behavior matched:
    - Detrend each segment (linear)
    - Binarize with threshold 0: (detrended > 0)
    - LZ78 dictionary size (number of unique substrings)
    - Normalize: LZC = c * log2(N) / N

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording.
    freq_range : tuple (low, high)
        Band-pass range applied before LZC (only if raw.info highpass/lowpass differ).
    status_channel : str
        Name of status/stim channel used for trial segmentation (if trial_mode != 'continuous').
    status_start_code : int or None
        Start event code. If None, LZC is computed over the continuous signal.
    status_end_code : int
        End event code.
        - If -1: end at the next non-zero impulse after the start impulse.
        - Else: end at the first impulse with value == status_end_code after the start.
    trial_mode : {'continuous', 'trials', 'average'}
        - 'continuous': compute LZC on the full continuous signal (per channel; then per zone).
        - 'trials': compute LZC per trial and return matrices (channels x trials, zones x trials).
        - 'average': same as 'trials' plus return zone-wise mean/std across trials.
    min_samples : int
        Minimum number of samples required to compute LZC for a segment. Shorter segments are set to NaN.

    Returns
    -------
    results : dict
        Continuous mode:
            - 'lzc_values' : (n_zones,)
            - 'lzc_per_channel' : (n_eeg_channels,)
        Trials mode:
            - 'lzc_matrix_channels' : (n_eeg_channels, n_trials)
            - 'lzc_matrix_zones' : (n_zones, n_trials)
            - 'lzc_avg' : (n_zones,)  (only if trial_mode == 'average')
            - 'lzc_std' : (n_zones,)  (only if trial_mode == 'average')
        Common:
            - 'zone_names', 'zone_channels', 'channel_names', 'trial_bounds' (if trials)
    """
    import mne
    import numpy as np

    # ---------------------------
    # Helper: extract impulses from Status channel (0 -> nonzero transitions)
    # ---------------------------
    def _get_status_impulses(raw_obj, ch_name):
        if ch_name not in raw_obj.ch_names:
            raise ValueError(f"status_channel='{ch_name}' not found in raw.ch_names")
        status = raw_obj.copy().pick_channels([ch_name]).get_data()[0]
        s = np.round(status).astype(int)
        # Mask to 16-bit to avoid spurious 65536 (BioSemi)
        s = s & 0xFFFF
        prev = np.r_[0, s[:-1]]
        impulse_mask = (s != 0) & (prev == 0)
        idx = np.where(impulse_mask)[0]
        vals = s[idx]
        return idx, vals

    # ---------------------------
    # Helper: build trial bounds (start, end, end_value)
    # ---------------------------
    def _build_trials(idx, vals, start_code, end_code):
        trials = []
        for k in range(len(idx)):
            if vals[k] != start_code:
                continue
            start = idx[k]
            end = None
            end_val = None

            if end_code == -1:
                if k + 1 < len(idx):
                    end = idx[k + 1]
                    end_val = int(vals[k + 1])
            else:
                # find first subsequent impulse with desired end_code
                for j in range(k + 1, len(idx)):
                    if vals[j] == end_code:
                        end = idx[j]
                        end_val = int(vals[j])
                        break

            if end is None or end <= start:
                continue
            trials.append((int(start), int(end), end_val))
        return trials

    # ---------------------------
    # Zones definition
    # ---------------------------
    zone_names = list(get_regions())
    zones = get_regions()
    # ---------------------------
    # Filter (continuous) if necessary
    # ---------------------------
    if raw.info.get("highpass", None) != freq_range[0] or raw.info.get("lowpass", None) != freq_range[1]:
        raw_filt = raw.copy()
        raw_filt.filter(l_freq=freq_range[0], h_freq=freq_range[1], picks="eeg", method="iir", verbose=False)
    else:
        raw_filt = raw

    # EEG picks and data aligned
    eeg_picks = mne.pick_types(
        raw_filt.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False, exclude=[]
    )
    eeg_ch_names = [raw_filt.ch_names[i] for i in eeg_picks]
    data = raw_filt.get_data(picks=eeg_picks)  # (n_eeg, n_time)

    # Build zone indices against EEG-only channel list
    zone_indices = {}
    zone_channels = {}
    for zone_name in zone_names:
        chs = [ch for ch in zones[zone_name] if ch in eeg_ch_names]
        idxs = [eeg_ch_names.index(ch) for ch in chs]
        zone_indices[zone_name] = idxs
        zone_channels[zone_name] = chs

    # ---------------------------
    # Continuous mode (backward compatible)
    # ---------------------------
    if status_start_code is None or trial_mode == "continuous":
        n_timepoints = data.shape[1]
        lzc_per_channel = np.full(data.shape[0], np.nan, dtype=float)

        for ch_idx in range(data.shape[0]):
            signal = data[ch_idx, :]
            signal_detrended = detrend(signal, type="linear")
            binary_signal = (signal_detrended > 0).astype(int)
            c = _lz78_complexity(binary_signal)
            lzc_per_channel[ch_idx] = c * np.log2(n_timepoints) / n_timepoints

        lzc_per_zone = np.full(len(zone_names), np.nan, dtype=float)
        for zi, zone_name in enumerate(zone_names):
            idxs = zone_indices[zone_name]
            lzc_per_zone[zi] = np.nanmean(lzc_per_channel[idxs]) if len(idxs) else np.nan

        return {
            "lzc_values": lzc_per_zone,
            "lzc_per_channel": lzc_per_channel,
            "zone_names": zone_names,
            "zone_channels": zone_channels,
            "channel_names": eeg_ch_names,
        }

    # ---------------------------
    # Trials mode
    # ---------------------------
    # Build trials from raw (unfiltered is fine for markers); indices apply to raw_filt data
    if trial_info is None:
        idx, vals = _get_status_impulses(raw, status_channel)
        trials = _build_trials(idx, vals, int(status_start_code), int(status_end_code))
    else:
        trials = list(zip(trial_info["trial_starts"], trial_info["trial_ends"], trial_info["trial_end_values"]))
    n_trials = len(trials)
    if n_trials == 0:
        raise RuntimeError("No trials detected with the provided status_start_code/status_end_code.")

    lzc_matrix_channels = np.full((data.shape[0], n_trials), np.nan, dtype=float)

    for ti, (start, end, end_val) in enumerate(trials):
        # Slice trial segment
        seg = data[:, start:end]
        N = seg.shape[1]
        if N < int(min_samples):
            continue

        for ch_idx in range(seg.shape[0]):
            signal = seg[ch_idx, :]
            signal_detrended = detrend(signal, type="linear")
            binary_signal = (signal_detrended > 0).astype(int)
            c = _lz78_complexity(binary_signal)
            lzc_matrix_channels[ch_idx, ti] = c * np.log2(N) / N

    # Zone aggregation per trial
    lzc_matrix_zones = np.full((len(zone_names), n_trials), np.nan, dtype=float)
    for zi, zone_name in enumerate(zone_names):
        idxs = zone_indices[zone_name]
        if len(idxs) == 0:
            continue
        lzc_matrix_zones[zi, :] = np.nanmean(lzc_matrix_channels[idxs, :], axis=0)

    results = {
        "lzc_matrix_channels": lzc_matrix_channels,
        "lzc_matrix_zones": lzc_matrix_zones,
        "zone_names": zone_names,
        "zone_channels": zone_channels,
        "channel_names": eeg_ch_names,
        "trial_bounds": trials,  # list of (start, end, end_code)
    }

    if trial_mode == "average":
        results["lzc_avg"] = np.nanmean(lzc_matrix_zones, axis=1)
        results["lzc_std"] = np.nanstd(lzc_matrix_zones, axis=1)

    return results


def permutation_entropy(
    raw,
    band_range=(4, 8),
    embedding_dim=3,
    tau=None,
    status_channel="Status",
    status_start_code=None,
    status_end_code=-1,
    trial_mode="average",  # 'continuous' | 'trials' | 'average'
    min_samples=200,
    trial_info=None,  # Limites [inicio, fin) validados por segmentacion.py
):
    """
    Permutation Entropy (PE).

    MATLAB reference computes PE per channel and per trial by:
      1) Symbolizing the signal into ordinal patterns (Bandt–Pompe) using the
         *permutation of indices returned by sort* (not ranks).
      2) Computing marginal probabilities p over symbols (m! states)
      3) Shannon entropy: H = -sum(p*log(p))
      4) Normalization: PE = H / log(m!)  (natural log)

    This implementation:
      - Filters EEG in band_range (FIR, zero-phase)
      - Computes PE either on the continuous recording or per trial based on Status
      - Aggregates PE across channels into predefined zones

    Parameters
    ----------
    raw : mne.io.Raw
    band_range : tuple(float, float)
    embedding_dim : int
    tau : int | None
    status_channel : str
    status_start_code : int | None
    status_end_code : int
        - >0 : end at next impulse with this code
        - -1 : end at next non-zero impulse after start
        - 0  : end at transition back to 0 (block-style)
    trial_mode : str
        'continuous' | 'trials' | 'average'
    min_samples : int
        Minimum samples per segment (post-filter) required to compute PE.

    Returns
    -------
    dict
        Continuous:
          - pe_values (zones,)
          - pe_per_channel (n_eeg,)
        Trial-based:
          - pe_matrix_channels (n_eeg, n_trials)
          - pe_matrix_zones (n_zones, n_trials)
          - pe_avg / pe_std (n_zones,) if trial_mode == 'average'
    """
    import math

    import mne
    import numpy as np

    # ---- zones ----
    zones = get_regions()
    zone_names = list(zones.keys())
    n_zones = len(zone_names)

    # ---- tau (MATLAB-like) ----
    if tau is None:
        fs = float(raw.info["sfreq"])
        tau = int(round(fs / embedding_dim / band_range[1]))

    n_symbols = math.factorial(int(embedding_dim))
    log_norm = float(np.log(n_symbols))

    # --------
    # permutations(1:m) lexicographic, then force mirrored second half
    all_perms = sorted(list(permutations(range(1, int(embedding_dim) + 1))))
    n_perm = len(all_perms)
    all_perms = list(all_perms)
    for i in range(n_perm // 2):
        all_perms[n_perm - i - 1] = tuple(reversed(all_perms[i]))

    adjust = [10 ** (int(embedding_dim) - 1 - i) for i in range(int(embedding_dim))]
    symbols = [sum(int(p) * a for p, a in zip(perm, adjust)) for perm in all_perms]
    code_to_idx = {int(code): idx for idx, code in enumerate(symbols)}  # 0..m!-1

    def _symbolize_1d(seg_1d: np.ndarray) -> np.ndarray:
        """Return 0-based symbol indices for a 1D segment (MATLAB sort-permutation)."""
        n = int(seg_1d.size)
        n_samples = n - int(tau) * (int(embedding_dim) - 1)
        if n_samples <= 0:
            return np.array([], dtype=int)

        out = np.zeros(n_samples, dtype=int)
        m = int(embedding_dim)
        t = int(tau)

        for k in range(n_samples):
            idxs = [k + j * t for j in range(m)]
            pattern = seg_1d[idxs]
            perm = np.argsort(pattern, kind="mergesort") + 1  # 1..m
            code = sum(int(p) * a for p, a in zip(perm, adjust))
            out[k] = code_to_idx.get(int(code), 0)
        return out

    def _pe_from_segment(seg_1d: np.ndarray) -> float:
        symbols_seq = _symbolize_1d(seg_1d)
        if symbols_seq.size == 0:
            return np.nan
        counts = np.bincount(symbols_seq, minlength=n_symbols).astype(float)
        total = counts.sum()
        if total <= 0:
            return np.nan
        p = counts / total
        with np.errstate(divide="ignore", invalid="ignore"):
            prod = p * np.log(p)
        prod[~np.isfinite(prod)] = 0.0
        H = -float(np.sum(prod))
        return H / log_norm

    # ---- filter (band) ----
    raw_theta = raw.copy()
    raw_theta.filter(l_freq=band_range[0], h_freq=band_range[1], picks="eeg", method="fir", phase="zero", verbose=False)

    # ---- EEG-only picks and mapping (fix channel-index bug) ----
    eeg_picks = mne.pick_types(
        raw_theta.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False, exclude=[]
    )
    eeg_names = [raw_theta.ch_names[i] for i in eeg_picks]
    data = raw_theta.get_data(picks=eeg_picks)  # (n_eeg, n_times)

    zone_indices = {}
    zone_channels = {}
    for zn in zone_names:
        chs = [ch for ch in zones[zn] if ch in eeg_names]
        zone_channels[zn] = chs
        zone_indices[zn] = [eeg_names.index(ch) for ch in chs]

    # ---- mode: continuous ----
    if (trial_mode == "continuous") or (status_start_code is None) or (status_channel not in raw_theta.ch_names):
        pe_per_channel = np.array([_pe_from_segment(data[ch, :]) for ch in range(data.shape[0])], dtype=float)
        pe_per_zone = np.full((n_zones,), np.nan, dtype=float)
        for zi, zn in enumerate(zone_names):
            idx = zone_indices[zn]
            if idx:
                pe_per_zone[zi] = float(np.nanmean(pe_per_channel[idx]))
        return {
            "pe_values": pe_per_zone,
            "pe_per_channel": pe_per_channel,
            "zone_names": zone_names,
            "zone_channels": zone_channels,
            "tau": tau,
            "trial_mode": "continuous",
            "n_trials": 0,
        }

    # Limites explicitos para evitar redetectar marcas artificiales.
    trials_info = trial_info
    if trials_info is None:
        trials_info = _extract_trials_from_status(
            raw_theta, status_channel, "average", status_start_code, status_end_code)
    if trials_info is None:
        # fallback to continuous
        return permutation_entropy(
            raw,
            band_range=band_range,
            embedding_dim=embedding_dim,
            tau=tau,
            status_channel=status_channel,
            status_start_code=None,
            status_end_code=status_end_code,
            trial_mode="continuous",
            min_samples=min_samples,
        )

    trial_starts = np.array(trials_info["trial_starts"], dtype=int)
    trial_ends = np.array(trials_info["trial_ends"], dtype=int)
    trial_end_vals = trials_info.get("trial_end_values", [None] * len(trial_starts))
    n_trials = int(len(trial_starts))

    pe_trials_channels = np.full((data.shape[0], n_trials), np.nan, dtype=float)

    for t_i in range(n_trials):
        s = int(trial_starts[t_i])
        e = int(trial_ends[t_i])
        if e <= s or (e - s) < int(min_samples):
            continue
        seg = data[:, s:e]
        for ch in range(seg.shape[0]):
            pe_trials_channels[ch, t_i] = _pe_from_segment(seg[ch, :])

    pe_trials_zones = np.full((n_zones, n_trials), np.nan, dtype=float)
    for zi, zn in enumerate(zone_names):
        idx = zone_indices[zn]
        if idx:
            pe_trials_zones[zi, :] = np.nanmean(pe_trials_channels[idx, :], axis=0)

    results = {
        "channel_names": eeg_names,
        "pe_matrix_channels": pe_trials_channels,
        "pe_matrix_zones": pe_trials_zones,
        "zone_names": zone_names,
        "zone_channels": zone_channels,
        "tau": tau,
        "trial_mode": "average" if trial_mode == "average" else "trials",
        "n_trials": n_trials,
        "trial_bounds": list(zip(trial_starts.tolist(), trial_ends.tolist(), trial_end_vals)),
    }

    if trial_mode == "average":
        results["pe_avg"] = np.nanmean(pe_trials_zones, axis=1)
        results["pe_std"] = np.nanstd(pe_trials_zones, axis=1)

    return results


def ctransform(x):
    """
    Copula transformation (empirical CDF).

    WHAT IT DOES:
    -------------
    Converts data to uniform distribution [0,1] using empirical CDF.
    This is the first step in Gaussian copula transformation.

    Empirical CDF = rank of each value / (total values + 1)

    Example:
    --------
    Data: [5, 2, 8, 1]
    Ranks: [3, 2, 4, 1]
    CDF values: [3/5, 2/5, 4/5, 1/5] = [0.6, 0.4, 0.8, 0.2]

    WHY IT'S USEFUL:
    ----------------
    - Makes distribution uniform (all values equally likely)
    - Removes effect of marginal distributions
    - Preserves rank order (monotonic transformation)
    - Allows focus on dependencies, not marginals

    Parameters
    ----------
    x : array
        Input data (any distribution)

    Returns
    -------
    cx : array
        Empirical CDF values in (0, 1) open interval

    Notes
    -----
    Uses (rank+1)/(n+1) to avoid 0 and 1 (needed for inverse normal)
    """

    # Sort indices to get ranks
    xi = np.argsort(np.atleast_2d(x))  # Indices that would sort x
    xr = np.argsort(xi)  # Ranks of each element

    # Convert ranks to empirical CDF values
    # Formula: (rank + 1) / (n + 1)
    # +1 in numerator and denominator keeps values in open interval (0,1)
    cx = (xr + 1).astype(np.float64) / (xr.shape[-1] + 1)

    return cx


def copnorm(x):
    """
    Copula normalization - Gaussian copula transformation.

    WHAT IT DOES:
    -------------
    Transforms data to standard normal distribution while preserving
    rank structure (dependencies).

    Process:
    1. Convert to empirical CDF (uniform distribution)
    2. Apply inverse normal CDF (quantile function)
    Result: Gaussian distribution with same rank order

    EXAMPLE:
    --------
    Original: [100, 50, 200, 25] (skewed distribution)
    Ranks:    [3, 2, 4, 1]
    After copnorm: [-0.32, -0.84, 0.84, -1.65] (normal distribution, same order)

    WHY IT'S USEFUL:
    ----------------
    - Allows use of Gaussian methods on non-Gaussian data
    - Preserves dependencies (correlations, nonlinear relationships)
    - Removes effect of marginal distributions
    - Robust to outliers (based on ranks)

    Used in:
    --------
    - Transfer Entropy: makes CMI calculation valid for non-Gaussian EEG
    - Removes amplitude differences between channels
    - Focuses analysis on temporal relationships

    Parameters
    ----------
    x : array
        Input data (any continuous distribution)
        Operates along last axis

    Returns
    -------
    cx : array
        Gaussian copula normalized data
        Mean ≈ 0, Std ≈ 1, ranks preserved

    Notes
    -----
    - Uses scipy.special.ndtri (inverse normal CDF)
    - Equivalent to: norm.ppf(ctransform(x))
    - Result has zero mean (approximately)
    """

    # Apply inverse normal CDF to empirical CDF values
    # ndtri = inverse of normal CDF = quantile function
    # Transforms uniform [0,1] to normal distribution
    cx = sp_special.ndtri(ctransform(x))

    return cx


def cmi_ggg(x, y, z, biascorrect=True, demeaned=False):
    """
    Conditional Mutual Information (CMI) between Gaussian variables.

    WHAT IT MEASURES:
    -----------------
    CMI(X; Y | Z) = How much information X provides about Y,
                    given that we already know Z

    In other words:
    - How much does knowing X reduce uncertainty about Y,
      beyond what Z already tells us?

    CMI = 0: X provides no additional info about Y (given Z)
    CMI > 0: X provides useful info about Y (beyond Z)

    FORMULA:
    --------
    CMI(X; Y | Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    Where H() is entropy (uncertainty):
    - H(X,Z): Joint entropy of X and Z
    - H(Y,Z): Joint entropy of Y and Z
    - H(X,Y,Z): Joint entropy of all three
    - H(Z): Entropy of conditioning variable

    FOR TRANSFER ENTROPY:
    ---------------------
    Transfer Entropy from X to Y:
    TE(X→Y) = CMI(X_past; Y_future | Y_past)

    Interpretation:
    - X_past: Past values of signal X
    - Y_future: Future value of signal Y
    - Y_past: Past values of signal Y (what we already know)
    - TE measures: Does X's past help predict Y's future,
                   beyond Y's own past?

    HOW IT WORKS (for Gaussian data):
    ----------------------------------
    1. Compute covariance matrix of [X, Y, Z] combined
    2. Extract submatrices for [X,Z], [Y,Z], [Z]
    3. Compute determinants via Cholesky decomposition
    4. Calculate entropies from determinants
    5. Apply CMI formula
    6. Optional: Apply bias correction for finite samples

    Parameters
    ----------
    x, y, z : arrays
        Input variables, shape (n_variables, n_samples)
        - Rows = different variables/dimensions
        - Columns = different observations/timepoints
        For Transfer Entropy:
          x = past of source signal
          y = future of target signal
          z = past of target signal

    biascorrect : bool, default=True
        Apply bias correction for finite sample size.
        Recommended: True (more accurate for small samples)

    demeaned : bool, default=False
        Whether data already has zero mean.
        Set to True if using copnorm (it produces zero-mean data)

    Returns
    -------
    I : float
        Conditional Mutual Information in bits
        Range: [0, ∞)
        - 0: X and Y independent given Z
        - >0: X provides info about Y beyond Z
        - Typical EEG values: 0.001 - 0.05 bits

    Notes
    -----
    - Algorithm: Ince et al. (2017), Human Brain Mapping
    - Uses Gaussian copula: valid for any continuous distribution
    - Bias correction uses psi (digamma) function
    - Computationally efficient (uses Cholesky, not full covariance inverse)

    Advantages:
    -----------
    - Works for non-Gaussian data (via copula transform)
    - Robust to outliers (when using copnorm)
    - Handles multivariate variables (x, y, z can be vectors)
    - Bias correction for small samples

    Example
    -------
    >>> # Generate test data
    >>> x = np.random.randn(1, 1000)  # Past of signal 1
    >>> z = np.random.randn(1, 1000)  # Past of signal 2
    >>> y = 0.5*x + 0.3*z + 0.2*np.random.randn(1, 1000)  # Future of signal 2
    >>>
    >>> # Normalize with copula
    >>> x_norm = copnorm(x)
    >>> y_norm = copnorm(y)
    >>> z_norm = copnorm(z)
    >>>
    >>> # Calculate CMI
    >>> cmi = cmi_ggg(x_norm, y_norm, z_norm, biascorrect=True, demeaned=True)
    >>> print(f"CMI = {cmi:.4f} bits")
    >>> # Should be > 0 since Y depends on X

    References
    ----------
    Ince et al. (2017). "A statistical framework for neuroimaging data
    analysis based on mutual information estimated via a Gaussian copula."
    Human Brain Mapping, 38, 1541-1573.
    """

    # Ensure 2D arrays
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)

    # Check dimensions
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")

    # Get dimensions
    Ntrl = x.shape[1]  # Number of samples
    Nvarx = x.shape[0]  # Dimensionality of X
    Nvary = y.shape[0]  # Dimensionality of Y
    Nvarz = z.shape[0]  # Dimensionality of Z

    # Calculate combined dimensions
    Nvaryz = Nvary + Nvarz  # Dimensions of [Y, Z]
    Nvarxy = Nvarx + Nvary  # Dimensions of [X, Y]
    Nvarxz = Nvarx + Nvarz  # Dimensions of [X, Z]
    Nvarxyz = Nvarx + Nvaryz  # Dimensions of [X, Y, Z]

    # Check sample sizes match
    if y.shape[1] != Ntrl or z.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # Stack variables: xyz = [X; Y; Z]
    xyz = np.vstack((x, y, z))

    # Demean if not already done
    if not demeaned:
        xyz = xyz - xyz.mean(axis=1)[:, np.newaxis]

    # Compute covariance matrix of joint [X, Y, Z]
    # Cov = (1/(N-1)) * X * X^T
    Cxyz = np.dot(xyz, xyz.T) / float(Ntrl - 1)

    # Extract submatrices needed for CMI calculation
    # Cz: Covariance of Z alone
    Cz = Cxyz[Nvarxy:, Nvarxy:]

    # Cyz: Covariance of [Y, Z]
    Cyz = Cxyz[Nvarx:, Nvarx:]

    # Cxz: Covariance of [X, Z]
    Cxz = np.zeros((Nvarxz, Nvarxz))
    Cxz[:Nvarx, :Nvarx] = Cxyz[:Nvarx, :Nvarx]  # X block
    Cxz[:Nvarx, Nvarx:] = Cxyz[:Nvarx, Nvarxy:]  # X-Z cross block
    Cxz[Nvarx:, :Nvarx] = Cxyz[Nvarxy:, :Nvarx]  # Z-X cross block
    Cxz[Nvarx:, Nvarx:] = Cxyz[Nvarxy:, Nvarxy:]  # Z block

    # Compute Cholesky decompositions
    # Cholesky: C = L * L^T where L is lower triangular
    # Determinant: det(C) = prod(diag(L))^2
    # Log-determinant: log(det(C)) = 2 * sum(log(diag(L)))
    chCz = np.linalg.cholesky(Cz)
    chCxz = np.linalg.cholesky(Cxz)
    chCyz = np.linalg.cholesky(Cyz)
    chCxyz = np.linalg.cholesky(Cxyz)

    # Compute entropies from log-determinants
    # For Gaussian: H = 0.5*log(det(Cov)) + const
    # Constant terms cancel in CMI formula, so we omit them
    HZ = np.sum(np.log(np.diagonal(chCz)))
    HXZ = np.sum(np.log(np.diagonal(chCxz)))
    HYZ = np.sum(np.log(np.diagonal(chCyz)))
    HXYZ = np.sum(np.log(np.diagonal(chCxyz)))

    ln2 = np.log(2)  # Convert from nats to bits

    # Apply bias correction for finite samples
    if biascorrect:
        # Correction uses psi (digamma) function
        # Accounts for bias in entropy estimates from finite data
        psiterms = sp_special.psi((Ntrl - np.arange(1, Nvarxyz + 1)).astype(np.float64) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0

        # Apply correction to each entropy
        HZ = HZ - Nvarz * dterm - psiterms[:Nvarz].sum()
        HXZ = HXZ - Nvarxz * dterm - psiterms[:Nvarxz].sum()
        HYZ = HYZ - Nvaryz * dterm - psiterms[:Nvaryz].sum()
        HXYZ = HXYZ - Nvarxyz * dterm - psiterms[:Nvarxyz].sum()

    # Calculate CMI using entropy formula
    # CMI(X; Y | Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    # Convert from nats to bits by dividing by ln(2)
    I = (HXZ + HYZ - HXYZ - HZ) / ln2

    return I


# ============================================================================
# Transfer Entropy Function
# ============================================================================


def transfer_entropy(
    raw,
    maxlag_ms=400.0,
    maxlag=None,
    min_obs_needed=30,
    status_channel="Status",
    status_start_code=None,
    status_end_code=-1,
    trial_mode="average",
    biascorrect=True,
    trial_info=None,  # Limites [inicio, fin) validados por segmentacion.py
):
    """
    Transfer Entropy (Gaussian Copula / GCMI) between brain zones, MATLAB-like.

    This replicates the MATLAB logic in TE.m / func_TE.m:
    - Build zone signals as the mean across EEG channels in each zone.
    - Segment trials using Status impulses (start_code -> end_code, or -1 = next non-zero impulse).
    - For each trial and each lag=1..maxlag:
        TE(i->j, lag) = CMI( X_i_past ; X_j_future | X_j_past )
      using copula normalization (copnorm) and Gaussian CMI (cmi_ggg).
    - Report both directions (i->j and j->i) for all zone pairs.
    - Return the full lag-resolved TE and the mean over lags (omit NaNs), plus
      average / std across trials.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw object. No additional filtering is applied here.
    maxlag_ms : float
        Maximum lag in milliseconds (robust to sampling rate changes).
        Ignored if `maxlag` is provided.
    maxlag : int or None
        Maximum lag in samples. If provided, overrides maxlag_ms.
    min_obs_needed : int
        Minimum number of *samples* required beyond the lag for a valid estimate.
        (This is NOT number of trials; it is number of time samples.)
    status_channel : str
        Name of the Status channel.
    status_start_code : int
        Impulse code that marks the start of a trial.
    status_end_code : int
        If -1, end at the next non-zero impulse after the start.
        Otherwise, end at the first subsequent impulse that equals this code.
    trial_mode : str
        'trials'  -> return per-trial TE
        'average' -> also compute average/std across trials (recommended)
    biascorrect : bool
        Bias correction for cmi_ggg (matches MATLAB default behavior).

    Returns
    -------
    results : dict with keys
        - 'te_full' : ndarray (n_trials, n_zones, n_zones, maxlag)
            Lag-resolved TE in bits. Diagonal is NaN.
        - 'te_mean_lag' : ndarray (n_trials, n_zones, n_zones)
            Mean over lags (omit NaNs), per trial.
        - 'te_matrix' : ndarray (n_zones, n_zones)
            Mean over trials of te_mean_lag (omit NaNs).
        - 'te_std_trials' : ndarray (n_zones, n_zones)
            Std over trials of te_mean_lag.
        - 'zone_names', 'zone_channels', 'eeg_channel_names'
        - 'trial_bounds' : list[(start, end, end_code)]
        - 'maxlag_samples', 'maxlag_ms', 'sfreq'
        - 'errors' : list[dict] with exception info (if any)
    """
    import mne
    import numpy as np

    # ---------------------------
    # Helper: extract impulses from Status channel (0 -> nonzero transitions)
    # ---------------------------
    def _get_status_impulses(raw_obj, ch_name):
        if ch_name not in raw_obj.ch_names:
            raise ValueError(f"status_channel='{ch_name}' not found in raw.ch_names")
        status = raw_obj.copy().pick_channels([ch_name]).get_data()[0]
        s = np.round(status).astype(int)
        # Mask to 16-bit to avoid spurious 65536 (BioSemi)
        s = s & 0xFFFF
        prev = np.r_[0, s[:-1]]
        impulse_mask = (s != 0) & (prev == 0)
        idx = np.where(impulse_mask)[0]
        vals = s[idx]
        return idx, vals

    # ---------------------------
    # Helper: build trial bounds (start, end, end_value)
    # ---------------------------
    def _build_trials(idx, vals, start_code, end_code):
        trials = []
        for k in range(len(idx)):
            if vals[k] != start_code:
                continue
            start = int(idx[k])
            end = None
            end_val = None

            if end_code == -1:
                if k + 1 < len(idx):
                    end = int(idx[k + 1])
                    end_val = int(vals[k + 1])
            else:
                for j in range(k + 1, len(idx)):
                    if vals[j] == end_code:
                        end = int(idx[j])
                        end_val = int(vals[j])
                        break

            if end is None or end <= start:
                continue
            trials.append((start, end, end_val))
        return trials

    if status_start_code is None:
        raise ValueError("status_start_code must be provided for trial-by-trial TE.")

    sfreq = float(raw.info["sfreq"])

    # Convert maxlag_ms -> samples (robust to sampling rate changes)
    if maxlag is None:
        maxlag_samples = int(np.round((float(maxlag_ms) / 1000.0) * sfreq))
    else:
        maxlag_samples = int(maxlag)
        maxlag_ms = (maxlag_samples / sfreq) * 1000.0

    if maxlag_samples < 1:
        raise ValueError(f"maxlag_samples must be >= 1 (got {maxlag_samples}).")

    # ---------------------------
    # Zones definition (same structure used across the pipeline)
    # ---------------------------
    zone_names = list(get_regions())
    zones = get_regions()

    # ---------------------------
    # EEG-only channel mapping (fixes index misalignment bugs)
    # ---------------------------
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False, exclude=[])
    eeg_channel_names = [raw.ch_names[i] for i in eeg_picks]
    data_eeg = raw.get_data(picks=eeg_picks)  # (n_eeg, n_times)

    zone_channels = {}
    zone_indices = {}
    for zname in zone_names:
        ch_list = [ch for ch in zones[zname] if ch in eeg_channel_names]
        zone_channels[zname] = ch_list
        zone_indices[zname] = [eeg_channel_names.index(ch) for ch in ch_list]

    # Sanity: ensure every zone has at least 1 EEG channel
    empty_zones = [z for z in zone_names if len(zone_indices.get(z, [])) == 0]
    if empty_zones:
        raise RuntimeError(
            f"TE: zones with no EEG channels found: {empty_zones}. Check channel naming / montage vs zones mapping."
        )
    # ---------------------------
    # Trials from Status impulses
    # ---------------------------
    if trial_info is None:
        idx, vals = _get_status_impulses(raw, status_channel)
        trial_bounds = _build_trials(idx, vals, int(status_start_code), int(status_end_code))
    else:
        trial_bounds = list(zip(trial_info["trial_starts"], trial_info["trial_ends"], trial_info["trial_end_values"]))

    if len(trial_bounds) == 0:
        raise RuntimeError("No trials detected. Check status codes and Status channel.")

    n_trials = len(trial_bounds)
    n_zones = len(zone_names)

    # Output arrays
    te_full = np.full((n_trials, n_zones, n_zones, maxlag_samples), np.nan, dtype=float)
    te_mean_lag = np.full((n_trials, n_zones, n_zones), np.nan, dtype=float)

    errors = []

    # ---------------------------
    # Compute TE per trial
    # ---------------------------
    for t, (start, end, end_code_val) in enumerate(trial_bounds):
        seg = data_eeg[:, start:end]  # (n_eeg, n_samples)
        n_samples = seg.shape[1]

        # Build zone time series: mean across channels in zone
        zone_ts = np.zeros((n_zones, n_samples), dtype=float)
        for zi, zname in enumerate(zone_names):
            inds = zone_indices[zname]
            if len(inds) == 0:
                zone_ts[zi, :] = np.nan
            else:
                zone_ts[zi, :] = np.nanmean(seg[inds, :], axis=0)

        # TE for each lag and each unordered pair; fill both directions
        for lag in range(1, maxlag_samples + 1):
            if n_samples < (lag + int(min_obs_needed)):
                # Not enough observations at this lag
                continue

            # Past/future slicing (shape: n_zones x (n_samples - lag))
            past = zone_ts[:, : n_samples - lag]
            future = zone_ts[:, lag:]

            # Compute pairwise TE (i<j) and fill i->j and j->i
            for i in range(n_zones):
                for j in range(i + 1, n_zones):
                    # Skip if either zone is NaN-only
                    if (
                        np.all(np.isnan(past[i]))
                        or np.all(np.isnan(past[j]))
                        or np.all(np.isnan(future[i]))
                        or np.all(np.isnan(future[j]))
                    ):
                        continue

                    try:
                        # build x=[i_past, j_past], y=[i_future, j_future], copnorm both
                        x_stack = np.vstack([past[i], past[j]])
                        y_stack = np.vstack([future[i], future[j]])

                        x_c = copnorm(x_stack)
                        y_c = copnorm(y_stack)

                        # i -> j : CMI( i_past ; j_future | j_past )
                        te_ij = cmi_ggg(x_c[0], y_c[1], x_c[1], biascorrect=biascorrect)
                        # j -> i : CMI( j_past ; i_future | i_past )
                        te_ji = cmi_ggg(x_c[1], y_c[0], x_c[0], biascorrect=biascorrect)

                        te_full[t, i, j, lag - 1] = float(te_ij)
                        te_full[t, j, i, lag - 1] = float(te_ji)

                    except Exception as e:
                        errors.append({"trial": int(t), "lag": int(lag), "pair": (int(i), int(j)), "error": repr(e)})
                        # leave NaNs for this pair/lag
                        continue

        # Mean over lags (omit NaNs), per trial
        # Compute off-diagonal means explicitly to avoid warnings from all-NaN diagonal slices
        # te_mean_lag is pre-filled with NaNs
        for i_mean in range(n_zones):
            for j_mean in range(n_zones):
                if i_mean == j_mean:
                    continue
                vals = te_full[t, i_mean, j_mean, :]
                if not np.any(np.isfinite(vals)):
                    continue
                te_mean_lag[t, i_mean, j_mean] = float(np.nanmean(vals))

        # Force diagonal to NaN (like MATLAB never defines self-TE)
        for k in range(n_zones):
            te_mean_lag[t, k, k] = np.nan
            te_full[t, k, k, :] = np.nan

    # Aggregate across trials
    # Compute off-diagonal aggregates explicitly to avoid warnings from all-NaN diagonal slices
    te_matrix = np.full((n_zones, n_zones), np.nan, dtype=float)
    te_std_trials = np.full((n_zones, n_zones), np.nan, dtype=float)

    for i_ag in range(n_zones):
        for j_ag in range(n_zones):
            if i_ag == j_ag:
                continue
            vals = te_mean_lag[:, i_ag, j_ag]
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            te_matrix[i_ag, j_ag] = float(np.nanmean(vals))
            if int(np.sum(finite)) >= 2:
                te_std_trials[i_ag, j_ag] = float(np.nanstd(vals))

    results = {
        "te_full": te_full,
        "te_mean_lag": te_mean_lag,
        "te_matrix": te_matrix,
        "te_std_trials": te_std_trials,
        "zone_names": zone_names,
        "zone_channels": zone_channels,
        "eeg_channel_names": eeg_channel_names,
        "trial_bounds": trial_bounds,
        "maxlag_samples": maxlag_samples,
        "maxlag_ms": float(maxlag_ms),
        "sfreq": sfreq,
        "min_obs_needed": int(min_obs_needed),
        "errors": errors,
    }

    return results


# ============================================================================
# EXAMPLE USAGE (MAIN)
# ============================================================================


def _build_aggregated_row(
    subject: str,
    band: str,
    condition: int,
    spectral_results: dict | None = None,
    wpli_results: dict | None = None,
    pe_results: dict | None = None,
    lzc_results: dict | None = None,
    te_results: dict | None = None,
    wsmi_results: dict | None = None,
):
    """Create one aggregated row per subject / band / condition."""

    row = {"subject": subject, "band": band, "condition": int(condition)}

    # ------------------------------------------------------------------
    # Spectral parametrization
    # ------------------------------------------------------------------
    if spectral_results is not None:
        if spectral_results.get("mode") == "trials":
            z = spectral_results["zone_names"]

            theta_avg = np.nanmean(spectral_results["theta_power_zones"], axis=0)
            exp_avg = np.nanmean(spectral_results["aperiodic_exponent_zones"], axis=0)
            off_avg = np.nanmean(spectral_results["aperiodic_offset_zones"], axis=0)

            for zi, zn in enumerate(z):
                row[f"spec_period_{zn}"] = float(theta_avg[zi])
                row[f"spec_exp_{zn}"] = float(exp_avg[zi])
                row[f"spec_off_{zn}"] = float(off_avg[zi])

        elif spectral_results.get("mode") == "continuous":
            # por si alguna vez lo usás en continuo
            ch_names = spectral_results.get("channel_names", [])
            row["spec_period_global"] = float(np.nanmean(spectral_results["theta_power"]))
            row["spec_exp_global"] = float(np.nanmean(spectral_results["aperiodic_exponent"]))
            row["spec_off_global"] = float(np.nanmean(spectral_results["aperiodic_offset"]))

    # ------------------------------------------------------------------
    # wPLI
    # ------------------------------------------------------------------
    if wpli_results is not None:
        if wpli_results.get("mode") == "trials":
            zone_names = wpli_results["zone_names"]
            wpli_avg = np.nanmean(wpli_results["wpli_trials"], axis=0)  # (zones, zones)
            row.update(_flatten_symmetric_zone_matrix(wpli_avg, zone_names, prefix="wpli"))

        elif wpli_results.get("mode") == "continuous":
            zone_names = wpli_results["zone_names"]
            row.update(_flatten_symmetric_zone_matrix(wpli_results["wpli_matrix"], zone_names, prefix="wpli"))

    # ------------------------------------------------------------------
    # Permutation Entropy
    # ------------------------------------------------------------------
    if pe_results is not None:
        zone_names = pe_results["zone_names"]

        if "pe_matrix_zones" in pe_results:
            pe_avg = np.nanmean(pe_results["pe_matrix_zones"], axis=1)
            for zi, zn in enumerate(zone_names):
                row[f"pe_{zn}"] = float(pe_avg[zi])

        elif "pe_values" in pe_results:
            for zi, zn in enumerate(zone_names):
                row[f"pe_{zn}"] = float(pe_results["pe_values"][zi])

    # ------------------------------------------------------------------
    # LZC
    # ------------------------------------------------------------------
    if lzc_results is not None:
        zone_names = lzc_results["zone_names"]

        if "lzc_matrix_zones" in lzc_results:
            lzc_avg = np.nanmean(lzc_results["lzc_matrix_zones"], axis=1)
            for zi, zn in enumerate(zone_names):
                row[f"lzc_{zn}"] = float(lzc_avg[zi])

        elif "lzc_values" in lzc_results:
            for zi, zn in enumerate(zone_names):
                row[f"lzc_{zn}"] = float(lzc_results["lzc_values"][zi])

    # ------------------------------------------------------------------
    # Transfer Entropy
    # ------------------------------------------------------------------
    if te_results is not None:
        zone_names = te_results["zone_names"]

        if "te_mean_lag" in te_results:
            te_avg = np.nanmean(te_results["te_mean_lag"], axis=0)  # (zones, zones)
            row.update(_flatten_directed_zone_matrix(te_avg, zone_names, prefix="te", include_self=False))

        elif "te_matrix" in te_results:
            row.update(
                _flatten_directed_zone_matrix(te_results["te_matrix"], zone_names, prefix="te", include_self=False)
            )

    # ------------------------------------------------------------------
    # wSMI
    # ------------------------------------------------------------------
    if wsmi_results is not None:
        zone_names = wsmi_results["zone_names"]

        if "wsmi_matrix" in wsmi_results:
            wsmi_matrix = wsmi_results["wsmi_matrix"]

            if wsmi_matrix.ndim == 3:
                wsmi_avg = np.nanmean(wsmi_matrix, axis=2)  # (zones, zones)
            else:
                wsmi_avg = wsmi_matrix

            row.update(_flatten_symmetric_zone_matrix(wsmi_avg, zone_names, prefix="wsmi"))

    return row




