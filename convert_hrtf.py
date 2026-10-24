#!/usr/bin/env python3
"""
SONICOM HRTF -> Original Xbox MCPX APU Optimal Converter & Datasheet Generator
(v5: Hardware-Accurate 1-3-4 Sign-Magnitude Minifloat Encoding + Unity H_inf Gain
     + Band-Pass Midrange ILD Reinforcement + Preserved Pinna Elevation Notches)
"""

import argparse
import math
import os
import sys

try:
    import h5py
    import numpy as np
except ImportError:
    print("ERROR: h5py and numpy are required. Install with: pip install h5py numpy")
    sys.exit(1)


# ============================================================================
# Xbox MCPX Hardware Constants
# ============================================================================

SAMPLE_RATE = 48000
NUM_FIR_TAPS = 31           # MCPX hardware FIR filter length (645.8 us at 48kHz)
MAX_ITD = 42                # MCPX hardware ITD delay line limit (875 us at 48kHz)
MAX_INT8 = 127              # Symmetric 7-bit magnitude limit for legacy int8 modes
MINIFLOAT_MAX = 1984.0 / 2048.0  # 0x7F = (16 + 15) * 2^6 / 2048 = 31/32 = 0.96875

# Diffuse-field ear-canal de-resonance strength
DF_EQ_STRENGTH = 1.0

# Output grid (matches nxdk-audio's nxAudioHRTFSetParamsFromAngles)
AZIMUTH_MIN = 0
AZIMUTH_MAX = 180
AZIMUTH_STEP = 5            # 37 azimuth entries (0, 5, 10, ..., 180)

ELEVATION_MIN = -40
ELEVATION_MAX = 90
ELEVATION_STEP = 10         # 14 elevation entries (-40, -30, ..., +90)

NUM_AZIMUTHS = (AZIMUTH_MAX - AZIMUTH_MIN) // AZIMUTH_STEP + 1          # 37
NUM_ELEVATIONS = (ELEVATION_MAX - ELEVATION_MIN) // ELEVATION_STEP + 1  # 14


# ============================================================================
# MCPX 1-3-4 Sign-Magnitude Minifloat Codec
# ============================================================================

def decode_minifloat_byte(b):
    """Decode an 8-bit MCPX 1-3-4 sign-magnitude minifloat byte to float [-31/32, +31/32]."""
    b = int(b) & 0xFF
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 4) & 0x07
    mant = b & 0x0F
    if exp == 0:
        mag = mant / 2048.0
    else:
        mag = ((16 + mant) << (exp - 1)) / 2048.0
    return sign * mag


# Precompute lookup table of all 255 valid minifloat bytes (excluding 0x80 = -0.0)
_MINIFLOAT_BYTES = np.array([b for b in range(256) if b != 0x80], dtype=np.uint8)
_MINIFLOAT_VALS = np.array([decode_minifloat_byte(b) for b in _MINIFLOAT_BYTES], dtype=np.float64)


def quantize_minifloat_error_feedback(taps_float):
    """Quantize float taps to 8-bit MCPX 1-3-4 minifloat bytes with 1st-order error feedback."""
    q_bytes = []
    q_vals = []
    err = 0.0
    for s in taps_float:
        target = float(np.clip(s + err, -MINIFLOAT_MAX, MINIFLOAT_MAX))
        idx = int(np.argmin(np.abs(_MINIFLOAT_VALS - target)))
        b = int(_MINIFLOAT_BYTES[idx])
        val = float(_MINIFLOAT_VALS[idx])
        err = target - val
        q_bytes.append(b)
        q_vals.append(val)
    return q_bytes, np.array(q_vals, dtype=np.float64)


def quantize_int8_error_feedback(taps_scaled, encoding="signmag"):
    """Legacy linear int8 quantizer for 'signmag' or 'twos' modes."""
    q_bytes = []
    q_vals = []
    err = 0.0
    for s in taps_scaled:
        val = s + err
        q = int(round(val))
        q = max(-MAX_INT8, min(MAX_INT8, q))
        err = val - q
        if q == 0:
            b = 0
        elif encoding == "signmag":
            b = (0x80 | abs(q)) if q < 0 else q
        else:
            b = (q + 256) if q < 0 else q
        q_bytes.append(b)
        q_vals.append(q / 127.0)
    return q_bytes, np.array(q_vals, dtype=np.float64)


# ============================================================================
# SOFA File Reading
# ============================================================================

def read_sofa(filepath):
    """Read a SOFA file and extract HRIRs, delays, and source positions."""
    print(f"Reading SOFA file: {os.path.basename(filepath)}")

    with h5py.File(filepath, "r") as f:
        ir_data = np.array(f["Data.IR"][:], dtype=np.float64)
        delay_data = np.array(f["Data.Delay"][:], dtype=np.float64)
        positions = np.array(f["SourcePosition"][:], dtype=np.float64)

        sr = f["Data.SamplingRate"][()]
        sr = float(sr.flat[0]) if hasattr(sr, "flat") else float(sr)

    num_measurements, num_receivers, ir_length = ir_data.shape
    azimuths = sorted(set(int(round(p)) for p in positions[:, 0]))
    elevations = sorted(set(int(round(p)) for p in positions[:, 1]))

    print(f"  Measurements : {num_measurements}")
    print(f"  Receivers    : {num_receivers}")
    print(f"  IR length    : {ir_length} samples ({ir_length / sr * 1000:.2f} ms)")
    print(f"  Sample rate  : {int(sr)} Hz")
    print(f"  Delay shape  : {delay_data.shape} (range: {delay_data.min():.1f} to {delay_data.max():.1f})")
    print(f"  Azimuths     : {min(azimuths)} deg to {max(azimuths)} deg ({len(azimuths)} unique)")
    print(f"  Elevations   : {min(elevations)} deg to {max(elevations)} deg ({len(elevations)} unique)")

    if int(round(sr)) != SAMPLE_RATE:
        print(f"  WARNING: Expected {SAMPLE_RATE} Hz, got {int(sr)} Hz.")

    return ir_data, delay_data, positions, sr


# ============================================================================
# Coordinate Geometry on the Unit Sphere (S^2)
# ============================================================================

def sph_to_cart(az_deg, el_deg):
    """Convert spherical (azimuth, elevation) in degrees to unit 3D vector."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    cos_el = np.cos(el)
    return np.stack([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)], axis=-1)


def find_spherical_weights(unit_vectors, target_az_deg, target_el_deg, k=3, power=2.0):
    """Find k-nearest measurements on S^2 using Great-Circle distance."""
    target_vec = sph_to_cart(target_az_deg, target_el_deg)
    dots = np.clip(unit_vectors @ target_vec, -1.0, 1.0)
    angles_rad = np.arccos(dots)

    min_idx = int(np.argmin(angles_rad))
    if angles_rad[min_idx] < np.deg2rad(0.05):
        return [min_idx], np.array([1.0], dtype=np.float64)

    nearest_idx = np.argsort(angles_rad)[:k]
    nearest_dist = angles_rad[nearest_idx]

    mask = nearest_dist <= (nearest_dist[0] + np.deg2rad(15.0))
    nearest_idx = nearest_idx[mask]
    nearest_dist = nearest_dist[mask]

    weights = 1.0 / (np.maximum(nearest_dist, 1e-6) ** power)
    weights /= np.sum(weights)
    return nearest_idx.tolist(), weights


# ============================================================================
# Signal Processing: Cross-Correlation ITD, Spectral Shaping & Min-Phase
# ============================================================================

def compute_measurement_itds(ir_data, delay_data, sr):
    """Extract sub-sample accurate ITD for every measurement using band-limited
    (< 1.3 kHz) cross-correlation with 8x zero-padded sinc interpolation.
    """
    num_m, _, _ = ir_data.shape
    n_fft = 1024
    up_factor = 8
    n_fft_up = n_fft * up_factor
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    lp_weight = 1.0 / (1.0 + (freqs / 1300.0) ** 6)
    hp_weight = 1.0 - 1.0 / (1.0 + (np.maximum(freqs, 1.0) / 80.0) ** 4)
    band_weight = (lp_weight * hp_weight) ** 2

    max_lag_up = int((MAX_ITD + 4) * up_factor)
    itds = np.zeros(num_m, dtype=np.float64)

    for m in range(num_m):
        if delay_data.shape[0] == num_m:
            meta_itd = float(delay_data[m, 0] - delay_data[m, 1])
        else:
            meta_itd = float(delay_data[0, 0] - delay_data[0, 1])

        spec_l = np.fft.rfft(ir_data[m, 0, :], n=n_fft)
        spec_r = np.fft.rfft(ir_data[m, 1, :], n=n_fft)

        cross_spec = spec_l * np.conj(spec_r) * band_weight
        cc = np.fft.irfft(cross_spec, n=n_fft_up)

        cc_pos = cc[: max_lag_up + 1]
        cc_neg = cc[-max_lag_up:]
        cc_window = np.concatenate([cc_neg, cc_pos])
        lags = np.arange(-max_lag_up, max_lag_up + 1, dtype=np.float64)

        pk_idx = int(np.argmax(cc_window))
        if 0 < pk_idx < len(cc_window) - 1:
            y0, y1, y2 = cc_window[pk_idx - 1], cc_window[pk_idx], cc_window[pk_idx + 1]
            denom = y0 - 2.0 * y1 + y2
            frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            lag_samples = (lags[pk_idx] + frac) / up_factor
        else:
            lag_samples = lags[pk_idx] / up_factor

        itds[m] = meta_itd + lag_samples

    return itds


def compute_diffuse_field_eq(spectra_l, spectra_r, positions, n_fft, sr, strength=DF_EQ_STRENGTH):
    """Remove stationary ear-simulator and high-frequency concha resonances."""
    cos_weights = np.maximum(np.cos(np.deg2rad(positions[:, 1])), 0.05)
    cos_weights /= np.sum(cos_weights)

    power_l = np.abs(spectra_l) ** 2
    power_r = np.abs(spectra_r) ** 2
    df_rms = np.sqrt(np.sum(0.5 * (power_l + power_r) * cos_weights[:, None], axis=0))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    ref_mask = (freqs >= 200.0) & (freqs <= 1000.0)
    ref_level = float(np.mean(df_rms[ref_mask]))
    df_log = np.log(np.maximum(df_rms / max(ref_level, 1e-12), 1e-6))

    kernel = np.hanning(7)
    kernel /= np.sum(kernel)
    df_log_smooth = np.convolve(np.pad(df_log, 3, mode="edge"), kernel, mode="valid")

    band_taper = np.ones_like(freqs)
    band_taper[freqs < 150.0] = (freqs[freqs < 150.0] / 150.0) ** 2

    hf_rolloff_db = np.zeros_like(freqs)
    hf_mask = freqs > 14000.0
    hf_rolloff_db[hf_mask] = -12.0 * ((freqs[hf_mask] - 14000.0) / 10000.0) ** 2
    hf_rolloff_nepers = hf_rolloff_db * (math.log(10.0) / 20.0)

    return (-strength * df_log_smooth * band_taper) + hf_rolloff_nepers


def apply_psychoacoustic_spatial_cues(log_l, log_r, az_deg, el_deg, freqs):
    """Preserve natural KEMAR pinna cues while gently reinforcing mid-band
    head shadow (200-2000 Hz).
    """
    az_rad = math.radians(az_deg)
    el_rad = math.radians(el_deg)

    lateral = math.sin(az_rad) * math.cos(el_rad)
    rear = max(0.0, -math.cos(az_rad)) * math.cos(el_rad)

    # 1. Gentle peak taming only above +6.0 dB (preserves pinna elevation ridges)
    lf_ref = 0.5 * (float(np.mean(log_l[1:15])) + float(np.mean(log_r[1:15])))
    ceiling = lf_ref + (3.0 * math.log(10.0) / 20.0)
    for arr in (log_l, log_r):
        excess = np.maximum(0.0, arr - ceiling)
        arr -= 0.65 * excess

    # 2. Band-pass head-shadow reinforcement centered in the 200-2000 Hz band
    hp_200 = 1.0 - 1.0 / (1.0 + (np.maximum(freqs, 1.0) / 200.0) ** 2)
    lp_2000 = 1.0 / (1.0 + (freqs / 2000.0) ** 2)
    mid_band = hp_200 * lp_2000

    contra_shadow_db = -4.5 * (lateral ** 0.9) * mid_band
    ipsi_boost_db = +1.0 * (lateral ** 0.9) * mid_band

    log_l = log_l + contra_shadow_db * (math.log(10.0) / 20.0)
    log_r = log_r + ipsi_boost_db * (math.log(10.0) / 20.0)

    # 3. Subtle rear-hemisphere pinna shadowing (> 2 kHz)
    if rear > 0.0:
        rear_hf_shelf = 1.0 - 1.0 / (1.0 + (freqs / 2000.0) ** 2)
        rear_eq_db = (-1.5 * rear) + (-4.0 * (rear ** 1.2) * rear_hf_shelf)
        rear_nepers = rear_eq_db * (math.log(10.0) / 20.0)
        log_l = log_l + rear_nepers
        log_r = log_r + rear_nepers

    return log_l, log_r


def synthesize_min_phase_31tap(log_mag_spectrum, n_fft, num_taps=NUM_FIR_TAPS):
    """Synthesize a 31-tap minimum-phase FIR filter from a log-magnitude spectrum."""
    full_log_mag = np.concatenate([log_mag_spectrum, log_mag_spectrum[-2:0:-1]])
    cepstrum = np.fft.ifft(full_log_mag).real

    w = np.zeros(n_fft, dtype=np.float64)
    w[0] = 1.0
    w[1 : n_fft // 2] = 2.0
    w[n_fft // 2] = 1.0

    min_phase_spec = np.exp(np.fft.fft(cepstrum * w))
    hrir_min = np.fft.ifft(min_phase_spec).real

    taps = hrir_min[:num_taps].copy()
    fade_start = (num_taps * 3) // 4  # Tap 23
    fade_len = num_taps - fade_start  # 8 taps (23..30)
    for i in range(fade_len):
        window = 0.5 * (1.0 + math.cos(math.pi * (i + 1) / (fade_len + 1)))
        taps[fade_start + i] *= window

    return taps


# ============================================================================
# Grid Table Builder & Global Normalization
# ============================================================================

def build_filter_table(ir_data, delay_data, positions, sr, encoding="minifloat"):
    """Build the complete 37x14 filter table."""
    n_fft = 1024
    eps = 1e-10
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    print("  Pre-computing cepstral spectra, diffuse-field EQ, and cross-correlation ITDs...")
    spectra_l = np.fft.rfft(ir_data[:, 0, :], n=n_fft)
    spectra_r = np.fft.rfft(ir_data[:, 1, :], n=n_fft)

    df_eq_log = compute_diffuse_field_eq(spectra_l, spectra_r, positions, n_fft, sr)
    log_mag_l = np.log(np.maximum(np.abs(spectra_l), eps)) + df_eq_log[None, :]
    log_mag_r = np.log(np.maximum(np.abs(spectra_r), eps)) + df_eq_log[None, :]

    raw_itds = compute_measurement_itds(ir_data, delay_data, sr)
    unit_vectors = sph_to_cart(positions[:, 0], positions[:, 1])

    float_pairs = []
    filter_index = {}

    for az_idx in range(NUM_AZIMUTHS):
        our_azimuth = AZIMUTH_MIN + az_idx * AZIMUTH_STEP
        sofa_azimuth = (360 - our_azimuth) % 360
        is_median_plane = (our_azimuth == 0 or our_azimuth == 180)

        for el_idx in range(NUM_ELEVATIONS):
            target_elevation = ELEVATION_MIN + el_idx * ELEVATION_STEP
            is_zenith = (target_elevation == 90)

            if is_zenith:
                idxs, weights = find_spherical_weights(unit_vectors, 0.0, 90.0, k=4)
            else:
                idxs, weights = find_spherical_weights(
                    unit_vectors, sofa_azimuth, target_elevation, k=3
                )

            interp_log_l = np.sum(log_mag_l[idxs] * weights[:, None], axis=0)
            interp_log_r = np.sum(log_mag_r[idxs] * weights[:, None], axis=0)
            interp_itd = float(np.sum(raw_itds[idxs] * weights))

            dist_to_sym = 0.0 if is_zenith else min(our_azimuth, 180 - our_azimuth)
            if dist_to_sym <= 0:
                sym_blend = 1.0
            elif dist_to_sym < 6:
                sym_blend = 0.5 * (1.0 + math.cos(math.pi * dist_to_sym / 6.0))
            else:
                sym_blend = 0.0

            if sym_blend > 0.0:
                avg_log = 0.5 * (interp_log_l + interp_log_r)
                interp_log_l = (1.0 - sym_blend) * interp_log_l + sym_blend * avg_log
                interp_log_r = (1.0 - sym_blend) * interp_log_r + sym_blend * avg_log
                if is_median_plane or is_zenith:
                    interp_itd = 0.0

            interp_log_l, interp_log_r = apply_psychoacoustic_spatial_cues(
                interp_log_l, interp_log_r, our_azimuth, target_elevation, freqs
            )

            left_taps = synthesize_min_phase_31tap(interp_log_l, n_fft, NUM_FIR_TAPS)
            right_taps = synthesize_min_phase_31tap(interp_log_r, n_fft, NUM_FIR_TAPS)

            itd_samples = int(round(interp_itd))
            itd_samples = max(-MAX_ITD, min(MAX_ITD, itd_samples))

            float_pairs.append((our_azimuth, target_elevation, left_taps, right_taps, itd_samples))

    max_abs_tap = 0.0
    max_h_inf = 0.0
    max_l1_norm = 0.0

    for _, _, l_taps, r_taps, _ in float_pairs:
        for taps in (l_taps, r_taps):
            peak = float(np.max(np.abs(taps)))
            l1 = float(np.sum(np.abs(taps)))
            h_inf = float(np.max(np.abs(np.fft.rfft(taps, n=512))))
            max_abs_tap = max(max_abs_tap, peak)
            max_l1_norm = max(max_l1_norm, l1)
            max_h_inf = max(max_h_inf, h_inf)

    if encoding == "minifloat":
        # In MCPX minifloat hardware, 1.0 (2048/2048) is unity gain (0 dB).
        # Scale so worst-case passband magnitude H_inf = 1.0 (0 dB) and peak tap <= 31/32,
        global_scale = min(1.0 / max(max_h_inf, 1e-12), MINIFLOAT_MAX / max(max_abs_tap, 1e-12))
    else:
        global_scale = MAX_INT8 / max(max_abs_tap, 1e-12)

    filters = []
    diagnostics = {
        "global_scale": global_scale,
        "max_abs_tap": max_abs_tap,
        "max_h_inf_db": 20.0 * math.log10(max(max_h_inf * (global_scale if encoding == "minifloat" else global_scale / 127.0), 1e-12)),
        "max_l1_ratio": max_l1_norm / max(max_abs_tap, 1e-12),
        "horizontal_table": [],
    }

    f_512 = np.fft.rfftfreq(512, d=1.0 / sr)
    mid_mask = (f_512 >= 200.0) & (f_512 <= 1500.0)

    for our_az, target_el, l_taps, r_taps, itd in float_pairs:
        if encoding == "minifloat":
            left_bytes, left_vals = quantize_minifloat_error_feedback(l_taps * global_scale)
            right_bytes, right_vals = quantize_minifloat_error_feedback(r_taps * global_scale)
        else:
            left_bytes, left_vals = quantize_int8_error_feedback(l_taps * global_scale, encoding=encoding)
            right_bytes, right_vals = quantize_int8_error_feedback(r_taps * global_scale, encoding=encoding)

        pair_index = len(filters)
        even_delay = max(0, min(MAX_ITD, abs(itd))) if itd > 0 else 0
        odd_delay = max(0, min(MAX_ITD, abs(itd))) if itd < 0 else 0

        filters.append((left_bytes, even_delay))   # even = left ear (contralateral)
        filters.append((right_bytes, odd_delay))   # odd  = right ear (ipsilateral)
        filter_index[(our_az, target_el)] = pair_index

        if target_el == 0 and our_az % 15 == 0:
            spec_l = np.abs(np.fft.rfft(left_vals, n=512))[mid_mask]
            spec_r = np.abs(np.fft.rfft(right_vals, n=512))[mid_mask]
            mid_ild_db = 20.0 * math.log10(
                max(float(np.sqrt(np.mean(spec_r ** 2))), 1e-9)
                / max(float(np.sqrt(np.mean(spec_l ** 2))), 1e-9)
            )
            pk_l = int(round(np.max(np.abs(left_vals)) * 2048.0))
            pk_r = int(round(np.max(np.abs(right_vals)) * 2048.0))
            diagnostics["horizontal_table"].append((our_az, itd, pk_l, pk_r, mid_ild_db))

    return filters, filter_index, diagnostics


# ============================================================================
# Header File Generation & Datasheet Summary
# ============================================================================

def write_header_file(filepath, filters, filter_index, sofa_filename, encoding="minifloat"):
    """Write a single .h file containing both filter data and index table."""
    if encoding == "minifloat":
        enc_desc = (
            "8-bit MCPX sign-magnitude minifloats (1-bit sign, 3-bit exp, 4-bit mant; "
            "(16+m)*2^(e-1)/2048 or m/2048 when e=0)"
        )
    elif encoding == "signmag":
        enc_desc = "signed 8-bit in linear sign-magnitude (bit 7 = sign, bits 6..0 = magnitude)"
    else:
        enc_desc = "signed 8-bit in two's complement (stored as uint8_t)"

    with open(filepath, "w") as f:
        f.write("/*\n")
        f.write(" * SONICOM HRTF data (Cepstral Min-Phase, 1-3-4 Minifloat, 48kHz) for Xbox APU.\n")
        f.write(" *\n")
        f.write(f" * Source: {sofa_filename}\n")
        f.write(" * Dataset: https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/\n")
        f.write(" * Information: https://aes.org/publications/elibrary-page/?id=22128\n")
        f.write(" * License: Creative Commons Attribution 4.0 (CC BY 4.0)\n")
        f.write(" *\n")
        f.write(" * Generated by convert_hrtf.py (Upgraded DSP Pipeline v5)\n")
        f.write(" */\n\n")

        f.write("#ifndef NXAUDIO_HRTF_H\n")
        f.write("#define NXAUDIO_HRTF_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("/* 31-tap FIR filter + interaural time delay */\n")
        f.write("typedef struct {\n")
        f.write("    uint8_t coeff[31];\n")
        f.write("    uint8_t delay;\n")
        f.write("} nxaudio_hrtf_filter_t;\n\n")

        num_filters = len(filters)
        f.write(f"#define NXAUDIO_HRTF_NUM_FILTERS    {num_filters}\n")
        f.write(
            f"#define NXAUDIO_HRTF_NUM_AZIMUTHS   {NUM_AZIMUTHS}"
            f"  /* {AZIMUTH_MIN} to {AZIMUTH_MAX} in {AZIMUTH_STEP}-degree steps */\n"
        )
        f.write(
            f"#define NXAUDIO_HRTF_NUM_ELEVATIONS {NUM_ELEVATIONS}"
            f"  /* {ELEVATION_MIN} to {ELEVATION_MAX} in {ELEVATION_STEP}-degree steps */\n"
        )
        f.write(f"#define NXAUDIO_HRTF_AZIMUTH_STEP   {AZIMUTH_STEP}\n")
        f.write(f"#define NXAUDIO_HRTF_ELEVATION_STEP {ELEVATION_STEP}\n\n")

        f.write("/*\n")
        f.write(" * Filter coefficients stored as consecutive left/right pairs.\n")
        f.write(f" * Coefficients are {enc_desc}.\n")
        f.write(" * Delay is the ITD in samples at 48kHz for the delayed ear.\n")
        f.write(" */\n")
        f.write("static const nxaudio_hrtf_filter_t nxaudio_hrtf_filters[] = {\n")

        tap_header = "/* tap:" + "".join(f" {t:4d} " for t in range(NUM_FIR_TAPS)) + "  ITD */"
        f.write(tap_header + "\n")

        for i, (coeff_bytes, delay) in enumerate(filters):
            coeff_str = ", ".join(f"0x{b:02X}" for b in coeff_bytes)
            f.write(f"    {{ {{ {coeff_str} }}, {delay:2d} }}, /* {i:4d} */\n")

        f.write("};\n\n")

        f.write("/*\n")
        f.write(" * Index table: nxaudio_hrtf_index[azimuth_idx][elevation_idx]\n")
        f.write(" * Maps grid position to index into nxaudio_hrtf_filters[].\n")
        f.write(" * Each entry points to a left/right filter pair (2 consecutive entries).\n")
        f.write(" */\n")
        f.write(
            f"static const uint16_t nxaudio_hrtf_index"
            f"[{NUM_AZIMUTHS}][{NUM_ELEVATIONS}] = {{\n"
        )

        f.write("  /* elevation= ")
        for el_idx in range(NUM_ELEVATIONS):
            elev_deg = ELEVATION_MIN + el_idx * ELEVATION_STEP
            f.write(f"   {elev_deg:3d}")
        f.write("  */\n")

        for az_idx in range(NUM_AZIMUTHS):
            azim_deg = AZIMUTH_MIN + az_idx * AZIMUTH_STEP
            f.write(f"  /* az={azim_deg:3d} deg */ {{ ")
            entries = [
                f"{filter_index.get((azim_deg, ELEVATION_MIN + el_idx * ELEVATION_STEP), 0):4d}"
                for el_idx in range(NUM_ELEVATIONS)
            ]
            f.write(", ".join(entries) + " },\n")

        f.write("};\n\n")
        f.write("#endif /* NXAUDIO_HRTF_H */\n")


def print_datasheet(diagnostics, filepath, num_filters, encoding):
    """Print hardware verification datasheet for the generated HRTF table."""
    if encoding == "minifloat":
        enc_label = "8-Bit MCPX Minifloat (1s:3e:4m, /2048 hardware scale)"
    elif encoding == "signmag":
        enc_label = "8-Bit Linear Sign-Magnitude"
    else:
        enc_label = "8-Bit Two's Complement"

    print("\n" + "=" * 74)
    print("ORIGINAL XBOX MCPX APU - HRTF HARDWARE DATASHEET (v5)")
    print("=" * 74)
    print(f"  Target Header           : {filepath}")
    print(f"  Coefficient Encoding    : {enc_label}")
    print(f"  Total Filter Structs    : {num_filters} ({num_filters // 2} L/R pairs)")
    print(f"  ROM Footprint           : {num_filters * 32 + NUM_AZIMUTHS * NUM_ELEVATIONS * 2:,} bytes")
    print(f"  Hardware Sample Rate    : {SAMPLE_RATE} Hz (20.83 us/sample)")
    print(f"  FIR Window Length       : {NUM_FIR_TAPS} taps (645.83 us)")
    print(f"  Peak Passband Gain H_inf: {diagnostics['max_h_inf_db']:+.2f} dB (rel. unity 2048/2048)")
    print(f"  Worst-Case L1 / Peak    : {diagnostics['max_l1_ratio']:.2f}x")
    print("-" * 74)
    print("  Horizontal Plane (Elevation = 0 deg) Verification Profile:")
    print("  Azimuth | ITD (smp / us)  | Peak L (/2048) | Peak R (/2048) | 0.2-1.5kHz ILD")
    print("  --------+-----------------+----------------+----------------+---------------")
    for az, itd, pk_l, pk_r, mid_ild_db in diagnostics["horizontal_table"]:
        itd_us = itd * (1e6 / SAMPLE_RATE)
        print(f"   {az:3d} deg | {itd:3d} ({itd_us:6.1f} us) |      {pk_l:4d}      |      {pk_r:4d}      |   {mid_ild_db:+6.2f} dB")
    print("=" * 74)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SONICOM HRTF (SOFA) to Optimal Original Xbox APU C header"
    )
    parser.add_argument("sofa_file", help="Path to SONICOM .sofa file")
    parser.add_argument(
        "--output", "-o", default="nxaudio_hrtf.h", help="Output header file path"
    )
    parser.add_argument(
        "--encoding",
        "-e",
        choices=["minifloat", "signmag", "twos"],
        default="minifloat",
        help="Coefficient byte encoding: 'minifloat' (default, hardware 1-3-4 minifloat), 'signmag', or 'twos'",
    )
    args = parser.parse_args()

    if not os.path.exists(args.sofa_file):
        print(f"ERROR: File not found: {args.sofa_file}")
        sys.exit(1)

    ir_data, delay_data, positions, sr = read_sofa(args.sofa_file)
    filters, filter_index, diagnostics = build_filter_table(
        ir_data, delay_data, positions, sr, encoding=args.encoding
    )
    write_header_file(
        args.output,
        filters,
        filter_index,
        os.path.basename(args.sofa_file),
        encoding=args.encoding,
    )
    print_datasheet(diagnostics, args.output, len(filters), args.encoding)


if __name__ == "__main__":
    main()