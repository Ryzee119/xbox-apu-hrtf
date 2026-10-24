#!/usr/bin/env python3
"""
Original Xbox MCPX APU - HRTF Header Verification & Acoustic Datasheet Plotter

Parses nxaudio_hrtf.h directly, decodes 8-bit MCPX 1-3-4 sign-magnitude minifloats
((16 + m) * 2^(e - 1) / 2048 or m / 2048 when e == 0), and generates a 6-panel
verification dashboard:
  1. Ipsilateral & Contralateral 31-tap Magnitude Responses (0, 30, 60, 90, 135, 180 deg)
  2. 360-Degree Polar ITD Profile vs. Woodworth Spherical Head Model
  3. 360-Degree Polar ILD Profile (Broadband RMS vs. High-Frequency Pinna Band)
  4. Horizontal Plane Binaural Spectrogram (-180 to +180 deg Azimuth vs. Frequency)
  5. Median Plane Elevation Pinna-Notch Map (-40 to +90 deg Elevation vs. Frequency)
  6. Time-Domain 31-Tap IR & MCPX Hardware Linear-Interpolation Comb-Filter Check

Usage:
  python3 verify_hrtf.py [nxaudio_hrtf.h] [--save hrtf_datasheet.png] [--compare old_hrtf.h]
"""

import argparse
from dataclasses import dataclass
import os
import re
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: numpy and matplotlib are required. Install with: pip install numpy matplotlib")
    sys.exit(1)

SAMPLE_RATE = 48000
NUM_TAPS = 31
AZIMUTHS = np.arange(0, 185, 5)       # 37 entries: 0..180 deg (step 5)
ELEVATIONS = np.arange(-40, 100, 10)  # 14 entries: -40..+90 deg (step 10)


@dataclass
class HRTFHeaderData:
    name: str
    filters: list            # List of (np.ndarray[31] float in [-1..1], delay_samples)
    index_table: np.ndarray  # Shape: (37, 14)
    encoding: str            # 'minifloat', 'signmag', or 'twos'


def _decode_minifloat_byte(b):
    """Decode an 8-bit MCPX 1-3-4 sign-magnitude minifloat byte to float [-31/32, +31/32]."""
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 4) & 0x07
    mant = b & 0x0F
    if exp == 0:
        mag = mant / 2048.0
    else:
        mag = ((16 + mant) << (exp - 1)) / 2048.0
    return sign * mag


def _decode_taps(raw_bytes, encoding="minifloat"):
    """Decode 8-bit filter taps to normalized float [-1.0, +1.0]."""
    if encoding == "minifloat":
        return [_decode_minifloat_byte(v) for v in raw_bytes]
    if encoding == "signmag":
        return [(-(v & 0x7F) if (v & 0x80) else v) / 127.0 for v in raw_bytes]
    return [((v - 256) if v > 127 else v) / 127.0 for v in raw_bytes]


def _detect_encoding(all_raw_rows, text="", requested="auto"):
    """Auto-detect whether header uses MCPX 1-3-4 minifloat or legacy two's complement."""
    if requested in ("minifloat", "signmag", "twos"):
        return requested

    if "two's complement" in text.lower() and "minifloat" not in text.lower():
        neg_bytes = [b for row in all_raw_rows for b in row if b >= 0x80]
        low_neg = sum(1 for b in neg_bytes if 0x81 <= b <= 0x9F)
        high_neg = sum(1 for b in neg_bytes if 0xE0 <= b <= 0xFF)
        if high_neg > low_neg:
            return "twos"

    return "minifloat"


def parse_hrtf_header(filepath, encoding="auto"):
    """Parse coeff[31], delay, and nxaudio_hrtf_index[37][14] from nxaudio_hrtf.h."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Header file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Match filter rows: { { 0x.., ... }, delay }
    filter_pattern = re.compile(
        r"\{\s*\{\s*((?:0x[0-9A-Fa-f]{2}\s*,\s*){30}0x[0-9A-Fa-f]{2})\s*\}\s*,\s*(\d+)\s*\}"
    )
    raw_rows = []
    delays = []
    for match in filter_pattern.finditer(text):
        hex_vals = [int(x.strip(), 16) for x in match.group(1).split(",")]
        raw_rows.append(hex_vals)
        delays.append(int(match.group(2)))

    if not raw_rows:
        raise ValueError(f"No valid 31-tap filter entries found in {filepath}")

    detected_enc = _detect_encoding(raw_rows, text=text, requested=encoding)
    filters = [
        (np.array(_decode_taps(row, detected_enc), dtype=np.float64), delay)
        for row, delay in zip(raw_rows, delays)
    ]

    # Extract index table block
    idx_block_match = re.search(
        r"nxaudio_hrtf_index\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\};",
        text,
        re.DOTALL,
    )
    if not idx_block_match:
        raise ValueError(f"Could not find nxaudio_hrtf_index table in {filepath}")

    row_pattern = re.compile(r"\{\s*([\d\s,]+)\s*\}")
    index_rows = []
    for row_match in row_pattern.finditer(idx_block_match.group(1)):
        nums = [int(n.strip()) for n in row_match.group(1).split(",") if n.strip()]
        if len(nums) == len(ELEVATIONS):
            index_rows.append(nums)

    index_table = np.array(index_rows, dtype=np.int32)
    if index_table.shape != (len(AZIMUTHS), len(ELEVATIONS)):
        raise ValueError(
            f"Unexpected index table shape {index_table.shape}, "
            f"expected {(len(AZIMUTHS), len(ELEVATIONS))}"
        )

    return HRTFHeaderData(
        name=os.path.basename(filepath),
        filters=filters,
        index_table=index_table,
        encoding=detected_enc,
    )


def get_ear_pair(dataset: HRTFHeaderData, az_deg, el_deg=0):
    """Retrieve (left_taps, right_taps, left_delay, right_delay) for any 360-deg angle.

    Mirrors left-hemisphere angles (-180..0) by swapping Left/Right ears,
    matching the runtime behavior of nxAudioHRTFSetParamsFromAngles.
    """
    az_wrapped = ((az_deg + 180) % 360) - 180
    swap_ears = az_wrapped < 0
    lookup_az = abs(az_wrapped)

    az_idx = int(round((lookup_az - AZIMUTHS[0]) / 5.0))
    el_idx = int(round((el_deg - ELEVATIONS[0]) / 10.0))
    az_idx = max(0, min(len(AZIMUTHS) - 1, az_idx))
    el_idx = max(0, min(len(ELEVATIONS) - 1, el_idx))

    pair_idx = dataset.index_table[az_idx, el_idx]
    left_taps, left_delay = dataset.filters[pair_idx]
    right_taps, right_delay = dataset.filters[pair_idx + 1]

    if swap_ears:
        return right_taps, left_taps, right_delay, left_delay
    return left_taps, right_taps, left_delay, right_delay


def compute_mag_db(taps_norm, n_fft=1024):
    """Compute magnitude response in dB relative to hardware unity gain (1.0 = 2048/2048 = 0 dB)."""
    spec = np.fft.rfft(taps_norm, n=n_fft)
    return 20.0 * np.log10(np.maximum(np.abs(spec), 1e-6))


def plot_datasheet(dataset: HRTFHeaderData, comp_dataset: HRTFHeaderData = None, save_path=None):
    n_fft = 1024
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    freqs_khz = freqs / 1000.0

    enc_map = {
        "minifloat": "1-3-4 Minifloat (/2048)",
        "signmag": "Linear Sign-Magnitude (/127)",
        "twos": "Two's Complement (/127)",
    }
    enc_label = enc_map.get(dataset.encoding, dataset.encoding)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=120)
    fig.suptitle(
        f"Original Xbox MCPX APU — 31-Tap HRTF Verification Datasheet "
        f"({dataset.name} | 37x14 Grid, 5° Az / 10° El Step, {enc_label})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.26)

    # ------------------------------------------------------------------------
    # Panel 1: Horizontal Magnitude Responses (Ipsilateral vs. Contralateral)
    # ------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    test_angles = [
        (0, "#00e5ff"),
        (30, "#76ff03"),
        (60, "#ffea00"),
        (90, "#ff3d00"),
        (135, "#d500f9"),
        (180, "#90a4ae"),
    ]
    peak_db = 8.0
    for az, color in test_angles:
        l_taps, r_taps, _, _ = get_ear_pair(dataset, az, 0)
        mag_r = compute_mag_db(r_taps, n_fft)
        peak_db = max(peak_db, float(np.max(mag_r)) + 2.0)
        ax1.semilogx(freqs, mag_r, color=color, lw=1.6, label=f"{az}° R (Ipsi)")
        if az not in (0, 180):
            mag_l = compute_mag_db(l_taps, n_fft)
            peak_db = max(peak_db, float(np.max(mag_l)) + 2.0)
            ax1.semilogx(freqs, mag_l, color=color, lw=1.0, ls="--", alpha=0.65)

    ax1.set_xlim(150, 20000)
    ax1.set_ylim(-32, min(20, np.ceil(peak_db)))
    ax1.set_title("1. Horizontal Magnitude (Solid=Right/Ipsi, Dashed=Left/Contra)", fontsize=10)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Gain rel. Unity 2048/2048 (dB)")
    ax1.grid(True, which="both", alpha=0.2)
    ax1.legend(loc="lower left", fontsize=8, ncol=2)

    # ------------------------------------------------------------------------
    # Panel 2: 360-Degree Polar ITD Profile (Samples & Microseconds)
    # ------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)  # Clockwise: 0=Front, 90=Right, 180=Rear

    full_az = np.arange(-180, 185, 5)
    thetas = np.deg2rad(full_az)
    itds = [max(get_ear_pair(dataset, az, 0)[2:]) for az in full_az]

    # Folded Woodworth spherical head model reference (r = 8.75 cm -> ~31.5 samples max at 90 deg)
    abs_th = np.abs(thetas)
    lat_th = np.minimum(abs_th, np.pi - abs_th)
    woodworth = 12.25 * (lat_th + np.sin(lat_th))

    ax2.plot(thetas, woodworth, color="#ffffff", ls=":", lw=1.3, alpha=0.6, label="Woodworth Model")
    ax2.plot(thetas, itds, color="#00e5ff", lw=2.2, label="Header ITD (samples)")

    if comp_dataset:
        comp_itds = [max(get_ear_pair(comp_dataset, az, 0)[2:]) for az in full_az]
        ax2.plot(thetas, comp_itds, color="#ff5252", lw=1.5, ls="--", label=f"Compare ({comp_dataset.name})")

    ax2.set_ylim(0, 42)
    ax2.set_title("2. Horizontal ITD Polar Profile (Max Limit = 42 smp)", fontsize=10, pad=14)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.15, -0.08), fontsize=8)

    # ------------------------------------------------------------------------
    # Panel 3: Azimuth ILD Profile (Broadband vs. High-Frequency > 3 kHz)
    # ------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ild_broadband = []
    ild_hf = []
    hf_mask = (freqs >= 3000.0) & (freqs <= 16000.0)

    for az in AZIMUTHS:
        l_taps, r_taps, _, _ = get_ear_pair(dataset, az, 0)
        rms_l = np.sqrt(np.mean(l_taps ** 2))
        rms_r = np.sqrt(np.mean(r_taps ** 2))
        ild_broadband.append(20.0 * np.log10(max(rms_r, 1e-9) / max(rms_l, 1e-9)))

        spec_l = np.abs(np.fft.rfft(l_taps, n=n_fft))[hf_mask]
        spec_r = np.abs(np.fft.rfft(r_taps, n=n_fft))[hf_mask]
        ild_hf.append(20.0 * np.log10(np.sqrt(np.mean(spec_r ** 2)) / max(np.sqrt(np.mean(spec_l ** 2)), 1e-9)))

    ax3.plot(AZIMUTHS, ild_broadband, color="#76ff03", lw=2.0, marker="o", ms=3, label="Broadband RMS ILD")
    ax3.plot(AZIMUTHS, ild_hf, color="#ff9100", lw=1.8, ls="--", label="High-Freq ILD (3–16 kHz)")

    if comp_dataset:
        comp_ild_bb = []
        for az in AZIMUTHS:
            l_t, r_t, _, _ = get_ear_pair(comp_dataset, az, 0)
            comp_ild_bb.append(
                20.0 * np.log10(max(np.sqrt(np.mean(r_t ** 2)), 1e-9) / max(np.sqrt(np.mean(l_t ** 2)), 1e-9))
            )
        ax3.plot(
            AZIMUTHS,
            comp_ild_bb,
            color="#ff5252",
            lw=1.3,
            ls="-.",
            alpha=0.8,
            label=f"Compare ({comp_dataset.name}) BB ILD",
        )

    ax3.axvline(90, color="#ffffff", ls=":", alpha=0.35, label="90° Bright Spot Dip")
    ax3.set_xlim(0, 180)
    ax3.set_xticks(np.arange(0, 181, 30))
    ax3.set_title("3. Interaural Level Difference (Right vs. Left Ear)", fontsize=10)
    ax3.set_xlabel("Azimuth (deg)")
    ax3.set_ylabel("ILD (dB)")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------------
    # Panel 4: Full 360-Deg Horizontal Plane HRTF Carpet Spectrogram (Right Ear)
    # ------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    horiz_map = np.zeros((len(full_az), len(freqs)))
    for i, az in enumerate(full_az):
        _, r_taps, _, _ = get_ear_pair(dataset, az, 0)
        horiz_map[i, :] = compute_mag_db(r_taps, n_fft)

    f_mask = (freqs_khz >= 0.5) & (freqs_khz <= 18.0)
    im4 = ax4.pcolormesh(
        freqs_khz[f_mask],
        full_az,
        horiz_map[:, f_mask],
        shading="gouraud",
        cmap="inferno",
        vmin=-28,
        vmax=max(6.0, float(np.percentile(horiz_map[:, f_mask], 99))),
    )
    ax4.axhline(0, color="#00e5ff", ls="--", lw=0.8, alpha=0.6)
    ax4.set_title("4. Right-Ear Horizontal Spectrogram (-180° to +180°)", fontsize=10)
    ax4.set_xlabel("Frequency (kHz)")
    ax4.set_ylabel("Source Azimuth (deg)")
    ax4.set_yticks(np.arange(-180, 181, 45))
    fig.colorbar(im4, ax=ax4, label="dB")

    # ------------------------------------------------------------------------
    # Panel 5: Median Plane Elevation Spectrogram (Azimuth = 0 deg)
    # ------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    elev_map = np.zeros((len(ELEVATIONS), len(freqs)))
    for i, el in enumerate(ELEVATIONS):
        _, r_taps, _, _ = get_ear_pair(dataset, 0, el)
        elev_map[i, :] = compute_mag_db(r_taps, n_fft)

    im5 = ax5.pcolormesh(
        freqs_khz[f_mask],
        ELEVATIONS,
        elev_map[:, f_mask],
        shading="gouraud",
        cmap="magma",
        vmin=-25,
        vmax=max(6.0, float(np.percentile(elev_map[:, f_mask], 99))),
    )
    ax5.set_title("5. Front Median-Plane Elevation Cue Map (Az = 0°, -40°..+90°)", fontsize=10)
    ax5.set_xlabel("Frequency (kHz)")
    ax5.set_ylabel("Elevation (deg)")
    ax5.set_yticks(np.arange(-40, 91, 20))
    fig.colorbar(im5, ax=ax5, label="dB")

    # ------------------------------------------------------------------------
    # Panel 6: Time-Domain 31-Tap IR & MCPX Linear-Interpolation Test
    # ------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    taps_x = np.arange(NUM_TAPS)
    l_90, r_90, _, _ = get_ear_pair(dataset, 90, 0)
    _, r_85, _, _ = get_ear_pair(dataset, 85, 0)

    # Display in 11-bit hardware accumulator units (/2048)
    r_90_hw = r_90 * 2048.0
    l_90_hw = l_90 * 2048.0
    interp_mid_hw = 0.5 * (r_85 + r_90) * 2048.0

    ax6.step(taps_x, r_90_hw, where="mid", color="#00e5ff", lw=1.8, label="90° Ipsilateral (Right)")
    ax6.step(taps_x, l_90_hw, where="mid", color="#ff3d00", lw=1.5, label="90° Contralateral (Left)")
    ax6.plot(taps_x, interp_mid_hw, color="#ffea00", ls=":", lw=1.4, label="85°/90° Hardware Interp Midpoint")

    ax6.axhline(0, color="#ffffff", lw=0.6, alpha=0.4)
    ax6.set_xlim(0, 30)
    peak_hw = max(1024.0, float(np.max(np.abs(r_90_hw))) * 1.15)
    ax6.set_ylim(-peak_hw, peak_hw)
    ax6.set_title(f"6. 31-Tap Hardware Impulse ({enc_label})", fontsize=10)
    ax6.set_xlabel("FIR Tap Index (0..30 @ 48 kHz)")
    ax6.set_ylabel("Decoded Tap Weight (x 1/2048)")
    ax6.grid(True, alpha=0.2)
    ax6.legend(loc="upper right", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved verification datasheet to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Verify and plot Xbox APU nxaudio_hrtf.h")
    parser.add_argument("header", nargs="?", default="nxaudio_hrtf.h", help="Path to nxaudio_hrtf.h")
    parser.add_argument("--compare", "-c", help="Optional second header file to compare ITD/ILD against")
    parser.add_argument(
        "--encoding",
        choices=["auto", "minifloat", "signmag", "twos"],
        default="auto",
        help="Coefficient byte encoding: auto (default -> minifloat), minifloat (1-3-4 /2048), signmag, or twos",
    )
    parser.add_argument("--save", "-s", help="Save plot to PNG instead of opening an interactive window")
    args = parser.parse_args()

    try:
        dataset = parse_hrtf_header(args.header, encoding=args.encoding)
        comp_dataset = parse_hrtf_header(args.compare, encoding=args.encoding) if args.compare else None
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    plot_datasheet(dataset, comp_dataset=comp_dataset, save_path=args.save)


if __name__ == "__main__":
    main()
