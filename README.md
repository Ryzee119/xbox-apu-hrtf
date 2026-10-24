# Xbox APU HRTF Converter

A Python DSP utility suite to convert standard **SOFA** (Spatially Oriented Format for Acoustics) HRTF datasets into the proprietary 31-tap, 8-bit sign-magnitude minifloat C-header format (`nxaudio_hrtf.h`) expected by the original Xbox **MCPX Audio Processing Unit (APU)** and [`nxdk-audio`](https://github.com/Ryzee119/nxdk-audio).

![Original Xbox MCPX APU — 31-Tap HRTF Verification Datasheet](https://github.com/Ryzee119/xbox-apu-hrtf/releases/latest/download/hrtf_datasheet.png)

## Overview

The original Xbox MCPX Voice Processor (VP) includes a dedicated hardware 3D HRTF engine operating at **48 kHz**. To fit within the hardware register and memory architecture, directional head-related impulse responses (HRIRs) must adhere to strict hardware constraints:

- **31-Tap FIR Filter (`645.8 µs` at 48 kHz):** The hardware convolves 31 coefficients per ear (`coeff[31]`), uploaded as interleaved L/R pairs via `NV1BA0_PIO_SET_HRIR` and `NV1BA0_PIO_SET_HRIR_X`.
- **8-Bit Sign-Magnitude Minifloat (`1s:3e:4m`, `/ 2048` scale):** Filter taps are **not** linear two's complement integers. Each byte `b` is decoded by the MCPX hardware as a sign-magnitude minifloat with a 1-bit sign (`s = (b >> 7) & 1`), 3-bit exponent (`e = (b >> 4) & 7`), and 4-bit mantissa (`m = b & 15`) over an 11-bit (`2048`) denominator:
  - **Subnormal (`e == 0`):** `(-1)^s * m / 2048` (steps of `1/2048`, e.g., `0x01 = 1/2048`, `0x81 = -1/2048`)
  - **Normalized (`1 <= e <= 7`):** `(-1)^s * (16 + m) * 2^(e - 1) / 2048` (e.g., `0x20 = 1/64`, `0x40 = 1/16`, `0x7F = 31/32 = 0.96875`)
  - This provides **66.2 dB of dynamic range** (11-bit + sign precision for quiet contralateral and decaying tail taps) with unity passband gain at `2048 / 2048 = 1.0` (`0 dB`).
- **Separate Hardware ITD (`0` to `42` samples / `875 µs`):** The Interaural Time Delay (`delay`) is stripped from the impulse response and applied via a dedicated per-ear hardware delay line (programmed in signed 7.9 fixed-point format in the upper 16 bits of `NV1BA0_PIO_SET_HRIR_X`).
- **Hemisphere Mirroring (`37 × 14` Lookup Grid):** The driver indexes `37` azimuths (`0°` to `180°` in `5°` steps) by `14` elevations (`-40°` to `+90°` in `10°` steps) for the right hemisphere (`518` L/R pairs = `1,036` filter structs, `34,188 bytes` total ROM footprint) and mirrors Left/Right ear channels (and negates ITD) for sources in the left hemisphere (`-180°` to `0°`).

## DSP Pipeline (`convert_hrtf.py`)

1. **Band-Limited Cross-Correlation ITD Extraction:** Extracts sub-sample ITDs from the raw HRIRs using 8x oversampled sinc cross-correlation weighted to the `< 1.3 kHz` phase-locking band.
2. **Spherical ($S^2$) Great-Circle Interpolation:** Interpolates log-magnitude spectra and ITDs onto the `37 × 14` grid using inverse Great-Circle distance weighting, with smooth bilateral symmetry feathering within `6°` of the median plane (`0°`/`180°`) and zenith (`+90°`).
3. **Full Diffuse-Field De-Resonance:** Computes the area-weighted diffuse-field average across the sphere and removes stationary ear-canal and concha resonances (`> 150 Hz`) plus ultrasonic energy (`> 14 kHz`), yielding a flat, natural midrange passband matching Microsoft's `hrtffull.flt`.
4. **Band-Pass Midrange ILD & Rear Pinna Cues:** Reinforces `200–2000 Hz` head shadowing while preserving natural low-frequency diffraction below `200 Hz`, and applies subtle high-frequency pinna shadowing in the rear hemisphere (`90°–180°`) for front/back externalization.
5. **Cepstral Minimum-Phase 31-Tap Synthesis:** Reconstructs minimum-phase impulse responses via the real cepstrum so acoustic energy is concentrated in Taps `0–1`, eliminating comb-filtering when the MCPX hardware linearly interpolates between adjacent filter entries.
6. **Unity $H_\infty$ Normalization & Error-Feedback Quantization:** Scales the table so peak passband magnitude ($H_\infty$) is unity (`0 dB`) and quantizes coefficients into `1-3-4` minifloats using 1st-order error-feedback noise shaping.

## Requirements

- Python 3.8+
- `h5py` and `numpy` (for `convert_hrtf.py`)
- `matplotlib` (optional, for generating the verification datasheet via `verify_hrtf.py`)

```bash
pip install h5py numpy matplotlib
