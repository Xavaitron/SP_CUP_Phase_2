# Task 1 — Anechoic Source Separation

Angle-conditioned audio source separation under **anechoic** conditions (RT60 = 0.0) using a DCCRN model (~10M parameters).

---

## 📁 Folder Contents

| File | Description |
|------|-------------|
| `process_task1.py` | Self-contained inference script (model definition + inference logic) |
| `anechoic_Conformer.pth` | Trained model weights (anechoic condition) |
| `mixture_signal{1,2,3}.wav` | Stereo input mixtures (16kHz, 4s) |
| `target_signal{1,2,3}.wav` | Ground-truth target signals (mono, 16kHz) |
| `interference_signal{1,2,3}.wav` | Interference sources (mono, 16kHz) |
| `Task1_Anechoic_5dB_sample{1,2,3}.mat` | MATLAB data files with sample metadata |
| `requirements.txt` | Python dependencies |

**Sample mapping:** 1 = Male+Female, 2 = Male+Music, 3 = Male+Noise

---

## 🔧 Setup

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
cd Task1_Anechoic
pip install -r requirements.txt
```

> **Note:** If you have a CUDA-compatible GPU, install the appropriate PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally/) for GPU acceleration. The script defaults to CPU.

---

## 🚀 How to Run

### Process a single sample

```bash
python process_task1.py --sample <1|2|3>
```

### Examples

```bash
# Process sample 1 (Male + Female) on CPU
python process_task1.py --sample 1

# Process sample 2 (Male + Music) on GPU
python process_task1.py --sample 2 --device cuda

# Process sample 3 (Male + Noise) with custom angle
python process_task1.py --sample 3 --angle 45
```

### Process all samples at once

```bash
python process_task1.py --sample 1
python process_task1.py --sample 2
python process_task1.py --sample 3
```

### Command-line arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--sample` | `-s` | *(required)* | Sample number: 1, 2, or 3 |
| `--angle` | `-a` | `90` | Target source angle in degrees (0–180) |
| `--device` | `-d` | `cpu` | Compute device: `cpu` or `cuda` |

---

## 📤 Output

Running the script produces:
- `processed_signal{1,2,3}.wav` — The separated target audio (mono, 16kHz)

If ground-truth `target_signal*.wav` files are present, the script automatically computes and prints evaluation metrics:
- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio, in dB)
- **PESQ** (Perceptual Evaluation of Speech Quality, -0.5 to 4.5)
- **STOI** (Short-Time Objective Intelligibility, 0 to 1)

---

## 🏗️ Model Details

- **Architecture:** DCCRNConformer (Deep Complex Convolution Recurrent Network + Dual-Path Conformer)
- **Parameters:** ~10M
- **Audio config:** 16kHz sample rate, STFT with n_fft=512, hop_length=128, fixed 3s input
- **Angle conditioning:** Target angle is injected via an MLP at the bottleneck layer
- **Condition:** Trained on anechoic data (RT60 = 0.0, SIR = 0 dB, SNR = 5 dB)
