# SP CUP Phase 2 - Audio Source Separation with Conformer

Angle-conditioned audio source separation using a DCCRNConformer architecture for IEEE Signal Processing Cup 2026.

---

## 📁 Project Structure

```
SP_CUP_Phase_2/
├── Dataset Generation/              # MATLAB scripts for synthetic dataset creation
│   ├── train_anechoic.m             # Training data (150k samples, RT60=0.0)
│   ├── train_reverb.m               # Training data (150k samples, RT60=0.5)
│   ├── test_anechoic.m              # Test data (5k samples, fixed 90°/40° angles)
│   └── test_reverb.m                # Test data (5k samples, fixed 90°/40° angles)
│
├── Model Inference/                 # Python training, testing, and inference
│   ├── train_Conformer.py           # Training script
│   ├── test_Conformer.py            # Evaluation script (SI-SDR, STOI, PESQ)
│   ├── inference_Conformer.py       # Single-file inference
│   ├── anechoic_Conformer.pth       # Trained model (anechoic)
│   ├── reverb_Conformer.pth         # Trained model (reverberant)
│   ├── evaluation_anechoic/         # Evaluation outputs
│   └── evaluation_reverb/           # Evaluation outputs
│
├── Submission/                      # Self-contained competition submission
│   ├── Task1_Anechoic/
│   │   ├── Task1_Anechoic_5dB.mat
│   │   ├── anechoic_Conformer.pth
│   │   ├── process_task1.py
│   │   └── [audio files]
│   └── Task2_Reverberant/
│       ├── Task2_Reverberant_5dB.mat
│       ├── reverb_Conformer.pth
│       ├── process_task2.py
│       └── [audio files]
│
├── prepare_submission.m             # Generates submission folder from evaluation
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🔧 Requirements

### Python
```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.6.0 | Deep learning |
| torchaudio | 2.6.0 | Audio I/O |
| torchmetrics | 1.8.2 | PESQ, STOI, SI-SDR |
| soundfile | latest | Audio backend |
| pesq, pystoi | latest | Metrics |

### MATLAB
- MATLAB R2020b+
- Signal Processing Toolbox
- Parallel Computing Toolbox
- `rir_generator` MEX function

---

## 🚀 Pipeline

### 1. Dataset Generation (MATLAB)

```matlab
cd "Dataset Generation"
train_anechoic   % 150k samples, RT60=0.0, random angles
train_reverb     % 150k samples, RT60=0.5, random angles
test_anechoic    % 5k samples, RT60=0.0, fixed angles (90°/40°)
test_reverb      % 5k samples, RT60=0.5, fixed angles (90°/40°)
```

**Output per sample:**
```
sample_XXXXX/
├── mixture.wav      # Stereo (target + interferer + noise)
├── target.wav       # Ground truth
├── interference.wav # Scaled interferer
└── meta.json        # {target_angle, interf_angle, rt60, ...}
```

**Settings:** SIR=0dB, SNR=5dB, 16kHz, 4s duration

---

### 2. Training

```bash
cd "Model Inference"
python train_Conformer.py
```

Edit config in script:
```python
DATASET_ROOT = r"../Train_Dataset/reverb"  # or anechoic
RESUME_FROM = "reverb_Conformer.pth"       # or None
```

---

### 3. Evaluation

```bash
cd "Model Inference"
python test_Conformer.py
```

Edit config:
```python
MODEL_PATH = "anechoic_Conformer.pth"
TEST_DATASET_ROOT = r"../Test_Dataset/anechoic"
OUTPUT_DIR = "evaluation_anechoic"
```

**Outputs:** Best samples by category (Overall, Male+Female, Male+Music, Male+Noise)

---

### 4. Single-File Inference

```bash
python inference_Conformer.py -i input.wav -a 90 -o output.wav -m reverb_Conformer.pth -d cuda
```

| Arg | Description |
|-----|-------------|
| `-i` | Input stereo audio |
| `-a` | Target angle (0-180°) |
| `-o` | Output file |
| `-m` | Model checkpoint |
| `-d` | Device (cpu/cuda) |

---

### 5. Generate Submission

```bash
matlab -batch "run('prepare_submission.m')"
```

Creates self-contained `Submission/` folder ready for competition.

---

## 🏗️ Model Architecture

**DCCRNConformer** (~10M parameters)

| Component | Details |
|-----------|---------|
| Encoder | Complex Conv2d: 2→48→96→192→256 |
| Bottleneck | Dual-Path Conformer (3 blocks, 4 heads) |
| Decoder | Complex ConvTranspose2d with skip connections |
| Conditioning | Angle MLP injection at bottleneck |

**Audio:** 16kHz, STFT n_fft=512, hop=128, 3s fixed input

---

## � Evaluation Results

### Anechoic Condition (5,000 samples)

| Category | SI-SDR (dB) | STOI | PESQ |
|----------|-------------|------|------|
| **Best Overall** | 16.91 | 0.950 | 2.64 |
| Male + Noise | 16.91 | 0.950 | 2.64 |
| Male + Music | 13.46 | 0.956 | 2.54 |
| Male + Female | 12.96 | 0.959 | 2.64 |

**Inference:** 50.6ms avg (59x real-time for 3s audio)

### Reverberant Condition (5,000 samples)

| Category | SI-SDR (dB) | STOI | PESQ |
|----------|-------------|------|------|
| **Best Overall** | 12.49 | 0.942 | 2.48 |
| Male + Noise | 12.62 | 0.850 | 2.00 |
| Male + Music | 11.58 | 0.886 | 2.27 |
| Male + Female | 12.49 | 0.942 | 2.48 |

**Inference:** 50.5ms avg (59x real-time for 3s audio)

---

## �📊 Metrics

| Metric | Description |
|--------|-------------|
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio (dB) |
| **STOI** | Short-Time Objective Intelligibility (0-1) |
| **PESQ** | Perceptual Evaluation of Speech Quality (-0.5 to 4.5) |

---

## 📋 Quick Start

```bash
# Setup
cd SP_CUP_Phase_2
pip install -r requirements.txt

# Inference
cd "Model Inference"
python inference_Conformer.py -i audio.wav -a 90 -o out.wav -d cuda

# Evaluate
python test_Conformer.py

# Generate submission
cd ..
matlab -batch "run('prepare_submission.m')"
```

---

##  License

Developed for IEEE Signal Processing Cup 2026 competition.
