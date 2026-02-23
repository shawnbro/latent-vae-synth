# Latent Synth

A neural wavetable synthesizer. A Variational Autoencoder (VAE) is trained on thousands of single-cycle waveforms, compressing them into a smooth 2D latent space. At runtime, an XY pad navigates that space in real time — every position decodes to a unique waveform, and moving through it morphs the timbre continuously.

![Latent space colored by waveform category](plots/latent_ep200_categories.png)

*Each point is one of 45,000 AKWF waveforms, projected into the 2D latent space. The star-shaped structure reflects the diversity of the dataset — clusters correspond to different timbral families.*

---

## How it works

```
AKWF waveforms  →  VAE encoder  →  2D latent space  →  VAE decoder  →  waveform
                                          ↑
                                      XY pad
```

- The **encoder** compresses a 2048-sample waveform into a 2D point (z₀, z₁)
- The **decoder** is exported to ONNX and runs at runtime — no PyTorch needed
- The **oscillator** loops the decoded waveform at the target frequency (wavetable synthesis)
- KL divergence during training enforces smoothness, so nearby latent points sound similar

### Latent space navigation grid

![6×6 decode grid across the latent space](plots/latent_ep200_grid.png)

*Each cell is a waveform decoded from a point on a regular grid. You can hear the morphing as you sweep the XY pad across the space.*

---

## Synth UI

Classic Macintosh-inspired interface built with tkinter:

- **XY Pad** — drag to navigate the latent space; displays current z₀/z₁ coordinates
- **Oscillator** — live waveform display updates as you move
- **Envelope** — ADSR amplitude envelope with log-scale sliders
- **Filter** — resonant lowpass with envelope modulation amount
- **Motion** — LFO with four shapes, Z0/Z1 scan sliders, and a Randomize button
- **I/O** — master gain, MIDI input selector, audio output selector

### Motion panel

| Control | Behaviour |
|---------|-----------|
| **LFO: OFF/ON** | Toggles automatic latent cycling |
| **Circ** | Traces a circle around the center point |
| **X / Y** | Scans one axis sinusoidally, holds the other |
| **Walk** | Ornstein-Uhlenbeck random walk (mean-reverts to center) |
| **Rate** | 0.05 – 4 Hz |
| **Depth** | Radius/amplitude in latent units (0 – 4) |
| **Z0 / Z1** | Direct axis control; sets LFO center when LFO is on |
| **Randomize** | Single-click = nearby jump (r ≤ 1.5); double-click = anywhere |

---

## Quickstart (pre-trained model included)

```bash
git clone <repo-url>
cd latent-vst

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app/synth.py
```

The repo includes a pre-trained ONNX decoder (`export/decoder.onnx` + `export/decoder.onnx.data`), so you can run the synth immediately without training.

**MIDI:** Connect a MIDI controller and select it from the MIDI In dropdown. The synth responds to note-on/off velocity. Audio routes to whatever device you select in Audio Out — use a virtual cable (e.g. [BlackHole](https://existential.audio/blackhole/)) to send it into a DAW.

---

## Training from scratch

### 1. Download the dataset

```bash
git clone https://github.com/KristofferKarlAxelEkstrand/AKWF-FREE data/akwf
```

The AKWF-FREE dataset contains ~48,000 single-cycle WAV files across 40+ categories.

### 2. Train the VAE

```bash
python model/train.py --data data/akwf --epochs 200 --beta 0.001
```

Key flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 100 | Total epochs to train (ceiling, not additional) |
| `--beta` | 1.0 | KL weight — keep low (0.001–0.01) to prevent posterior collapse |
| `--warmup` | 20 | Epochs to ramp β from 0 to target |
| `--batch-size` | 256 | Reduce if you run out of memory |
| `--resume` | — | Path to checkpoint to continue from |

Training logs are saved to `model/checkpoints/train_log.csv`. Best checkpoint (by validation loss) is saved to `model/checkpoints/best.pt`.

### 3. Visualize the latent space

```bash
MPLBACKEND=Agg python notebooks/visualize_latent.py \
  --checkpoint model/checkpoints/best.pt --grid
```

Saves scatter plot, density heatmap, category highlights, and a decode grid to `plots/`.

### 4. Export to ONNX

```bash
python export/export.py --checkpoint model/checkpoints/best.pt
```

Validates the ONNX output against PyTorch and benchmarks inference latency. Target: < 1 ms per waveform.

### 5. Run the synth

```bash
python app/synth.py --model export/decoder.onnx
```

---

## Architecture

### VAE

```
Encoder
  Conv1d(1→64,   k=8, stride=4)  + BN + LeakyReLU   # 2048 → 512
  Conv1d(64→128, k=8, stride=4)  + BN + LeakyReLU   # 512  → 128
  Conv1d(128→256,k=8, stride=4)  + BN + LeakyReLU   # 128  → 32
  Conv1d(256→512,k=8, stride=4)  + BN + LeakyReLU   # 32   → 8
  Flatten → Linear → (μ, log σ²)                     # 8×512 → 2

Decoder  (runtime — exported to ONNX)
  Linear(2 → 4096)  + reshape to (512, 8)
  ConvTranspose1d(512→256, k=8, stride=4) + BN + LeakyReLU
  ConvTranspose1d(256→128, k=8, stride=4) + BN + LeakyReLU
  ConvTranspose1d(128→64,  k=8, stride=4) + BN + LeakyReLU
  ConvTranspose1d(64→1,    k=8, stride=4) + Tanh
  # output: (batch, 1, 2048) in [-1, 1]
```

~705K parameters total.

### Loss

```
L = MSE(recon, target) + β · KL(q(z|x) || N(0,I))
```

β = 0.001 worked well on the full AKWF dataset. Higher β produces smoother navigation at the cost of reconstruction fidelity.

### Audio engine

- Decoded waveform is looped as a single-cycle wavetable
- Phase increment: `WAVEFORM_LEN × freq / SAMPLE_RATE`
- Linear interpolation between samples for alias-free playback
- 512-sample blocks at 44100 Hz → ~11.6 ms latency

---

## Project structure

```
.
├── app/
│   └── synth.py              # Standalone synth app
├── data/
│   └── akwf/                 # AKWF dataset (not in repo — see above)
├── export/
│   ├── export.py             # ONNX export script
│   ├── decoder.onnx          # Exported model graph
│   └── decoder.onnx.data     # Model weights (1.3 MB)
├── model/
│   ├── vae.py                # VAE architecture
│   ├── dataset.py            # AKWF dataset loader + cache
│   ├── train.py              # Training loop
│   └── checkpoints/
│       ├── best.pt           # Best checkpoint by val loss
│       └── train_log.csv     # Loss history
├── notebooks/
│   └── visualize_latent.py   # Latent space plots
├── plots/                    # Generated visualizations
└── CLAUDE.md                 # Project spec & working notes
```

---

## Dependencies

- Python 3.13
- PyTorch 2.10 + torchaudio
- soundfile (WAV loading — torchaudio 2.10 dropped old backends)
- onnx / onnxruntime / onnxscript
- sounddevice (audio output)
- pygame (MIDI input via PortMidi — avoids Python 3.13 GIL issues with rtmidi)
- scipy (biquad filter)
- matplotlib (visualization only)

---

## Next Steps

Ideas for where to take this next, roughly in order of effort:

**Sound quality**
- Add a spectral loss term (FFT magnitude MSE) alongside the reconstruction loss — the current MSE-only loss treats all frequencies equally, which tends to blur high-frequency content
- Increase latent dim to 8–16 and learn a 2D projection for the XY pad (PCA or a small learned network), giving the model more expressive capacity while keeping the interface simple
- Train on a focused subset of AKWF categories (e.g. organ + strings only) for a more coherent, navigable space

**Sequencing**
- **Waypoint sequencer** — store 4–8 latent points, step through or interpolate between them on a clock or MIDI trigger; like a wavetable step sequencer for the XY pad
- Tempo-sync the LFO rate to MIDI clock

**Synth features**
- Polyphony (voice pool with per-voice phase/envelope state)
- MIDI CC mapping for envelope, filter, and gain parameters
- Preset save/load — serialize named latent positions to JSON

**Distribution**
- Package as a VST3/AU plugin via [JUCE](https://juce.com/) or [iPlug2](https://iplug2.github.io/) with the ONNX runtime embedded
- Or wrap the Python app in a standalone `.app` bundle with PyInstaller

## Resources

- [AKWF Dataset](https://www.adventurekid.se/akwf/) — Adventure Kid Waveforms
- [β-VAE paper](https://openreview.net/forum?id=Sy2fzU9gl)
- [RBJ Audio EQ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html) — filter coefficients
- [ONNX Runtime](https://onnxruntime.ai/docs/api/python/)
