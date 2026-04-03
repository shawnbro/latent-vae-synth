# Latent Synth

A neural wavetable synthesizer. A Variational Autoencoder (VAE) is trained on thousands of single-cycle waveforms, compressing them into a smooth 2D latent space. At runtime, an XY pad navigates that space in real time — every position decodes to a unique waveform, and moving through it morphs the timbre continuously.

Available as a **native VST3/AU plugin** (JUCE + ONNX Runtime, no Python at runtime) and as a **Python standalone app**.

![Latent space colored by waveform category](plots/latent_ep200_categories.png)

*Each point is one of 45,000 AKWF waveforms projected into the 2D latent space. Clusters correspond to different timbral families.*

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

## Plugin (VST3 / AU)

### Quickstart — pre-built assets included

```bash
git clone <repo-url>
cd latent-vst/plugin

# Download ONNX Runtime 1.20.1 for macOS arm64 and unpack into vendor/
# https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1
# → onnxruntime-osx-arm64-1.20.1.tgz → extract to plugin/vendor/

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Install
cp -r LatentSynth_artefacts/Release/VST3/Latent\ Synth.vst3 \
      ~/Library/Audio/Plug-Ins/VST3/
cp -r LatentSynth_artefacts/Release/AU/Latent\ Synth.component \
      ~/Library/Audio/Plug-Ins/Components/
```

`plugin/Assets/decoder.onnx` and `plugin/Assets/latent_index.npz` are embedded at build time via JUCE's BinaryData mechanism — no external files needed at runtime.

### Features

| Section | Controls |
|---------|----------|
| **Latent Space** | XY pad + dedicated X / Y sliders |
| **MIDI CC** | CC1 (mod wheel) → X, CC11 (expression) → Y; all params DAW-automatable |
| **Envelope** | ADSR (1 ms – 8 s) |
| **Filter** | 4-pole resonant lowpass (24 dB/oct), envelope mod amount |
| **Motion** | LFO on/off, shape (Circ / X Scan / Y Scan / Walk / Wave), Rate, Depth, Glide, Vel Depth, Vel Angle |
| **Voices** | Unison count (1–8), detune spread (0–100 ¢) |
| **Reverb** | Freeverb (8 comb + 4 allpass), room size + wet |
| **Delay** | Feedback delay, time + feedback + wet |
| **I/O** | Master gain |

**LFO shapes**

| Shape | Behaviour |
|-------|-----------|
| Circ | Circular orbit around the center point |
| X Scan | Sinusoidal sweep on X axis |
| Y Scan | Sinusoidal sweep on Y axis |
| Walk | Ornstein-Uhlenbeck random walk (mean-reverts to center) |
| Wave | Uses the current decoded waveform as an LFO shape |

**Velocity → Latent** — at noteOn, the waveform is decoded from a point offset from the current XY position by `Vel Depth` latent units in the direction of `Vel Angle`. At velocity 0 the center waveform is used; at velocity 127 the full offset is applied, giving per-note timbre variation.

**Unison** — each note spawns N voices with detune spread symmetrically around 0¢. All voices share the same MIDI note so they release together.

**Glide** — cross-fades between the previous and new waveform over the specified time (0–2000 ms), so moving the XY pad or LFO smoothly morphs the timbre rather than switching abruptly.

### Architecture notes

- ONNX inference runs on the **message thread** (30 Hz timer), never the audio thread — eliminating dropout and priority-inversion issues
- Velocity→latent and LFO position updates decode a new waveform at each noteOn / timer tick
- Filter is a 4-pole transposed direct form II biquad (two cascaded RBJ lowpass stages); state is preserved across coefficient changes for click-free cutoff modulation
- `plugin/build/` and `vendor/*/lib/` (the 33 MB dylib) are gitignored; everything else is in-tree

---

## Python standalone app

### Quickstart

```bash
git clone <repo-url>
cd latent-vst

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app/synth.py
```

The repo includes a pre-trained ONNX decoder (`export/decoder.onnx` + `export/decoder.onnx.data`), so you can run the synth immediately without training.

**MIDI:** Connect a MIDI controller and select it from the MIDI In dropdown. Audio routes to the device selected in Audio Out — use a virtual cable (e.g. [BlackHole](https://existential.audio/blackhole/)) to send into a DAW.

### UI

Classic Macintosh-inspired interface built with tkinter:

- **XY Pad** — drag to navigate the latent space; shows z₀/z₁ coordinates and the name of the nearest AKWF waveform
- **Oscilloscope** — live audio display with zero-crossing trigger; shows decoded waveform shape when silent
- **Envelope** — ADSR amplitude envelope with log-scale sliders (1 ms – 8000 ms)
- **Filter** — resonant lowpass with envelope modulation amount
- **Motion** — LFO with five shapes, Vel Depth/Angle, Z0/Z1 scan sliders, glide, and a Randomize button
- **Arpeggiator** — step arpeggiator (1–4 steps) with per-step latent positions, BPM/gate control, and up/down/up-down/random order
- **I/O** — master gain, MIDI input selector (multi-port), audio output selector, keyboard piano toggle

### Motion panel

| Control | Behaviour |
|---------|-----------|
| **LFO: OFF/ON** | Toggles automatic latent cycling |
| **Circ** | Traces a circle around the center point |
| **X / Y** | Scans one axis sinusoidally, holds the other |
| **Walk** | Ornstein-Uhlenbeck random walk (mean-reverts to center) |
| **Rate** | 0.05 – 4 Hz |
| **Depth** | Radius/amplitude in latent units (0 – 4) |
| **Glide** | Portamento time (0 – 1000 ms) |
| **Z0 / Z1** | Direct axis control; sets LFO center when LFO is on |
| **Randomize** | Single-click = nearby jump (r ≤ 1.5); double-click = anywhere |

### Keyboard piano

Press **M** to toggle. When on:

| Key | Action |
|-----|--------|
| `A W S E D F T G Y H U J K O L` | Piano keys (white + black), one octave |
| `Z` / `X` | Octave down / up |
| `C` / `V` | Velocity −10 / +10 |

### MIDI CC Learn

Right-click any slider to bind a MIDI CC. Wiggle a hardware knob — the slider shows the bound CC number. Right-click again to clear.

Assignable: Latent X/Y, Gain, Cutoff, Resonance, Env Amount, ADSR, Glide, LFO Rate/Depth.

CC assignments persist to `~/.latent_synth_cc.json` and are restored on next launch.

---

## Training from scratch

### 1. Download the dataset

```bash
git clone https://github.com/KristofferKarlAxelEkstrand/AKWF-FREE data/akwf
```

~48,000 single-cycle WAV files across 40+ categories.

### 2. Train the VAE

```bash
python model/train.py --data data/akwf --epochs 200 --beta 0.001
```

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 100 | Total epochs (ceiling, not additional) |
| `--beta` | 1.0 | KL weight — keep low (0.001–0.01) to prevent posterior collapse |
| `--warmup` | 20 | Epochs to ramp β from 0 to target |
| `--batch-size` | 256 | Reduce if you run out of memory |
| `--resume` | — | Path to checkpoint to continue from |

### 3. Visualize the latent space

```bash
MPLBACKEND=Agg python notebooks/visualize_latent.py \
  --checkpoint model/checkpoints/best.pt --grid
```

### 4. Export to ONNX

```bash
python export/export.py --checkpoint model/checkpoints/best.pt
```

Validates ONNX output against PyTorch and benchmarks inference latency. Target: < 1 ms per waveform.

```bash
# Convert to inline format (required for embedding in the plugin via BinaryData)
python -c "
import onnx
m = onnx.load('export/decoder.onnx')
onnx.save_model(m, 'plugin/Assets/decoder.onnx', save_as_external_data=False)
"
```

### 5. Build the latent index

```bash
python export/build_latent_index.py
cp export/latent_index.npz plugin/Assets/latent_index.npz
```

Encodes all waveforms and saves their 2D coordinates for nearest-neighbour waveform name lookup. Re-run after retraining.

### 6. Rebuild the plugin

```bash
cd plugin/build && cmake --build . --config Release
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
- Up to 16 voices (plugin) / 8 voices (Python app) with square-root gain scaling

---

## Project structure

```
.
├── app/
│   └── synth.py              # Python standalone app (engine + UI)
├── data/
│   └── akwf/                 # AKWF dataset (not in repo)
├── export/
│   ├── export.py             # ONNX export script
│   ├── build_latent_index.py # Build waveform name lookup index
│   ├── decoder.onnx          # Exported model graph
│   ├── decoder.onnx.data     # Model weights (1.3 MB)
│   ├── decoder_inline.onnx   # Same, single-file (for plugin BinaryData)
│   └── latent_index.npz      # Latent coords for 45K waveforms
├── model/
│   ├── vae.py                # VAE architecture
│   ├── dataset.py            # AKWF dataset loader + cache
│   ├── train.py              # Training loop
│   └── checkpoints/
│       ├── best.pt           # Best checkpoint by val loss
│       └── train_log.csv     # Loss history
├── notebooks/
│   └── visualize_latent.py   # Latent space plots
├── packaging/
│   └── latent_synth.spec     # PyInstaller spec for standalone .app
├── plots/                    # Generated visualizations
├── plugin/
│   ├── CMakeLists.txt        # JUCE 8 + ONNX Runtime CMake build
│   ├── Assets/               # decoder.onnx + latent_index.npz (embedded)
│   ├── Source/
│   │   ├── PluginProcessor.h/cpp   # AudioProcessor, APVTS, MIDI, timer
│   │   ├── PluginEditor.h/cpp      # Mac Classic UI, XY pad, sections
│   │   └── dsp/
│   │       ├── LatentSynth.h       # Engine: ONNX session, voice pool
│   │       ├── Voice.h             # Wavetable osc + ADSR + filter
│   │       ├── BiquadFilter.h      # 4-pole resonant LP (TDFII)
│   │       ├── LatentLFO.h         # Background LFO thread (5 shapes)
│   │       ├── SchroederReverb.h   # Freeverb topology
│   │       ├── FeedbackDelay.h     # Stereo delay
│   │       └── ADSREnvelope.h      # State-machine ADSR
│   └── vendor/
│       └── onnxruntime-osx-arm64-1.20.1/  # Headers + cmake (dylib gitignored)
└── CLAUDE.md                 # Project spec & working notes
```

---

## Dependencies

### Plugin (C++)
- JUCE 8 (fetched automatically via CMake FetchContent)
- ONNX Runtime 1.20.1 (arm64 macOS — download separately, place in `plugin/vendor/`)
- Xcode CLT / CMake 3.22+

### Python app
- Python 3.13
- PyTorch 2.10 + torchaudio
- soundfile, onnx, onnxruntime, sounddevice, pygame, scipy, matplotlib

---

## Next Steps

**Sound quality**
- Add a spectral loss term (FFT magnitude MSE) alongside reconstruction loss — MSE-only tends to blur high-frequency content
- Increase latent dim to 8–16 and learn a 2D projection for the XY pad (PCA or a small learned network)
- Train on a focused subset of AKWF categories for a more coherent, navigable space

**Sequencing**
- **Waypoint sequencer** — store 4–8 latent points, step through or interpolate between them on a clock or MIDI trigger
- Tempo-sync LFO rate to MIDI clock

**Synth features**
- Preset save/load — serialize named latent positions to JSON
- MIDI program change to step through stored presets
- Arpeggiator port to the plugin

**Distribution**
- Windows / Linux builds (ONNX Runtime is cross-platform; JUCE handles the rest)
- Code-sign with a Developer ID for Gatekeeper-free distribution

---

## Resources

- [AKWF Dataset](https://www.adventurekid.se/akwf/) — Adventure Kid Waveforms
- [β-VAE paper](https://openreview.net/forum?id=Sy2fzU9gl)
- [RBJ Audio EQ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html)
- [ONNX Runtime](https://onnxruntime.ai/docs/api/python/)
- [JUCE](https://juce.com/)
