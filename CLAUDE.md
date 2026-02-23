# Latent - A Neural Wavetable Synth

A neural synthesizer that uses a Variational Autoencoder (VAE) trained on single-cycle waveforms to create an infinite, navigable wavetable synthesis engine.

## Project Goals
- Train a VAE on single-cycle waveforms (AKWF dataset)
- Export trained model to ONNX for runtime inference
- Build a playable standalone synth app with XY pad latent space navigation
- Output as VST3/AU plugin or standalone desktop app

## Architecture Overview
- **Model**: VAE with 1D convolutional encoder/decoder, 2D latent space
- **Dataset**: Adventure Kid Waveforms (AKWF) — 4000+ single-cycle WAV files
- **Training**: PyTorch
- **Export**: ONNX
- **Audio runtime**: TBD (JUCE / iPlug2 / Python + sounddevice)
- **UI**: XY pad for latent space navigation, real-time waveform display

## Repo Structure
```
/
├── data/               # Raw and processed waveforms
│   └── akwf/           # AKWF dataset
├── model/              # VAE architecture and training code
│   ├── vae.py
│   ├── dataset.py
│   └── train.py
├── export/             # ONNX export scripts
│   └── export.py
├── app/                # Standalone synth application
│   ├── synth.py        # Audio engine (sounddevice / RTAudio)
│   └── ui/             # UI components
├── notebooks/          # Experimentation and visualization
└── CLAUDE.md
```

## Core Conventions
- Waveforms are normalized to [-1, 1] and resampled to a fixed length (e.g. 2048 samples)
- Latent space is 2D to enable direct XY pad mapping
- Model input/output shape: (batch, 1, 2048)
- Use float32 throughout for audio and model weights
- ONNX export must support real-time inference (no dynamic shapes)

## Development Phases

### Phase 1 — Data + Model
- [ ] Download and preprocess AKWF dataset
- [ ] Implement VAE (encoder, reparameterization, decoder)
- [ ] Train and validate — target reconstruction loss < 0.01 MSE
- [ ] Visualize latent space (t-SNE or direct 2D plot if latent_dim=2)

### Phase 2 — Export
- [ ] Export decoder only to ONNX (encoder only needed at training time)
- [ ] Validate ONNX output matches PyTorch output
- [ ] Benchmark inference latency (target < 1ms per waveform)

### Phase 3 — Audio App
- [ ] Basic audio engine: decode latent point → waveform → oscillator → output
- [ ] MIDI input for pitch (frequency scaling of output waveform)
- [ ] XY pad UI mapped to latent dimensions
- [ ] Real-time waveform visualizer

### Phase 4 — Polish
- [ ] Plugin format (VST3/AU via JUCE or iPlug2) or polished standalone
- [ ] Demo track / video
- [ ] README with architecture diagram and audio examples

## Key Technical Notes
- **Decoder only at runtime**: once trained, only the decoder is needed. A 2D latent point goes in, a waveform comes out.
- **Wavetable playback**: the decoded waveform is used as a single cycle that gets looped and pitch-shifted via playback rate, exactly like a wavetable synth.
- **Latent space continuity**: KL divergence loss enforces smooth interpolation — this is what makes XY navigation musically useful rather than chaotic.
- **AKWF preprocessing**: strip silence, find zero crossings to cleanly extract one cycle, resample to 2048 samples.

## Commands
```bash
# Install deps
pip install torch torchaudio onnx onnxruntime librosa sounddevice numpy

# Train
python model/train.py --data data/akwf --epochs 100 --latent-dim 2

# Export decoder to ONNX
python export/export.py --checkpoint model/checkpoints/best.pt --output export/decoder.onnx

# Run standalone app
python app/synth.py --model export/decoder.onnx
```

## Working Style
- Before writing any code, explain what you're about to do and why
- After writing code, explain the key architectural decisions
- Work one phase at a time — do not proceed to the next phase without confirmation
- If there are multiple valid approaches, briefly describe the tradeoffs before choosing one
- After completing a task, suggest one thing I could modify or break to deepen my understanding
- Prefer clarity over cleverness — optimize code for readability and learning, not conciseness

## Resources
- [AKWF Dataset](https://www.adventurekid.se/akwf/)
- [DDSP Paper](https://arxiv.org/abs/2001.04643) — useful conceptual background
- [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) — consider β > 1 for better latent disentanglement
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/)
