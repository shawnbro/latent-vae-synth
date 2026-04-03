# Latent Synth — JUCE Plugin

VST3 + AU + Standalone build of the Latent Synth using JUCE and ONNX Runtime.

## Prerequisites

- macOS 13+ with Xcode 15+ (for AU validation)
- CMake 3.22+
- Git (for JUCE FetchContent)

## One-time setup: ONNX Runtime

Download the prebuilt macOS arm64 package from GitHub Releases:

```bash
cd plugin/vendor
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz \
     -o onnxruntime.tgz
tar xzf onnxruntime.tgz
# Result: plugin/vendor/onnxruntime-osx-arm64-1.20.1/
```

## Build

```bash
cd plugin
cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_DIR="${PWD}/vendor/onnxruntime-osx-arm64-1.20.1"

cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
```

Artefacts appear in `build/LatentSynth_artefacts/`:
```
VST3/Latent Synth.vst3
AU/Latent Synth.component
Standalone/Latent Synth.app
```

## Install plugins

```bash
# AU  (Logic Pro / GarageBand)
cp -r "build/LatentSynth_artefacts/AU/Latent Synth.component" \
      ~/Library/Audio/Plug-Ins/Components/

# VST3  (Ableton / Reaper / Bitwig)
cp -r "build/LatentSynth_artefacts/VST3/Latent Synth.vst3" \
      ~/Library/Audio/Plug-Ins/VST3/
```

After installing, rescan plugins in your DAW.

## Validate AU (macOS)

```bash
auval -v aumu LtSn Lsyn
```

## Architecture

```
plugin/
├── CMakeLists.txt               Build system (JUCE + ONNX Runtime)
├── Assets/
│   ├── decoder.onnx             VAE decoder (copied from export/)
│   └── latent_index.npz         KDTree index (copied from export/)
└── Source/
    ├── PluginProcessor.h/cpp    JUCE AudioProcessor + APVTS parameters
    ├── PluginEditor.h/cpp       XY pad UI + waveform display + knobs
    └── dsp/
        ├── ADSREnvelope.h       State-machine ADSR (ported from Python)
        ├── BiquadFilter.h       2-stage RBJ lowpass (ported from Python)
        ├── SchroederReverb.h    Freeverb topology (ported from Python)
        ├── FeedbackDelay.h      Circular-buffer delay (ported from Python)
        ├── Voice.h              Wavetable oscillator voice (ported from Python)
        └── LatentSynth.h/cpp    Core engine — ONNX + voice pool + FX
```

## DAW parameters exposed for automation

| Parameter | Range | Default |
|---|---|---|
| Latent X | -4 – +4 | 0 |
| Latent Y | -4 – +4 | 0 |
| Attack | 1–5000 ms | 10 |
| Decay | 1–5000 ms | 100 |
| Sustain | 0–1 | 0.7 |
| Release | 1–8000 ms | 300 |
| Cutoff | 30–18000 Hz | 18000 |
| Resonance | 0–0.95 | 0 |
| Filter Env | 0–1 | 0 |
| Reverb Size | 0–1 | 0.5 |
| Reverb Wet | 0–1 | 0 |
| Delay Time | 20–1000 ms | 250 |
| Delay Feedback | 0–0.9 | 0.4 |
| Delay Wet | 0–1 | 0 |
| Gain | 0–1 | 0.7 |
