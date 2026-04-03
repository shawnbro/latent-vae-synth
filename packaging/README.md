# PyInstaller Standalone Bundle

Packages `app/synth.py` into a self-contained `Latent Synth.app` for macOS.

## Build

```bash
# From the project root
pip install pyinstaller
pyinstaller packaging/latent_synth.spec
# → dist/Latent Synth.app
```

## Using with a DAW (via BlackHole)

The `.app` uses the system audio device. Route audio into a DAW with the free
BlackHole virtual audio driver:

1. Install BlackHole: https://existential.audio/blackhole/
2. In **Latent Synth.app** → I/O panel, set Output to "BlackHole 2ch"
3. In your DAW, create an **External Instrument** track pointed at BlackHole 2ch
4. Play MIDI from the DAW → audio flows back through BlackHole → DAW track

## Troubleshooting

### `libportaudio.dylib` not found
sounddevice depends on PortAudio. Install it and let PyInstaller bundle it:

```bash
brew install portaudio
# PyInstaller picks it up automatically via sounddevice's __init__
```

### App won't open ("damaged or can't be opened")
Code-sign or remove the quarantine flag:
```bash
xattr -cr "dist/Latent Synth.app"
```

### onnxruntime hidden import missing
If you see an import error about onnxruntime, add the missing submodule to
`hiddenimports` in `latent_synth.spec` and rebuild.
