# packaging/latent_synth.spec
# PyInstaller spec for "Latent Synth.app" macOS bundle.
#
# Build:
#   pip install pyinstaller
#   pyinstaller packaging/latent_synth.spec
#
# The resulting .app is in dist/Latent Synth.app
# Users must also install BlackHole (free virtual audio) to route audio into a DAW:
#   https://existential.audio/blackhole/

import os

block_cipher = None

a = Analysis(
    [os.path.join('..', 'app', 'synth.py')],
    pathex=[os.path.join('..', 'app')],
    binaries=[],
    datas=[
        (os.path.join('..', 'export', 'decoder.onnx'),     'export'),
        (os.path.join('..', 'export', 'latent_index.npz'), 'export'),
    ],
    hiddenimports=[
        'scipy.signal',
        'scipy.spatial',
        'scipy.spatial._ckdtree',
        'scipy.spatial.ckdtree',
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi.onnxruntime_pybind11_state',
        'pygame',
        'pygame.midi',
        'sounddevice',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='latent_synth',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,         # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='latent_synth',
)

app = BUNDLE(
    coll,
    name='Latent Synth.app',
    icon=None,
    bundle_identifier='com.latentsynth.standalone',
    info_plist={
        'NSMicrophoneUsageDescription': 'MIDI input via pygame.midi',
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleName': 'Latent Synth',
    },
)
