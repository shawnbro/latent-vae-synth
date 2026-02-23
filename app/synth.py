"""
synth.py — Latent wavetable synthesizer

Usage:
    python app/synth.py --model export/decoder.onnx

Controls:
    - Drag the XY pad to navigate the latent space
    - Play MIDI notes to hear the current waveform at pitch
    - Use ADSR sliders to shape the amplitude envelope
    - Use Filter sliders to control brightness and resonance
    - Use Motion panel: LFO auto-cycling, Z0/Z1 scan sliders, Randomize
    - Gain slider in I/O panel controls master output level
"""

import sys
import os
import argparse
import threading
import tkinter as tk
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from scipy.signal import lfilter

try:
    import pygame
    import pygame.midi
    pygame.midi.init()
    MIDI_AVAILABLE = True
except Exception:
    MIDI_AVAILABLE = False
    print("Warning: pygame.midi unavailable. MIDI input disabled.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE  = 44100
BLOCK_SIZE   = 512
WAVEFORM_LEN = 2048
LATENT_MIN   = -4.0
LATENT_MAX   =  4.0
PAD_SIZE     = 360
WAVE_WIDTH   = 360
WAVE_HEIGHT  = 90

# Mac Classic palette
MAC_BG      = "#c0c0c0"
MAC_WHITE   = "#ffffff"
MAC_BLACK   = "#000000"
MAC_SHADOW  = "#808080"
MAC_HILIGHT = "#f0f0f0"
MAC_TITLE   = "#000000"
MAC_STRIPE  = "#ffffff"

FONT_SYS    = ("Monaco", 10)
FONT_LABEL  = ("Geneva", 10)
FONT_SMALL  = ("Monaco", 9)
FONT_TINY   = ("Monaco", 8)


# ---------------------------------------------------------------------------
# ADSR Envelope
# ---------------------------------------------------------------------------

class ADSREnvelope:
    """
    Per-block ADSR amplitude envelope.

    Processes blocks of N samples and returns a float32 array of envelope
    levels in [0, 1]. State transitions happen mid-block for accuracy.
    """

    IDLE    = 0
    ATTACK  = 1
    DECAY   = 2
    SUSTAIN = 3
    RELEASE = 4

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr      = sample_rate
        self.state   = self.IDLE
        self.level   = 0.0
        self._release_start = 0.0

        self.attack_rate  = self._ms_to_rate(10)
        self.decay_rate   = self._ms_to_rate(100)
        self.sustain      = 0.7
        self.release_rate = self._ms_to_rate(300)

    def _ms_to_rate(self, ms: float) -> float:
        samples = max(1, ms * self.sr / 1000.0)
        return 1.0 / samples

    def set_attack(self,  ms: float): self.attack_rate  = self._ms_to_rate(max(1, ms))
    def set_decay(self,   ms: float): self.decay_rate   = self._ms_to_rate(max(1, ms))
    def set_sustain(self, level: float): self.sustain    = float(np.clip(level, 0.0, 1.0))
    def set_release(self, ms: float): self.release_rate = self._ms_to_rate(max(1, ms))

    def note_on(self):
        self.state = self.ATTACK

    def note_off(self):
        if self.state != self.IDLE:
            self._release_start = self.level
            self.state = self.RELEASE

    def process(self, n_frames: int) -> np.ndarray:
        out   = np.empty(n_frames, dtype=np.float32)
        level = self.level
        state = self.state

        for i in range(n_frames):
            if state == self.ATTACK:
                level += self.attack_rate
                if level >= 1.0:
                    level = 1.0
                    state = self.DECAY

            elif state == self.DECAY:
                level -= self.decay_rate
                if level <= self.sustain:
                    level = self.sustain
                    state = self.SUSTAIN

            elif state == self.SUSTAIN:
                level = self.sustain

            elif state == self.RELEASE:
                level -= self._release_start * self.release_rate
                if level <= 0.0:
                    level = 0.0
                    state = self.IDLE

            out[i] = level

        self.level = level
        self.state = state
        return out


# ---------------------------------------------------------------------------
# Biquad lowpass filter  (RBJ Audio EQ Cookbook)
# ---------------------------------------------------------------------------

class BiquadFilter:
    """
    Four-pole resonant lowpass filter (two cascaded biquad stages, 24 dB/oct).

    Stage 1 uses a fixed Butterworth Q for a flat passband.
    Stage 2 adds resonance — increasing Q creates a peak near cutoff.

    Key design: filter state (zi) is NOT reset when coefficients change.
    This allows smooth real-time cutoff modulation without artifacts.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr        = float(sample_rate)
        self.cutoff_hz = 18000.0
        self.resonance = 0.0
        # State for each biquad stage — intentionally preserved across coeff updates
        self._zi1 = np.zeros(2)
        self._zi2 = np.zeros(2)
        self._update_coeffs()

    def set_cutoff(self, hz: float):
        self.cutoff_hz = float(np.clip(hz, 30.0, self.sr * 0.49))
        self._update_coeffs()

    def set_resonance(self, r: float):
        self.resonance = float(np.clip(r, 0.0, 0.95))
        self._update_coeffs()

    def _biquad_lp(self, Q: float):
        w0     = 2.0 * np.pi * self.cutoff_hz / self.sr
        cos_w0 = np.cos(w0)
        alpha  = np.sin(w0) / (2.0 * Q)
        b0 = (1.0 - cos_w0) / 2.0
        b1 =  1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 =  1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 =  1.0 - alpha
        return (np.array([b0/a0, b1/a0, b2/a0]),
                np.array([1.0,   a1/a0, a2/a0]))

    def _update_coeffs(self):
        # 4th-order Butterworth pole-pair Q values; resonance boosts stage 2
        Q1 = 0.5412139                              # Butterworth stage 1 (flat)
        Q2 = 1.3065630 + self.resonance * 10.0     # Butterworth stage 2 + resonance peak
        self._b1, self._a1 = self._biquad_lp(Q1)
        self._b2, self._a2 = self._biquad_lp(Q2)
        # zi1/zi2 deliberately NOT touched — state persists for smooth modulation

    def reset(self):
        """Clear filter memory. Call on fresh note starts to prevent clicks from stale state."""
        self._zi1 = np.zeros(2)
        self._zi2 = np.zeros(2)

    def process(self, x: np.ndarray) -> np.ndarray:
        x64 = x.astype(np.float64)
        y,  self._zi1 = lfilter(self._b1, self._a1, x64, zi=self._zi1)
        y,  self._zi2 = lfilter(self._b2, self._a2, y,   zi=self._zi2)
        return np.clip(y, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Wavetable oscillator / audio engine
# ---------------------------------------------------------------------------

class LatentSynth:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])

        self._waveform   = np.zeros(WAVEFORM_LEN, dtype=np.float32)
        self._phase      = 0.0
        self._frequency  = 0.0
        self._velocity   = 0.0
        self._gain       = 0.8   # master output gain (0-1)
        self._held_notes = {}    # midi_note → velocity (0-1), insertion order = press order

        self.envelope   = ADSREnvelope(SAMPLE_RATE)
        self.filt       = BiquadFilter(SAMPLE_RATE)
        self._base_cutoff  = 18000.0
        self._env_amount   = 0.0

        self.set_latent(0.0, 0.0)

    # ------------------------------------------------------------------
    # Latent navigation
    # ------------------------------------------------------------------

    def set_latent(self, x: float, y: float):
        z = np.array([[x, y]], dtype=np.float32)
        raw = self.session.run(["waveform"], {"latent": z})[0]
        self._waveform = raw[0, 0].astype(np.float32)
        return self._waveform

    # ------------------------------------------------------------------
    # MIDI
    # ------------------------------------------------------------------

    def note_on(self, midi_note: int, velocity: int):
        vel = velocity / 127.0
        self._held_notes[midi_note] = vel   # update or add (preserves insertion order)
        self._frequency = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        self._velocity  = vel
        if self.envelope.state == ADSREnvelope.IDLE:
            self.filt.reset()   # clear stale filter state so it doesn't bleed into the attack
        self.envelope.note_on()

    def note_off(self, midi_note: int):
        self._held_notes.pop(midi_note, None)
        if self._held_notes:
            # Last-note priority: fall back to the most recently pressed held note
            last_note = list(self._held_notes.keys())[-1]
            self._frequency = 440.0 * (2.0 ** ((last_note - 69) / 12.0))
            self._velocity  = self._held_notes[last_note]
            # No envelope retrigger — legato behaviour
        else:
            self.envelope.note_off()

    # ------------------------------------------------------------------
    # Param setters  (called from UI thread)
    # ------------------------------------------------------------------

    def set_attack(self,  ms):    self.envelope.set_attack(ms)
    def set_decay(self,   ms):    self.envelope.set_decay(ms)
    def set_sustain(self, level): self.envelope.set_sustain(level)
    def set_release(self, ms):    self.envelope.set_release(ms)

    def set_cutoff(self, hz):
        self._base_cutoff = hz
        if self._env_amount == 0.0:
            self.filt.set_cutoff(hz)

    def set_filter_resonance(self, r):
        self.filt.set_resonance(r)

    def set_env_amount(self, amount):
        self._env_amount = float(np.clip(amount, 0.0, 1.0))

    def set_gain(self, g: float):
        self._gain = float(np.clip(g, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Audio callback  (runs on sounddevice thread)
    # ------------------------------------------------------------------

    def audio_callback(self, outdata, frames, time_info, status):
        waveform  = self._waveform
        frequency = self._frequency
        velocity  = self._velocity

        if frequency <= 0.0:
            outdata[:, 0] = 0.0
            return

        # ── 1. Vectorised wavetable oscillator ────────────────────────
        phase_inc = WAVEFORM_LEN * frequency / SAMPLE_RATE
        phases    = (self._phase + np.arange(frames, dtype=np.float64) * phase_inc) % WAVEFORM_LEN
        idx       = phases.astype(np.int32)
        frac      = (phases - idx).astype(np.float32)
        next_idx  = (idx + 1) % WAVEFORM_LEN
        samples   = waveform[idx] * (1.0 - frac) + waveform[next_idx] * frac
        self._phase = (self._phase + frames * phase_inc) % WAVEFORM_LEN

        # ── 2. ADSR envelope ──────────────────────────────────────────
        env = self.envelope.process(frames)

        # ── 3. Filter envelope modulation ─────────────────────────────
        if self._env_amount > 0.0:
            env_level = float(env.mean())
            mod_cutoff = self._base_cutoff + self._env_amount * (18000.0 - self._base_cutoff) * env_level
            self.filt.set_cutoff(mod_cutoff)

        # ── 4. Biquad lowpass filter ───────────────────────────────────
        samples = self.filt.process(samples)

        # ── 5. Apply envelope × velocity × master gain ────────────────
        outdata[:, 0] = samples * env * velocity * self._gain

    # ------------------------------------------------------------------
    # Audio stream management
    # ------------------------------------------------------------------

    def start_audio(self, device=None):
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=device,
            callback=self.audio_callback,
        )
        self._stream.start()

    def stop_audio(self):
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()


# ---------------------------------------------------------------------------
# MIDI handler
# ---------------------------------------------------------------------------

class MidiInput:
    """
    MIDI input using pygame.midi (PortMidi backend).
    Avoids the Python 3.13 GIL crash that affects python-rtmidi.
    """

    def __init__(self, synth: LatentSynth):
        self.synth   = synth
        self._input  = None
        self._active = False
        self._thread = None

    def list_ports(self):
        if not MIDI_AVAILABLE:
            return []
        ports = []
        for i in range(pygame.midi.get_count()):
            info = pygame.midi.get_device_info(i)
            if info[2]:  # is_input
                ports.append((i, info[1].decode("utf-8", errors="replace")))
        return ports

    def open_port(self, device_id: int):
        if not MIDI_AVAILABLE:
            return
        self.close()
        try:
            self._input  = pygame.midi.Input(device_id)
            self._active = True
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"MIDI open error: {e}")

    def _poll(self):
        while self._active and self._input:
            try:
                if self._input.poll():
                    for event in self._input.read(64):
                        data, _ = event
                        status  = data[0] & 0xF0
                        note    = data[1]
                        vel     = data[2]
                        if status == 0x90 and vel > 0:
                            self.synth.note_on(note, vel)
                        elif status == 0x80 or (status == 0x90 and vel == 0):
                            self.synth.note_off(note)
            except Exception:
                break
            threading.Event().wait(0.001)

    def close(self):
        self._active = False
        if self._input:
            self._input.close()
            self._input = None


# ---------------------------------------------------------------------------
# LFO — automatic latent space navigation
# ---------------------------------------------------------------------------

class LatentLFO:
    """
    Background thread that automatically moves the latent cursor.

    Shapes:
      CIRCLE  — traces a circle around the center point
      X_SCAN  — oscillates horizontally, center Y fixed
      Y_SCAN  — oscillates vertically, center X fixed
      WALK    — Ornstein-Uhlenbeck mean-reverting random walk

    update_fn(x, y) is called at ~50 Hz. Wrap in root.after for thread safety.
    """

    CIRCLE = "circle"
    X_SCAN = "x_scan"
    Y_SCAN = "y_scan"
    WALK   = "walk"

    def __init__(self, update_fn):
        self._update_fn = update_fn
        self.rate     = 0.2     # Hz
        self.depth    = 1.0     # latent units (radius / amplitude)
        self.shape    = self.CIRCLE
        self.center_x = 0.0
        self.center_y = 0.0
        self.active   = False
        self._phase   = 0.0
        self._wx      = 0.0    # walk state
        self._wy      = 0.0
        self._thread  = None

    def start(self):
        if self.active:
            return
        self._phase  = 0.0
        self._wx     = self.center_x
        self._wy     = self.center_y
        self.active  = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def _run(self):
        DT  = 0.02   # 50 Hz
        rng = np.random.default_rng()
        while self.active:
            self._phase += 2.0 * np.pi * self.rate * DT
            cx, cy = self.center_x, self.center_y
            d      = self.depth

            if self.shape == self.CIRCLE:
                x = cx + d * np.cos(self._phase)
                y = cy + d * np.sin(self._phase)
            elif self.shape == self.X_SCAN:
                x = cx + d * np.sin(self._phase)
                y = cy
            elif self.shape == self.Y_SCAN:
                x = cx
                y = cy + d * np.sin(self._phase)
            else:   # WALK — Ornstein-Uhlenbeck
                theta = 0.08
                sigma = d * 0.25
                self._wx += theta * (cx - self._wx) + sigma * rng.standard_normal()
                self._wy += theta * (cy - self._wy) + sigma * rng.standard_normal()
                x, y = self._wx, self._wy

            x = float(np.clip(x, LATENT_MIN, LATENT_MAX))
            y = float(np.clip(y, LATENT_MIN, LATENT_MAX))
            self._update_fn(x, y)
            threading.Event().wait(DT)


# ---------------------------------------------------------------------------
# Mac Classic UI helpers
# ---------------------------------------------------------------------------

def mac_frame(parent, **kwargs):
    return tk.Frame(parent, bg=MAC_BG, relief="groove", bd=2, **kwargs)

def mac_label(parent, text, **kwargs):
    kwargs.setdefault("font", FONT_LABEL)
    return tk.Label(parent, text=text, fg=MAC_BLACK, bg=MAC_BG, **kwargs)

def mac_button(parent, text, command, width=10):
    return tk.Button(
        parent, text=text, command=command,
        font=FONT_LABEL, fg=MAC_BLACK, bg=MAC_BG,
        relief="raised", bd=2, width=width,
        activebackground=MAC_HILIGHT, activeforeground=MAC_BLACK,
    )

def mac_slider(parent, from_, to, command, length=80,
               orient=tk.VERTICAL, resolution=0):
    return tk.Scale(
        parent,
        from_=from_, to=to,
        orient=orient,
        length=length,
        width=14,
        sliderlength=14,
        resolution=resolution,
        bg=MAC_BG,
        fg=MAC_BLACK,
        troughcolor=MAC_WHITE,
        highlightthickness=1,
        highlightbackground=MAC_BLACK,
        bd=1,
        relief="sunken",
        showvalue=False,
        command=command,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class SynthUI:
    def __init__(self, synth: LatentSynth, midi: MidiInput):
        self.synth = synth
        self.midi  = midi

        self.root = tk.Tk()
        self.root.title("Latent Synth")
        self.root.resizable(False, False)
        self.root.configure(bg=MAC_BG)

        self._latent_x       = 0.0
        self._latent_y       = 0.0
        self._dragging       = False
        self._in_scan_update = False   # prevents slider→callback→slider loops
        self._randomize_job  = None    # pending after_cancel handle
        self._lfo_tick       = 0       # rate-limit waveform display during LFO

        # LFO: callback fires on main thread via root.after
        self.lfo = LatentLFO(
            lambda x, y: self.root.after(0, self._lfo_update, x, y)
        )

        self._build_ui()
        self._update_waveform_display(synth._waveform)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self.root
        PAD  = 8

        # ── Window chrome ──────────────────────────────────────────────
        chrome = tk.Frame(root, bg=MAC_BLACK, bd=2, relief="solid")
        chrome.pack(padx=6, pady=6)

        # Title bar
        title_bar = tk.Frame(chrome, bg=MAC_BLACK, height=22)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        tk.Frame(title_bar, bg=MAC_WHITE, width=13, height=13,
                 bd=1, relief="solid").pack(side="left", padx=(5, 0), pady=4)

        stripe = tk.Frame(title_bar, bg=MAC_BLACK)
        stripe.pack(side="left", fill="both", expand=True, padx=6, pady=5)
        for _ in range(3):
            tk.Frame(stripe, bg=MAC_WHITE, height=1).pack(fill="x", pady=1)

        tk.Label(title_bar, text="Latent Synth",
                 font=("Monaco", 10, "bold"),
                 fg=MAC_WHITE, bg=MAC_BLACK).pack(side="left")

        stripe2 = tk.Frame(title_bar, bg=MAC_BLACK)
        stripe2.pack(side="left", fill="both", expand=True, padx=6, pady=5)
        for _ in range(3):
            tk.Frame(stripe2, bg=MAC_WHITE, height=1).pack(fill="x", pady=1)

        # Content area
        content = tk.Frame(chrome, bg=MAC_WHITE)
        content.pack(fill="both", expand=True, padx=1, pady=1)

        # ── Two-column layout ─────────────────────────────────────────
        columns = tk.Frame(content, bg=MAC_WHITE)
        columns.pack(fill="both", expand=True, padx=PAD, pady=PAD)

        left_col  = tk.Frame(columns, bg=MAC_WHITE)
        left_col.pack(side="left", anchor="n")

        right_col = tk.Frame(columns, bg=MAC_WHITE, padx=PAD)
        right_col.pack(side="left", anchor="n")

        # ── XY Pad ────────────────────────────────────────────────────
        tk.Label(left_col, text="◆ Latent Space",
                 font=("Monaco", 9, "bold"),
                 fg=MAC_BLACK, bg=MAC_WHITE).pack(anchor="w")

        xy_outer = tk.Frame(left_col, bg=MAC_BLACK, bd=1, relief="solid")
        xy_outer.pack(pady=(2, 0))

        self.canvas = tk.Canvas(
            xy_outer, width=PAD_SIZE, height=PAD_SIZE,
            bg=MAC_WHITE, highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack()

        step = PAD_SIZE // 4
        for i in range(1, 4):
            for y in range(0, PAD_SIZE, 4):
                self.canvas.create_line(i*step, y, i*step, y+2,
                                        fill=MAC_SHADOW, width=1)
            for x in range(0, PAD_SIZE, 4):
                self.canvas.create_line(x, i*step, x+2, i*step,
                                        fill=MAC_SHADOW, width=1)

        cx, cy = self._latent_to_canvas(0.0, 0.0)
        r = 5
        self._dot   = self.canvas.create_rectangle(cx-r, cy-r, cx+r, cy+r,
                                                    fill=MAC_BLACK, outline=MAC_BLACK)
        self._hline = self.canvas.create_line(cx-14, cy, cx+14, cy,
                                               fill=MAC_BLACK, width=1)
        self._vline = self.canvas.create_line(cx, cy-14, cx, cy+14,
                                               fill=MAC_BLACK, width=1)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self._coord_var = tk.StringVar(value="z0: +0.000    z1: +0.000")
        tk.Label(left_col, textvariable=self._coord_var,
                 font=FONT_TINY, fg=MAC_BLACK, bg=MAC_WHITE,
                 relief="sunken", bd=1, padx=4, anchor="w").pack(
                     fill="x", pady=(3, 0))

        # ── Waveform display ──────────────────────────────────────────
        tk.Label(left_col, text="◆ Oscillator",
                 font=("Monaco", 9, "bold"),
                 fg=MAC_BLACK, bg=MAC_WHITE).pack(anchor="w", pady=(8, 2))

        wave_outer = tk.Frame(left_col, bg=MAC_BLACK, bd=1, relief="solid")
        wave_outer.pack()

        self.wave_canvas = tk.Canvas(
            wave_outer, width=WAVE_WIDTH, height=WAVE_HEIGHT,
            bg=MAC_WHITE, highlightthickness=0,
        )
        self.wave_canvas.pack()

        # ── Right column panels ───────────────────────────────────────
        self._build_envelope_panel(right_col)
        self._build_filter_panel(right_col)
        self._build_motion_panel(right_col)
        self._build_io_panel(right_col)

    # ------------------------------------------------------------------
    # Envelope panel
    # ------------------------------------------------------------------

    def _build_envelope_panel(self, parent):
        panel = mac_frame(parent)
        panel.pack(fill="x", pady=(0, 6))

        tk.Label(panel, text="◆ Envelope",
                 font=("Monaco", 9, "bold"),
                 fg=MAC_BLACK, bg=MAC_BG).pack(anchor="w", padx=6, pady=(4, 2))

        sliders_row = tk.Frame(panel, bg=MAC_BG)
        sliders_row.pack(padx=10, pady=(0, 6))

        self._env_readout = tk.StringVar(value="A:10ms  D:100ms  S:0.70  R:300ms")
        tk.Label(panel, textvariable=self._env_readout,
                 font=FONT_TINY, fg=MAC_BLACK, bg=MAC_BG).pack(pady=(0, 4))

        def make_adsr_slider(col, label, from_, to, init, cmd):
            f = tk.Frame(sliders_row, bg=MAC_BG)
            f.grid(row=0, column=col, padx=6)
            s = mac_slider(f, from_=from_, to=to, command=cmd, length=100)
            s.set(init)
            s.pack()
            tk.Label(f, text=label, font=("Monaco", 9, "bold"),
                     fg=MAC_BLACK, bg=MAC_BG).pack()
            return s

        self._atk = make_adsr_slider(0, "A", 100, 0, 30,
                                      lambda v: self._on_adsr("A", v))
        self._dec = make_adsr_slider(1, "D", 100, 0, 55,
                                      lambda v: self._on_adsr("D", v))
        self._sus = make_adsr_slider(2, "S", 100, 0, 70,
                                      lambda v: self._on_adsr("S", v))
        self._rel = make_adsr_slider(3, "R", 100, 0, 60,
                                      lambda v: self._on_adsr("R", v))

        self._apply_adsr()

    def _log_time(self, slider_val):
        return 1.0 * (2000.0 ** (float(slider_val) / 100.0))

    def _on_adsr(self, param, val):
        self._apply_adsr()

    def _apply_adsr(self):
        atk_ms = self._log_time(self._atk.get())
        dec_ms = self._log_time(self._dec.get())
        sus    = self._sus.get() / 100.0
        rel_ms = self._log_time(self._rel.get())

        self.synth.set_attack(atk_ms)
        self.synth.set_decay(dec_ms)
        self.synth.set_sustain(sus)
        self.synth.set_release(rel_ms)

        self._env_readout.set(
            f"A:{atk_ms:.0f}ms  D:{dec_ms:.0f}ms  "
            f"S:{sus:.2f}  R:{rel_ms:.0f}ms"
        )

    # ------------------------------------------------------------------
    # Filter panel
    # ------------------------------------------------------------------

    def _build_filter_panel(self, parent):
        panel = mac_frame(parent)
        panel.pack(fill="x", pady=(0, 6))

        tk.Label(panel, text="◆ Filter",
                 font=("Monaco", 9, "bold"),
                 fg=MAC_BLACK, bg=MAC_BG).pack(anchor="w", padx=6, pady=(4, 2))

        inner = tk.Frame(panel, bg=MAC_BG)
        inner.pack(padx=8, pady=(0, 6), fill="x")

        def make_row(label, from_, to, init, cmd):
            row = tk.Frame(inner, bg=MAC_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=FONT_TINY, fg=MAC_BLACK,
                     bg=MAC_BG, width=10, anchor="w").pack(side="left")
            s = mac_slider(row, from_=from_, to=to, command=cmd,
                           length=130, orient=tk.HORIZONTAL)
            s.set(init)
            s.pack(side="left")
            val_lbl = tk.Label(row, text="", font=FONT_TINY,
                                fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w")
            val_lbl.pack(side="left", padx=(4, 0))
            return s, val_lbl

        self._cutoff_sl, self._cutoff_lbl = make_row(
            "Cutoff:", 0, 100, 85, self._on_cutoff)
        self._res_sl, self._res_lbl = make_row(
            "Resonance:", 0, 100, 0, self._on_resonance)
        self._env_amt_sl, self._env_amt_lbl = make_row(
            "Env Amount:", 0, 100, 0, self._on_env_amount)

        self._on_cutoff(self._cutoff_sl.get())
        self._on_resonance(self._res_sl.get())
        self._on_env_amount(self._env_amt_sl.get())

    def _slider_to_cutoff(self, val):
        return 30.0 * (600.0 ** (float(val) / 100.0))

    def _on_cutoff(self, val):
        hz = self._slider_to_cutoff(float(val))
        self.synth.set_cutoff(hz)
        self._cutoff_lbl.config(text=f"{hz:.0f} Hz")

    def _on_resonance(self, val):
        r = float(val) / 100.0 * 0.97
        self.synth.set_filter_resonance(r)
        self._res_lbl.config(text=f"{r:.2f}")

    def _on_env_amount(self, val):
        amt = float(val) / 100.0
        self.synth.set_env_amount(amt)
        self._env_amt_lbl.config(text=f"{int(float(val))}%")

    # ------------------------------------------------------------------
    # Motion panel  (LFO + scan sliders + randomize)
    # ------------------------------------------------------------------

    def _build_motion_panel(self, parent):
        panel = mac_frame(parent)
        panel.pack(fill="x", pady=(0, 6))

        tk.Label(panel, text="◆ Motion",
                 font=("Monaco", 9, "bold"),
                 fg=MAC_BLACK, bg=MAC_BG).pack(anchor="w", padx=6, pady=(4, 2))

        inner = tk.Frame(panel, bg=MAC_BG)
        inner.pack(padx=8, pady=(0, 6), fill="x")

        # LFO toggle + shape radio buttons
        top = tk.Frame(inner, bg=MAC_BG)
        top.pack(fill="x", pady=(0, 3))

        self._lfo_btn = tk.Button(
            top, text="LFO: OFF",
            font=FONT_TINY, fg=MAC_BLACK, bg=MAC_BG,
            relief="raised", bd=2, width=7,
            activebackground=MAC_HILIGHT,
            command=self._toggle_lfo,
        )
        self._lfo_btn.pack(side="left", padx=(0, 4))

        self._lfo_shape_var = tk.StringVar(value=LatentLFO.CIRCLE)
        for label, val in [
            ("Circ", LatentLFO.CIRCLE),
            ("X",    LatentLFO.X_SCAN),
            ("Y",    LatentLFO.Y_SCAN),
            ("Walk", LatentLFO.WALK),
        ]:
            tk.Radiobutton(
                top, text=label, variable=self._lfo_shape_var, value=val,
                font=FONT_TINY, fg=MAC_BLACK, bg=MAC_BG,
                activebackground=MAC_BG, selectcolor=MAC_BG,
                command=self._on_lfo_shape,
            ).pack(side="left", padx=1)

        # Rate / Depth / Z0 / Z1 sliders
        def make_row(label, from_, to, init, cmd, res=0):
            row = tk.Frame(inner, bg=MAC_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=FONT_TINY, fg=MAC_BLACK,
                     bg=MAC_BG, width=7, anchor="w").pack(side="left")
            s = mac_slider(row, from_=from_, to=to, command=cmd,
                           length=120, orient=tk.HORIZONTAL, resolution=res)
            s.set(init)
            s.pack(side="left")
            lbl = tk.Label(row, text="", font=FONT_TINY,
                            fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w")
            lbl.pack(side="left", padx=(3, 0))
            return s, lbl

        self._lfo_rate_sl,  self._lfo_rate_lbl  = make_row(
            "Rate:",  1, 100, 20, self._on_lfo_rate)
        self._lfo_depth_sl, self._lfo_depth_lbl = make_row(
            "Depth:", 0, 100, 50, self._on_lfo_depth)
        self._z0_sl, self._z0_lbl = make_row(
            "Z0:", LATENT_MIN, LATENT_MAX, 0.0, self._on_z0_scan, res=0.01)
        self._z1_sl, self._z1_lbl = make_row(
            "Z1:", LATENT_MIN, LATENT_MAX, 0.0, self._on_z1_scan, res=0.01)

        # Randomize button
        rand_row = tk.Frame(inner, bg=MAC_BG)
        rand_row.pack(fill="x", pady=(4, 0))

        self._rand_btn = tk.Button(
            rand_row, text="Randomize",
            font=FONT_LABEL, fg=MAC_BLACK, bg=MAC_BG,
            relief="raised", bd=2,
            activebackground=MAC_HILIGHT, activeforeground=MAC_BLACK,
        )
        self._rand_btn.pack(side="left")
        self._rand_btn.bind("<Button-1>",        self._on_randomize_click)
        self._rand_btn.bind("<Double-Button-1>", self._on_randomize_dblclick)

        tk.Label(rand_row, text=" dbl=far", font=FONT_TINY,
                 fg=MAC_SHADOW, bg=MAC_BG).pack(side="left")

        # Apply defaults
        self._on_lfo_rate(self._lfo_rate_sl.get())
        self._on_lfo_depth(self._lfo_depth_sl.get())

    # ------------------------------------------------------------------
    # Motion callbacks
    # ------------------------------------------------------------------

    def _toggle_lfo(self):
        if self.lfo.active:
            self.lfo.stop()
            self._lfo_btn.config(text="LFO: OFF")
        else:
            self.lfo.center_x = self._latent_x
            self.lfo.center_y = self._latent_y
            self.lfo.start()
            self._lfo_btn.config(text="LFO: ON ")

    def _on_lfo_shape(self):
        self.lfo.shape = self._lfo_shape_var.get()

    def _on_lfo_rate(self, val):
        # 1-100 slider → 0.05-4 Hz, log scale
        hz = 0.05 * (80.0 ** (float(val) / 100.0))
        self.lfo.rate = hz
        self._lfo_rate_lbl.config(text=f"{hz:.2f} Hz")

    def _on_lfo_depth(self, val):
        d = float(val) / 100.0 * 4.0
        self.lfo.depth = d
        self._lfo_depth_lbl.config(text=f"{d:.2f}")

    def _on_z0_scan(self, val):
        if self._in_scan_update:
            return
        x = float(val)
        self._z0_lbl.config(text=f"{x:+.3f}")
        if self.lfo.active:
            self.lfo.center_x = x
        else:
            self._move_to_latent(x, self._latent_y, update_sliders=False)

    def _on_z1_scan(self, val):
        if self._in_scan_update:
            return
        y = float(val)
        self._z1_lbl.config(text=f"{y:+.3f}")
        if self.lfo.active:
            self.lfo.center_y = y
        else:
            self._move_to_latent(self._latent_x, y, update_sliders=False)

    def _update_scan_sliders(self, x, y):
        """Sync Z0/Z1 sliders to current latent position without re-triggering callbacks."""
        self._in_scan_update = True
        self._z0_sl.set(x)
        self._z1_sl.set(y)
        self._z0_lbl.config(text=f"{x:+.3f}")
        self._z1_lbl.config(text=f"{y:+.3f}")
        self._in_scan_update = False

    def _on_randomize_click(self, event):
        """Single click: randomize to a nearby point (r ≤ 1.5)."""
        if self._randomize_job is not None:
            self.root.after_cancel(self._randomize_job)
        self._randomize_job = self.root.after(250, self._randomize_nearby)
        return "break"

    def _on_randomize_dblclick(self, event):
        """Double click: randomize anywhere in the full latent space."""
        if self._randomize_job is not None:
            self.root.after_cancel(self._randomize_job)
            self._randomize_job = None
        self._randomize_full()
        return "break"

    def _randomize_nearby(self):
        self._randomize_job = None
        r     = np.random.uniform(0.0, 1.5)
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        x = float(np.clip(self._latent_x + r * np.cos(theta), LATENT_MIN, LATENT_MAX))
        y = float(np.clip(self._latent_y + r * np.sin(theta), LATENT_MIN, LATENT_MAX))
        self._jump_to_latent(x, y)

    def _randomize_full(self):
        x = float(np.random.uniform(LATENT_MIN, LATENT_MAX))
        y = float(np.random.uniform(LATENT_MIN, LATENT_MAX))
        self._jump_to_latent(x, y)

    def _jump_to_latent(self, x, y):
        """Navigate to (x, y). If LFO is on, update the LFO center instead."""
        if self.lfo.active:
            self.lfo.center_x = x
            self.lfo.center_y = y
            self._update_scan_sliders(x, y)
        else:
            self._move_to_latent(x, y)

    # ------------------------------------------------------------------
    # I/O panel
    # ------------------------------------------------------------------

    def _build_io_panel(self, parent):
        panel = mac_frame(parent)
        panel.pack(fill="x")

        tk.Label(panel, text="◆ I/O",
                 font=("Monaco", 9, "bold"),
                 fg=MAC_BLACK, bg=MAC_BG).pack(anchor="w", padx=6, pady=(4, 2))

        inner = tk.Frame(panel, bg=MAC_BG)
        inner.pack(padx=6, pady=(0, 6), fill="x")

        # Gain
        gain_row = tk.Frame(inner, bg=MAC_BG)
        gain_row.pack(fill="x", pady=2)
        tk.Label(gain_row, text="Gain:", font=FONT_TINY,
                 fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w").pack(side="left")
        self._gain_sl = mac_slider(gain_row, from_=0, to=100,
                                    command=self._on_gain,
                                    length=130, orient=tk.HORIZONTAL)
        self._gain_sl.set(80)
        self._gain_sl.pack(side="left")
        self._gain_lbl = tk.Label(gain_row, text="80%", font=FONT_TINY,
                                   fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w")
        self._gain_lbl.pack(side="left", padx=(4, 0))

        # MIDI
        midi_row = tk.Frame(inner, bg=MAC_BG)
        midi_row.pack(fill="x", pady=2)
        tk.Label(midi_row, text="MIDI In:", font=FONT_TINY,
                 fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w").pack(side="left")

        self._midi_var = tk.StringVar(value="(no ports)")
        self._midi_menu = tk.OptionMenu(midi_row, self._midi_var, "(no ports)")
        self._midi_menu.config(font=FONT_TINY, bg=MAC_BG, fg=MAC_BLACK,
                                relief="raised", bd=2, width=18,
                                activebackground=MAC_BG)
        self._midi_menu["menu"].config(font=FONT_TINY, bg=MAC_WHITE)
        self._midi_menu.pack(side="left")
        mac_button(midi_row, "↺", self._refresh_midi_ports, width=2).pack(
            side="left", padx=(3, 0))

        # Audio
        audio_row = tk.Frame(inner, bg=MAC_BG)
        audio_row.pack(fill="x", pady=2)
        tk.Label(audio_row, text="Audio Out:", font=FONT_TINY,
                 fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w").pack(side="left")

        devices = sd.query_devices()
        output_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices)
                          if d["max_output_channels"] > 0]
        self._audio_var = tk.StringVar()
        default_out = sd.default.device[1]
        if output_devices:
            idx = default_out if isinstance(default_out, int) else 0
            self._audio_var.set(output_devices[min(idx, len(output_devices)-1)])

        audio_menu = tk.OptionMenu(audio_row, self._audio_var,
                                    *output_devices,
                                    command=self._on_audio_select)
        audio_menu.config(font=FONT_TINY, bg=MAC_BG, fg=MAC_BLACK,
                          relief="raised", bd=2, width=18,
                          activebackground=MAC_BG)
        audio_menu["menu"].config(font=FONT_TINY, bg=MAC_WHITE)
        audio_menu.pack(side="left")

        # Apply defaults
        self._on_gain(80)
        self._refresh_midi_ports()

    def _on_gain(self, val):
        g = float(val) / 100.0
        self.synth.set_gain(g)
        self._gain_lbl.config(text=f"{int(float(val))}%")

    # ------------------------------------------------------------------
    # XY pad interaction
    # ------------------------------------------------------------------

    def _canvas_to_latent(self, cx, cy):
        x = LATENT_MIN + (cx / PAD_SIZE) * (LATENT_MAX - LATENT_MIN)
        y = LATENT_MAX - (cy / PAD_SIZE) * (LATENT_MAX - LATENT_MIN)
        return (max(LATENT_MIN, min(LATENT_MAX, x)),
                max(LATENT_MIN, min(LATENT_MAX, y)))

    def _latent_to_canvas(self, x, y):
        cx = (x - LATENT_MIN) / (LATENT_MAX - LATENT_MIN) * PAD_SIZE
        cy = (LATENT_MAX - y) / (LATENT_MAX - LATENT_MIN) * PAD_SIZE
        return cx, cy

    def _on_press(self, event):
        self._dragging = True
        self._update_latent(event.x, event.y)

    def _on_drag(self, event):
        if self._dragging:
            self._update_latent(event.x, event.y)

    def _on_release(self, event):
        self._dragging = False

    def _update_latent(self, cx, cy):
        """Handle XY pad drag. When LFO is on, sets the LFO center."""
        x, y = self._canvas_to_latent(cx, cy)
        if self.lfo.active:
            self.lfo.center_x = x
            self.lfo.center_y = y
            self._update_scan_sliders(x, y)
            self._coord_var.set(f"z0: {x:+.3f}    z1: {y:+.3f}")
        else:
            self._move_to_latent(x, y)

    def _move_to_latent(self, x, y, update_sliders=True):
        """Move the XY cursor and trigger a decode. Must run on main thread."""
        self._latent_x = x
        self._latent_y = y

        px, py = self._latent_to_canvas(x, y)
        r = 5
        self.canvas.coords(self._dot,   px-r, py-r, px+r, py+r)
        self.canvas.coords(self._hline, px-14, py, px+14, py)
        self.canvas.coords(self._vline, px, py-14, px, py+14)
        self._coord_var.set(f"z0: {x:+.3f}    z1: {y:+.3f}")

        if update_sliders:
            self._update_scan_sliders(x, y)

        threading.Thread(target=self._decode_and_update,
                         args=(x, y), daemon=True).start()

    def _lfo_update(self, x, y):
        """Called on main thread by LFO at ~50 Hz via root.after."""
        self._latent_x = x
        self._latent_y = y

        px, py = self._latent_to_canvas(x, y)
        r = 5
        self.canvas.coords(self._dot,   px-r, py-r, px+r, py+r)
        self.canvas.coords(self._hline, px-14, py, px+14, py)
        self.canvas.coords(self._vline, px, py-14, px, py+14)
        self._coord_var.set(f"z0: {x:+.3f}    z1: {y:+.3f}")

        # Update scan sliders at 1/3 rate to reduce jank
        self._lfo_tick += 1
        if self._lfo_tick % 3 == 0:
            self._update_scan_sliders(x, y)

        # Decode inline — 0.26 ms, safe on main thread
        waveform = self.synth.set_latent(x, y)

        # Update waveform display at 1/5 rate (10 Hz is plenty for visual)
        if self._lfo_tick % 5 == 0:
            self._update_waveform_display(waveform)

    def _decode_and_update(self, x, y):
        waveform = self.synth.set_latent(x, y)
        self.root.after(0, self._update_waveform_display, waveform)

    # ------------------------------------------------------------------
    # Waveform display
    # ------------------------------------------------------------------

    def _update_waveform_display(self, waveform: np.ndarray):
        w, h   = WAVE_WIDTH, WAVE_HEIGHT
        margin = 6
        n      = len(waveform)
        step   = n / (w - 2 * margin)

        points = []
        for px in range(w - 2 * margin):
            val = waveform[int(px * step) % n]
            py  = margin + (1.0 - val) / 2.0 * (h - 2 * margin)
            points.extend([px + margin, py])

        self.wave_canvas.delete("all")
        for x in range(margin, w - margin, 6):
            self.wave_canvas.create_line(x, h//2, x+3, h//2,
                                          fill=MAC_SHADOW, width=1)
        if len(points) >= 4:
            self.wave_canvas.create_line(*points, fill=MAC_BLACK, width=1)

    # ------------------------------------------------------------------
    # MIDI
    # ------------------------------------------------------------------

    def _refresh_midi_ports(self):
        ports = self.midi.list_ports()
        menu  = self._midi_menu["menu"]
        menu.delete(0, "end")
        if ports:
            dev_id, name = ports[0]
            self._midi_var.set(name)
            for dev_id, name in ports:
                menu.add_command(
                    label=name,
                    command=lambda d=dev_id, n=name: (
                        self._midi_var.set(n), self.midi.open_port(d)))
            self.midi.open_port(ports[0][0])
        else:
            self._midi_var.set("(no ports)")
            menu.add_command(label="(no ports)", command=lambda: None)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _on_audio_select(self, selection):
        device_idx = int(selection.split(":")[0])
        self.synth.stop_audio()
        self.synth.start_audio(device=device_idx)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.lfo.stop()
        self.synth.stop_audio()
        self.midi.close()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Latent wavetable synthesizer")
    parser.add_argument("--model", default="export/decoder.onnx",
                        help="Path to decoder ONNX model")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run: python export/export.py --checkpoint model/checkpoints/best.pt")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    synth = LatentSynth(args.model)
    midi  = MidiInput(synth)

    print("Launching UI...")
    ui = SynthUI(synth, midi)

    print("Starting audio...")
    synth.start_audio()

    ui.run()


if __name__ == "__main__":
    main()
