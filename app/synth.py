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
    - Use Arpeggiator panel: step arp with per-step latent positions
    - Gain slider in I/O panel controls master output level
"""

import sys
import os
import argparse
import threading
import dataclasses
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from scipy.signal import lfilter
from scipy.spatial import KDTree

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
MAX_VOICES   = 8
PAD_SIZE     = 300
WAVE_WIDTH   = 300
WAVE_HEIGHT  = 60

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

ARP_MAX_STEPS    = 4
ARP_MIN_BPM      = 20.0
ARP_MAX_BPM      = 480.0
ARP_DEFAULT_BPM  = 120.0
ARP_DEFAULT_GATE = 0.8    # fraction of step duration, 0.05–1.0
ARP_TICK_HZ      = 200.0  # scheduler resolution (5 ms)

# Keyboard piano input
KB_NOTE_MAP: dict[str, int] = {
    "a": 0,  "w": 1,  "s": 2,  "e": 3,  "d": 4,
    "f": 5,  "t": 6,  "g": 7,  "y": 8,  "h": 9,
    "u": 10, "j": 11, "k": 12, "o": 13, "l": 14,
}
KB_DEFAULT_OCTAVE   = 4
KB_DEFAULT_VELOCITY = 100
KB_VELOCITY_STEP    = 10

CC_MAP_PATH        = os.path.expanduser("~/.latent_synth_cc.json")
SETTINGS_PATH      = os.path.expanduser("~/.latent_synth_settings.json")
LATENT_INDEX_PATH  = "export/latent_index.npz"
PRESETS_DIR        = os.path.expanduser("~/Documents/LatentSynth/Presets")


# ---------------------------------------------------------------------------
# CC Mapping
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CCMapping:
    """Binds a MIDI CC number to a synth parameter."""
    key:     str     # registry key, e.g. "latent_x"
    label:   str     # display name
    min_val: float
    max_val: float
    setter:  object  # callable(float) — must be thread-safe (uses root.after)


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
# Arpeggiator
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ArpStep:
    x:        float             = 0.0
    y:        float             = 0.0
    waveform: np.ndarray | None = None  # pre-decoded; None while decoding


class Arpeggiator:
    """
    Background daemon thread that fires arp notes at a set BPM.

    Each of the ARP_MAX_STEPS steps has its own latent position (decoded
    waveform). Held MIDI notes supply pitches; the arp cycles through them
    in the selected order. Per-step waveforms are decoded in background
    threads so the scheduler thread is never blocked by ONNX.
    """

    def __init__(self, synth, decode_fn, step_callback=None):
        self._synth         = synth
        self._decode_fn     = decode_fn        # callable(x, y) → np.ndarray
        self._step_callback = step_callback or (lambda i: None)

        self.bpm     = ARP_DEFAULT_BPM
        self.gate    = ARP_DEFAULT_GATE
        self.n_steps = ARP_MAX_STEPS
        self.order   = "up"   # "up" | "down" | "up-down" | "random"

        self._steps       = [ArpStep() for _ in range(ARP_MAX_STEPS)]
        self._held_notes  = []
        self._held_lock   = threading.Lock()
        self._active_note = None
        self._step_idx    = 0
        self._direction   = 1
        self._rng         = np.random.default_rng()

        self.active  = False
        self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self.active:
            return
        self._step_idx = 0
        self._direction = 1
        self.active  = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False
        if self._active_note is not None:
            self._synth._voice_note_off(self._active_note)
            self._active_note = None

    def set_bpm(self, bpm: float):
        self.bpm = float(np.clip(bpm, ARP_MIN_BPM, ARP_MAX_BPM))

    def set_gate(self, frac: float):
        self.gate = float(np.clip(frac, 0.05, 1.0))

    def set_n_steps(self, n: int):
        self.n_steps = int(np.clip(n, 1, ARP_MAX_STEPS))
        # Keep step_idx in bounds
        if self._step_idx >= self.n_steps:
            self._step_idx = 0

    def set_order(self, order: str):
        self.order = order
        self._direction = 1

    def set_step_latent(self, idx: int, x: float, y: float):
        """Update a step's latent position and trigger a background decode."""
        step = self._steps[idx]
        step.x = x
        step.y = y
        step.waveform = None   # invalidate until decode completes
        threading.Thread(
            target=self._decode_step, args=(idx, x, y), daemon=True
        ).start()

    def add_held(self, midi_note: int):
        with self._held_lock:
            if midi_note not in self._held_notes:
                self._held_notes.append(midi_note)

    def remove_held(self, midi_note: int):
        with self._held_lock:
            try:
                self._held_notes.remove(midi_note)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Background waveform decode
    # ------------------------------------------------------------------

    def _decode_step(self, idx: int, x: float, y: float):
        wf = self._decode_fn(x, y)
        step = self._steps[idx]
        if step.x == x and step.y == y:   # staleness guard
            step.waveform = wf            # GIL-atomic reference write

    # ------------------------------------------------------------------
    # Scheduling thread
    # ------------------------------------------------------------------

    def _pick_note(self, held):
        """Select pitch from held notes based on current order and step."""
        if self.order == "up":
            notes = sorted(held)
        elif self.order == "down":
            notes = sorted(held, reverse=True)
        elif self.order == "up-down":
            notes = sorted(held)
        else:  # random
            return int(self._rng.choice(held))
        return notes[self._step_idx % len(notes)]

    def _advance_step(self):
        """Move _step_idx forward according to the current order."""
        n = self.n_steps
        if self.order in ("up", "random"):
            self._step_idx = (self._step_idx + 1) % n
        elif self.order == "down":
            self._step_idx = (self._step_idx - 1 + n) % n
        else:  # up-down
            next_idx = self._step_idx + self._direction
            if next_idx >= n:
                self._direction = -1
                self._step_idx  = max(0, n - 2)
            elif next_idx < 0:
                self._direction = 1
                self._step_idx  = min(1, n - 1)
            else:
                self._step_idx = next_idx

    def _run(self):
        while self.active:
            step_duration = 60.0 / max(self.bpm, 1.0)
            gate          = float(np.clip(self.gate, 0.05, 1.0))
            n             = self.n_steps
            step          = self._steps[self._step_idx % ARP_MAX_STEPS]

            with self._held_lock:
                held = list(self._held_notes)

            if held:
                note = self._pick_note(held)

                # Notify UI before sleeping
                self._step_callback(self._step_idx % n)

                # Release previous note (ADSR release tail keeps sounding)
                if self._active_note is not None:
                    self._synth._voice_note_off(self._active_note)

                # Trigger this step's note with its own waveform
                self._synth.arp_note_on(note, 100, step.waveform)
                self._active_note = note

                # Hold for gate fraction
                threading.Event().wait(step_duration * gate)

                # Release (ADSR handles the tail)
                self._synth._voice_note_off(note)
                self._active_note = None

                # Rest for remaining step time
                remaining = step_duration * (1.0 - gate)
                if remaining > 1e-3:
                    threading.Event().wait(remaining)
            else:
                # No keys held — silence any lingering note and wait
                if self._active_note is not None:
                    self._synth._voice_note_off(self._active_note)
                    self._active_note = None
                threading.Event().wait(step_duration)

            self._advance_step()


# ---------------------------------------------------------------------------
# Polyphonic voice
# ---------------------------------------------------------------------------

class Voice:
    def __init__(self):
        self.midi_note      = None
        self._phase         = 0.0
        self._frequency     = 0.0
        self._velocity      = 0.0
        self.envelope       = ADSREnvelope(SAMPLE_RATE)
        self.filter_envelope = ADSREnvelope(SAMPLE_RATE)
        self.filt           = BiquadFilter(SAMPLE_RATE)
        self._own_waveform: np.ndarray | None = None

    @property
    def active(self):
        return self.envelope.state != ADSREnvelope.IDLE

    def note_on(self, midi_note: int, velocity: int, cutoff: float, resonance: float,
                waveform: np.ndarray | None = None):
        self.midi_note  = midi_note
        self._frequency = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        self._velocity  = velocity / 127.0
        self._own_waveform = waveform      # None → use global; array → per-voice
        self.filt.set_cutoff(cutoff)
        self.filt.set_resonance(resonance)
        if self.envelope.state == ADSREnvelope.IDLE:
            self.filt.reset()   # clear stale filter state on fresh starts only
        self.envelope.note_on()
        self.filter_envelope.note_on()

    def note_off(self):
        self.envelope.note_off()
        self.filter_envelope.note_off()
        self.midi_note = None

    def render(self, frames: int, waveform: np.ndarray,
               base_cutoff: float, env_amount: float) -> np.ndarray | None:
        """Render one audio block. Returns None if idle."""
        if not self.active:
            return None

        # Use per-voice waveform if set (arp), otherwise fall back to global
        wf        = self._own_waveform if self._own_waveform is not None else waveform
        phase_inc = WAVEFORM_LEN * self._frequency / SAMPLE_RATE
        phases    = (self._phase + np.arange(frames, dtype=np.float64) * phase_inc) % WAVEFORM_LEN
        idx       = phases.astype(np.int32)
        frac      = (phases - idx).astype(np.float32)
        next_idx  = (idx + 1) % WAVEFORM_LEN
        samples   = wf[idx] * (1.0 - frac) + wf[next_idx] * frac
        self._phase = (self._phase + frames * phase_inc) % WAVEFORM_LEN

        env      = self.envelope.process(frames)
        filt_env = self.filter_envelope.process(frames)

        if env_amount > 0.0:
            filt_level = float(filt_env.mean())
            mod_cutoff = base_cutoff + env_amount * (18000.0 - base_cutoff) * filt_level
            self.filt.set_cutoff(mod_cutoff)

        samples = self.filt.process(samples)
        return samples * env * self._velocity


# ---------------------------------------------------------------------------
# Wavetable oscillator / audio engine
# ---------------------------------------------------------------------------

class LatentSynth:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])

        self._waveform_from = np.zeros(WAVEFORM_LEN, dtype=np.float32)
        self._waveform_to   = np.zeros(WAVEFORM_LEN, dtype=np.float32)
        self._morph         = 1.0   # 0.0 = fully _from, 1.0 = fully _to
        self._morph_rate    = 0.0   # fraction of morph per audio block; 0 = instant
        self._gain       = 0.8   # master output gain (0-1)

        self._voices    = [Voice() for _ in range(MAX_VOICES)]
        self._resonance = 0.0   # stored so note_on can initialise stolen/reused voices
        self._base_cutoff  = 18000.0
        self._env_amount   = 0.0

        # Oscilloscope ring buffer — written by audio callback, read by UI
        _OSC_LEN        = 4096
        self._osc_buf   = np.zeros(_OSC_LEN, dtype=np.float32)
        self._osc_pos   = 0    # write head

        self.set_latent(0.0, 0.0)

        # Arpeggiator
        self._arp_enabled = False
        self.arp = Arpeggiator(
            synth     = self,
            decode_fn = self._decode_latent,
        )
        # Initialise all step waveforms from the already-decoded default waveform
        for step in self.arp._steps:
            step.waveform = self._waveform_to.copy()

    # ------------------------------------------------------------------
    # Latent navigation
    # ------------------------------------------------------------------

    def set_latent(self, x: float, y: float):
        z = np.array([[x, y]], dtype=np.float32)
        new_wf = self.session.run(["waveform"], {"latent": z})[0][0, 0].astype(np.float32)

        if self._morph_rate > 0.0 and self._morph < 1.0:
            # Mid-transition: freeze the current blend position as the new "from"
            t = float(self._morph)
            self._waveform_from = (1.0 - t) * self._waveform_from + t * self._waveform_to
        else:
            self._waveform_from = self._waveform_to   # previous target becomes new start

        self._waveform_to = new_wf
        # With glide off, snap immediately; otherwise restart morph from 0
        self._morph = 1.0 if self._morph_rate == 0.0 else 0.0
        return new_wf

    def set_glide(self, ms: float):
        if ms <= 0.0:
            self._morph_rate = 0.0
        else:
            glide_samples = ms * SAMPLE_RATE / 1000.0
            self._morph_rate = BLOCK_SIZE / glide_samples

    def _decode_latent(self, x: float, y: float) -> np.ndarray:
        """ONNX inference — thread-safe (session is stateless)."""
        z = np.array([[x, y]], dtype=np.float32)
        return self.session.run(["waveform"], {"latent": z})[0][0, 0].astype(np.float32)

    # ------------------------------------------------------------------
    # MIDI
    # ------------------------------------------------------------------

    def note_on(self, midi_note: int, velocity: int):
        if self._arp_enabled:
            self.arp.add_held(midi_note)
            return

        # Retrigger if already playing this note
        for v in self._voices:
            if v.midi_note == midi_note:
                v.note_on(midi_note, velocity, self._base_cutoff, self._resonance)
                return

        # Prefer a free (IDLE) voice
        for v in self._voices:
            if not v.active:
                v.note_on(midi_note, velocity, self._base_cutoff, self._resonance)
                return

        # Steal: quietest releasing voice first, then quietest overall
        best = min(
            (v for v in self._voices if v.envelope.state == ADSREnvelope.RELEASE),
            key=lambda v: v.envelope.level,
            default=None,
        )
        if best is None:
            best = min(self._voices, key=lambda v: v.envelope.level)
        best.note_on(midi_note, velocity, self._base_cutoff, self._resonance)

    def note_off(self, midi_note: int):
        if self._arp_enabled:
            self.arp.remove_held(midi_note)
            return
        for v in self._voices:
            if v.midi_note == midi_note:
                v.note_off()
                return

    def arp_note_on(self, midi_note: int, velocity: int,
                    waveform: np.ndarray | None = None):
        """Trigger a voice for the arpeggiator with an optional per-step waveform."""
        # Retrigger if already playing this note
        for v in self._voices:
            if v.midi_note == midi_note:
                v.note_on(midi_note, velocity, self._base_cutoff, self._resonance,
                          waveform=waveform)
                return

        # Prefer a free (IDLE) voice
        for v in self._voices:
            if not v.active:
                v.note_on(midi_note, velocity, self._base_cutoff, self._resonance,
                          waveform=waveform)
                return

        # Steal: quietest releasing voice first, then quietest overall
        best = min(
            (v for v in self._voices if v.envelope.state == ADSREnvelope.RELEASE),
            key=lambda v: v.envelope.level,
            default=None,
        )
        if best is None:
            best = min(self._voices, key=lambda v: v.envelope.level)
        best.note_on(midi_note, velocity, self._base_cutoff, self._resonance,
                     waveform=waveform)

    def _voice_note_off(self, midi_note: int):
        """Direct voice release — used by Arpeggiator (bypasses arp routing)."""
        for v in self._voices:
            if v.midi_note == midi_note:
                v.note_off()
                return

    # ------------------------------------------------------------------
    # Param setters  (called from UI thread)
    # ------------------------------------------------------------------

    def set_attack(self,  ms):    [v.envelope.set_attack(ms)     for v in self._voices]
    def set_decay(self,   ms):    [v.envelope.set_decay(ms)      for v in self._voices]
    def set_sustain(self, level): [v.envelope.set_sustain(level) for v in self._voices]
    def set_release(self, ms):    [v.envelope.set_release(ms)    for v in self._voices]

    def set_fenv_attack(self,  ms):    [v.filter_envelope.set_attack(ms)     for v in self._voices]
    def set_fenv_decay(self,   ms):    [v.filter_envelope.set_decay(ms)      for v in self._voices]
    def set_fenv_sustain(self, level): [v.filter_envelope.set_sustain(level) for v in self._voices]
    def set_fenv_release(self, ms):    [v.filter_envelope.set_release(ms)    for v in self._voices]

    def set_cutoff(self, hz):
        self._base_cutoff = hz
        for v in self._voices:
            v.filt.set_cutoff(hz)

    def set_filter_resonance(self, r):
        self._resonance = r
        for v in self._voices:
            v.filt.set_resonance(r)

    def set_env_amount(self, amount):
        self._env_amount = float(np.clip(amount, 0.0, 1.0))

    def set_gain(self, g: float):
        self._gain = float(np.clip(g, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Arpeggiator control
    # ------------------------------------------------------------------

    def set_arp_enabled(self, enabled: bool):
        self._arp_enabled = enabled
        if enabled:
            self.arp.start()
        else:
            self.arp.stop()
            for v in self._voices:   # release any arp-held voices
                if v.active:
                    v.note_off()

    def set_arp_bpm(self, bpm):          self.arp.set_bpm(bpm)
    def set_arp_gate(self, frac):        self.arp.set_gate(frac)
    def set_arp_steps(self, n):          self.arp.set_n_steps(n)
    def set_arp_order(self, order):      self.arp.set_order(order)
    def set_step_latent(self, idx, x, y): self.arp.set_step_latent(idx, x, y)

    # ------------------------------------------------------------------
    # Audio callback  (runs on sounddevice thread)
    # ------------------------------------------------------------------

    def audio_callback(self, outdata, frames, time_info, status):
        # Snapshot morph state atomically at block start
        morph      = self._morph
        wf_from    = self._waveform_from
        wf_to      = self._waveform_to
        morph_rate = self._morph_rate

        # Advance morph for next block
        if morph < 1.0:
            self._morph = min(1.0, morph + morph_rate)

        # Compute effective waveform for this block
        if morph >= 1.0:
            waveform = wf_to
        else:
            waveform = (1.0 - morph) * wf_from + morph * wf_to

        # ── Mix voices ────────────────────────────────────────────────
        base_cutoff = self._base_cutoff
        env_amount  = self._env_amount
        mix         = np.zeros(frames, dtype=np.float32)
        n_active    = 0

        for v in self._voices:
            rendered = v.render(frames, waveform, base_cutoff, env_amount)
            if rendered is not None:
                mix      += rendered
                n_active += 1

        if n_active > 0:
            mix *= self._gain / (n_active ** 0.5)

        # Feed oscilloscope ring buffer
        buf_len = len(self._osc_buf)
        pos     = self._osc_pos
        end     = pos + frames
        if end <= buf_len:
            self._osc_buf[pos:end] = mix
        else:
            cut = buf_len - pos
            self._osc_buf[pos:]  = mix[:cut]
            self._osc_buf[:frames - cut] = mix[cut:]
        self._osc_pos = end % buf_len

        outdata[:, 0] = mix

    def get_oscilloscope_data(self, n_display: int = 1024) -> np.ndarray:
        """Return a zero-crossing-triggered window of recent audio output."""
        pos = self._osc_pos
        # Linearise ring buffer: oldest sample first
        buf = np.concatenate([self._osc_buf[pos:], self._osc_buf[:pos]])
        # Search for a rising zero crossing in the second half
        search_start = max(0, len(buf) - 2 * n_display)
        for i in range(search_start, len(buf) - n_display):
            if buf[i] <= 0.0 < buf[i + 1]:
                return buf[i:i + n_display]
        return buf[-n_display:]

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
        self.synth             = synth
        self._input            = None
        self._active           = False
        self._thread           = None
        self.cc_map: dict      = {}    # {cc_num (int): CCMapping}
        self._learn_callback   = None  # callable(cc_num) during MIDI learn, else None

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
                        elif status == 0xB0:
                            self._handle_cc(data[1], data[2])
            except Exception:
                break
            threading.Event().wait(0.001)

    def _handle_cc(self, cc_num: int, cc_val: int):
        """Dispatch a CC event: complete a learn, or apply a mapped value."""
        if self._learn_callback is not None:
            cb, self._learn_callback = self._learn_callback, None
            cb(cc_num)
            return
        m = self.cc_map.get(cc_num)
        if m:
            scaled = m.min_val + (cc_val / 127.0) * (m.max_val - m.min_val)
            m.setter(scaled)

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

        self._kb_active   = False
        self._kb_octave   = KB_DEFAULT_OCTAVE
        self._kb_velocity = KB_DEFAULT_VELOCITY
        self._kb_held     = set()   # held keysyms — prevents OS key-repeat flood

        self._learn_target: dict | None = None  # {'key': str, 'btn': tk.Button}
        self._cc_buttons:   dict        = {}    # {key: tk.Button}

        # LFO: callback fires on main thread via root.after
        self.lfo = LatentLFO(
            lambda x, y: self.root.after(0, self._lfo_update, x, y)
        )

        self._build_ui()
        self._update_waveform_display(synth._waveform_to)
        self._setup_keyboard_input()
        self._load_cc_map()
        self._load_settings()
        self._wave_tree      = None
        self._wave_filenames = None
        self._try_load_latent_index()
        self._update_wave_name(synth._waveform_to)
        self._osc_timer()

        # Wire arp step highlight callback (after _build_ui creates the labels)
        self.synth.arp._step_callback = lambda i: self.root.after(
            0, self._highlight_arp_step, i)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self.root
        PAD  = 6

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

        # ── Menu bar ──────────────────────────────────────────────────
        self._build_menubar(chrome)

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

        self._wave_name_var = tk.StringVar(value="")
        tk.Label(left_col, textvariable=self._wave_name_var,
                 font=FONT_TINY, fg=MAC_SHADOW, bg=MAC_WHITE,
                 anchor="w").pack(fill="x", pady=(1, 0))

        # ── Right column panels ───────────────────────────────────────
        self._build_envelope_panel(right_col)
        self._build_filter_panel(right_col)
        self._build_motion_panel(right_col)
        self._build_arpeggiator_panel(right_col)
        self._build_io_panel(right_col)

    # ------------------------------------------------------------------
    # Menu bar + presets
    # ------------------------------------------------------------------

    def _build_menubar(self, chrome):
        bar = tk.Frame(chrome, bg=MAC_BG, bd=0)
        bar.pack(fill="x")
        tk.Frame(chrome, bg=MAC_SHADOW, height=1).pack(fill="x")

        file_menu = tk.Menu(self.root, tearoff=0,
                            bg=MAC_WHITE, fg=MAC_BLACK, font=FONT_SMALL,
                            activebackground=MAC_BLACK, activeforeground=MAC_WHITE,
                            bd=1, relief="solid")
        file_menu.add_command(label="Save Preset…", command=self._save_preset)
        file_menu.add_command(label="Load Preset…", command=self._load_preset)

        def _show(event):
            w = event.widget
            file_menu.tk_popup(w.winfo_rootx(), w.winfo_rooty() + w.winfo_height())

        file_btn = tk.Label(bar, text=" File ", font=FONT_SMALL,
                            fg=MAC_BLACK, bg=MAC_BG, padx=4, pady=2,
                            relief="flat", cursor="arrow")
        file_btn.pack(side="left")
        file_btn.bind("<Button-1>", _show)
        file_btn.bind("<Enter>", lambda e: file_btn.config(bg=MAC_SHADOW))
        file_btn.bind("<Leave>", lambda e: file_btn.config(bg=MAC_BG))

    # ── Preset collect / apply ────────────────────────────────────────

    def _collect_preset(self) -> dict:
        steps = [
            {"z0": float(self._arp_step_z0_sls[i].get()),
             "z1": float(self._arp_step_z1_sls[i].get())}
            for i in range(ARP_MAX_STEPS)
        ]
        return {
            # latent position
            "latent_x": self._latent_x,
            "latent_y": self._latent_y,
            # amplitude envelope
            "atk": float(self._atk.get()),
            "dec": float(self._dec.get()),
            "sus": float(self._sus.get()),
            "rel": float(self._rel.get()),
            # filter
            "cutoff":     float(self._cutoff_sl.get()),
            "resonance":  float(self._res_sl.get()),
            "env_amount": float(self._env_amt_sl.get()),
            # filter envelope
            "f_atk": float(self._fatk.get()),
            "f_dec": float(self._fdec.get()),
            "f_sus": float(self._fsus.get()),
            "f_rel": float(self._frel.get()),
            # motion
            "lfo_active": self.lfo.active,
            "lfo_shape":  self._lfo_shape_var.get(),
            "lfo_rate":   float(self._lfo_rate_sl.get()),
            "lfo_depth":  float(self._lfo_depth_sl.get()),
            "glide":      float(self._glide_sl.get()),
            "z0":         float(self._z0_sl.get()),
            "z1":         float(self._z1_sl.get()),
            # arpeggiator
            "arp_enabled": self.synth._arp_enabled,
            "arp_steps":   self.synth._arp_steps,
            "arp_order":   self._arp_order_var.get(),
            "arp_bpm":     float(self._arp_bpm_sl.get()),
            "arp_gate":    float(self._arp_gate_sl.get()),
            "arp_step_positions": steps,
            # gain
            "gain": float(self._gain_sl.get()),
        }

    def _apply_preset(self, d: dict):
        """Set every control to the values in d, calling existing callbacks."""
        g = d.get   # short alias

        # Latent position
        self._move_to_latent(float(g("latent_x", self._latent_x)),
                             float(g("latent_y", self._latent_y)))

        # Amplitude envelope
        for sl, key in [(self._atk, "atk"), (self._dec, "dec"),
                        (self._sus, "sus"), (self._rel, "rel")]:
            if key in d:
                sl.set(g(key))
        self._apply_adsr()

        # Filter main sliders
        if "cutoff" in d:
            self._cutoff_sl.set(g("cutoff"))
            self._on_cutoff(g("cutoff"))
        if "resonance" in d:
            self._res_sl.set(g("resonance"))
            self._on_resonance(g("resonance"))
        if "env_amount" in d:
            self._env_amt_sl.set(g("env_amount"))
            self._on_env_amount(g("env_amount"))

        # Filter envelope
        for sl, key in [(self._fatk, "f_atk"), (self._fdec, "f_dec"),
                        (self._fsus, "f_sus"), (self._frel, "f_rel")]:
            if key in d:
                sl.set(g(key))
        self._apply_filter_adsr()

        # Motion
        if "lfo_shape" in d:
            self._lfo_shape_var.set(g("lfo_shape"))
            self._on_lfo_shape()
        if "lfo_rate" in d:
            self._lfo_rate_sl.set(g("lfo_rate"))
            self._on_lfo_rate(g("lfo_rate"))
        if "lfo_depth" in d:
            self._lfo_depth_sl.set(g("lfo_depth"))
            self._on_lfo_depth(g("lfo_depth"))
        if "glide" in d:
            self._glide_sl.set(g("glide"))
            self._on_glide(g("glide"))
        if "z0" in d:
            self._z0_sl.set(g("z0"))
        if "z1" in d:
            self._z1_sl.set(g("z1"))

        # LFO toggle — only change state if it differs
        want_lfo = bool(g("lfo_active", self.lfo.active))
        if want_lfo != self.lfo.active:
            self._toggle_lfo()

        # Arpeggiator
        if "arp_order" in d:
            self._arp_order_var.set(g("arp_order"))
            self._on_arp_order()
        if "arp_bpm" in d:
            self._arp_bpm_sl.set(g("arp_bpm"))
            self._on_arp_bpm(g("arp_bpm"))
        if "arp_gate" in d:
            self._arp_gate_sl.set(g("arp_gate"))
            self._on_arp_gate(g("arp_gate"))
        if "arp_steps" in d:
            self._set_arp_steps(int(g("arp_steps")))
        for i, pos in enumerate(g("arp_step_positions", [])):
            if i < ARP_MAX_STEPS:
                self._arp_step_z0_sls[i].set(pos.get("z0", 0.0))
                self._arp_step_z1_sls[i].set(pos.get("z1", 0.0))
                self._on_arp_step_latent(i)

        want_arp = bool(g("arp_enabled", self.synth._arp_enabled))
        if want_arp != self.synth._arp_enabled:
            self._toggle_arp()

        # Gain
        if "gain" in d:
            self._gain_sl.set(g("gain"))
            self._on_gain(g("gain"))

    # ── File dialog helpers ───────────────────────────────────────────

    def _save_preset(self):
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save Preset",
            initialdir=PRESETS_DIR,
            defaultextension=".json",
            filetypes=[("Preset files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._collect_preset(), f, indent=2)
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    def _load_preset(self):
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Load Preset",
            initialdir=PRESETS_DIR,
            filetypes=[("Preset files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                d = json.load(f)
            self._apply_preset(d)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            messagebox.showerror("Load failed", str(e))

    # ------------------------------------------------------------------
    # Collapsible panel helper
    # ------------------------------------------------------------------

    def _make_collapsible_panel(self, parent, title, pady=(0, 4), start_open=True):
        """Create a Mac-style bordered panel with a collapsible body.

        Returns (panel, body).  Clicking the ▼/▶ button in the header
        hides or reveals `body`.  When start_open=False the body starts hidden.
        """
        panel = mac_frame(parent)
        panel.pack(fill="x", pady=pady)

        header = tk.Frame(panel, bg=MAC_BG)
        header.pack(fill="x", padx=6, pady=(3, 1))

        _open = [start_open]

        toggle_btn = tk.Button(
            header,
            text="▼" if start_open else "▶",
            font=FONT_TINY, fg=MAC_SHADOW, bg=MAC_BG,
            relief="flat", bd=0, width=2,
            activebackground=MAC_BG, cursor="arrow",
        )
        toggle_btn.pack(side="left", padx=(0, 2))

        tk.Label(header, text=title,
                 font=("Monaco", 9, "bold"),
                 fg=MAC_BLACK, bg=MAC_BG).pack(side="left")

        body = tk.Frame(panel, bg=MAC_BG)
        if start_open:
            body.pack(fill="x")

        def _toggle():
            if _open[0]:
                body.pack_forget()
                toggle_btn.config(text="▶")
            else:
                body.pack(fill="x")
                toggle_btn.config(text="▼")
            _open[0] = not _open[0]

        toggle_btn.config(command=_toggle)
        # clicking the title label also toggles
        header.bind("<Button-1>", lambda e: _toggle())
        return panel, body

    # ------------------------------------------------------------------
    # Envelope panel
    # ------------------------------------------------------------------

    def _build_envelope_panel(self, parent):
        _, body = self._make_collapsible_panel(parent, "◆ Envelope", pady=(0, 4))

        sliders_row = tk.Frame(body, bg=MAC_BG)
        sliders_row.pack(padx=6, pady=(0, 4))

        self._env_readout = tk.StringVar(value="A:10ms  D:100ms  S:0.70  R:300ms")
        tk.Label(body, textvariable=self._env_readout,
                 font=FONT_TINY, fg=MAC_BLACK, bg=MAC_BG).pack(pady=(0, 3))

        def make_adsr_slider(col, label, from_, to, init, cmd, cc_key=None):
            f = tk.Frame(sliders_row, bg=MAC_BG)
            f.grid(row=0, column=col, padx=4)
            s = mac_slider(f, from_=from_, to=to, command=cmd, length=80)
            s.set(init)
            s.pack()
            tk.Label(f, text=label, font=("Monaco", 9, "bold"),
                     fg=MAC_BLACK, bg=MAC_BG).pack()
            if cc_key:
                btn = tk.Button(f, text="CC", font=FONT_TINY,
                                fg=MAC_BLACK, bg=MAC_BG,
                                relief="raised", bd=2, width=3,
                                activebackground=MAC_HILIGHT)
                btn.config(command=lambda k=cc_key, b=btn: self._start_learn(k, b))
                btn.pack()
                self._cc_buttons[cc_key] = btn
            return s

        self._atk = make_adsr_slider(0, "A", 100, 0, 30,
                                      lambda v: self._on_adsr("A", v),
                                      cc_key="attack")
        self._dec = make_adsr_slider(1, "D", 100, 0, 55,
                                      lambda v: self._on_adsr("D", v),
                                      cc_key="decay")
        self._sus = make_adsr_slider(2, "S", 100, 0, 70,
                                      lambda v: self._on_adsr("S", v),
                                      cc_key="sustain")
        self._rel = make_adsr_slider(3, "R", 100, 0, 60,
                                      lambda v: self._on_adsr("R", v),
                                      cc_key="release")

        self._apply_adsr()

    def _log_time(self, slider_val):
        return 1.0 * (8000.0 ** (float(slider_val) / 100.0))

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
        _, body = self._make_collapsible_panel(parent, "◆ Filter", pady=(0, 4))

        inner = tk.Frame(body, bg=MAC_BG)
        inner.pack(padx=6, pady=(0, 4), fill="x")

        def make_row(label, from_, to, init, cmd, cc_key=None):
            row = tk.Frame(inner, bg=MAC_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=FONT_TINY, fg=MAC_BLACK,
                     bg=MAC_BG, width=10, anchor="w").pack(side="left")
            s = mac_slider(row, from_=from_, to=to, command=cmd,
                           length=110, orient=tk.HORIZONTAL)
            s.set(init)
            s.pack(side="left")
            val_lbl = tk.Label(row, text="", font=FONT_TINY,
                                fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w")
            val_lbl.pack(side="left", padx=(4, 0))
            if cc_key:
                btn = tk.Button(row, text="CC", font=FONT_TINY,
                                fg=MAC_BLACK, bg=MAC_BG,
                                relief="raised", bd=2, width=3,
                                activebackground=MAC_HILIGHT)
                btn.config(command=lambda k=cc_key, b=btn: self._start_learn(k, b))
                btn.pack(side="left", padx=(2, 0))
                self._cc_buttons[cc_key] = btn
            return s, val_lbl

        self._cutoff_sl, self._cutoff_lbl = make_row(
            "Cutoff:", 0, 100, 85, self._on_cutoff, cc_key="cutoff")
        self._res_sl, self._res_lbl = make_row(
            "Resonance:", 0, 100, 0, self._on_resonance, cc_key="resonance")
        self._env_amt_sl, self._env_amt_lbl = make_row(
            "Env Amount:", 0, 100, 0, self._on_env_amount, cc_key="env_amount")

        self._on_cutoff(self._cutoff_sl.get())
        self._on_resonance(self._res_sl.get())
        self._on_env_amount(self._env_amt_sl.get())

        # ── Filter envelope (F.Env) ADSR ─────────────────────────────
        tk.Frame(body, bg=MAC_SHADOW, height=1).pack(fill="x", padx=6, pady=(2, 2))

        fenv_top = tk.Frame(body, bg=MAC_BG)
        fenv_top.pack(fill="x", padx=6)
        tk.Label(fenv_top, text="F.Env", font=FONT_TINY,
                 fg=MAC_BLACK, bg=MAC_BG).pack(side="left")

        self._fenv_readout = tk.StringVar(value="A:10ms  D:100ms  S:0.70  R:300ms")
        tk.Label(body, textvariable=self._fenv_readout,
                 font=FONT_TINY, fg=MAC_BLACK, bg=MAC_BG).pack(pady=(0, 1))

        fenv_row = tk.Frame(body, bg=MAC_BG)
        fenv_row.pack(padx=6, pady=(0, 4))

        def make_fenv_slider(col, label, from_, to, init, cmd, cc_key=None):
            f = tk.Frame(fenv_row, bg=MAC_BG)
            f.grid(row=0, column=col, padx=4)
            s = mac_slider(f, from_=from_, to=to, command=cmd, length=70)
            s.set(init)
            s.pack()
            tk.Label(f, text=label, font=("Monaco", 9, "bold"),
                     fg=MAC_BLACK, bg=MAC_BG).pack()
            if cc_key:
                btn = tk.Button(f, text="CC", font=FONT_TINY,
                                fg=MAC_BLACK, bg=MAC_BG,
                                relief="raised", bd=2, width=3,
                                activebackground=MAC_HILIGHT)
                btn.config(command=lambda k=cc_key, b=btn: self._start_learn(k, b))
                btn.pack()
                self._cc_buttons[cc_key] = btn
            return s

        self._fatk = make_fenv_slider(0, "A", 100, 0, 30,
                                      lambda v: self._on_filter_adsr("A", v),
                                      cc_key="f_attack")
        self._fdec = make_fenv_slider(1, "D", 100, 0, 55,
                                      lambda v: self._on_filter_adsr("D", v),
                                      cc_key="f_decay")
        self._fsus = make_fenv_slider(2, "S", 100, 0, 70,
                                      lambda v: self._on_filter_adsr("S", v),
                                      cc_key="f_sustain")
        self._frel = make_fenv_slider(3, "R", 100, 0, 60,
                                      lambda v: self._on_filter_adsr("R", v),
                                      cc_key="f_release")

        self._apply_filter_adsr()

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

    def _on_filter_adsr(self, param, val):
        self._apply_filter_adsr()

    def _apply_filter_adsr(self):
        atk_ms = self._log_time(self._fatk.get())
        dec_ms = self._log_time(self._fdec.get())
        sus    = self._fsus.get() / 100.0
        rel_ms = self._log_time(self._frel.get())

        self.synth.set_fenv_attack(atk_ms)
        self.synth.set_fenv_decay(dec_ms)
        self.synth.set_fenv_sustain(sus)
        self.synth.set_fenv_release(rel_ms)

        self._fenv_readout.set(
            f"A:{atk_ms:.0f}ms  D:{dec_ms:.0f}ms  "
            f"S:{sus:.2f}  R:{rel_ms:.0f}ms"
        )

    # ------------------------------------------------------------------
    # Motion panel  (LFO + scan sliders + randomize)
    # ------------------------------------------------------------------

    def _build_motion_panel(self, parent):
        _, body = self._make_collapsible_panel(
            parent, "◆ Motion", pady=(0, 6), start_open=False)

        inner = tk.Frame(body, bg=MAC_BG)
        inner.pack(padx=6, pady=(0, 4), fill="x")

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
        def make_row(label, from_, to, init, cmd, res=0, cc_key=None):
            row = tk.Frame(inner, bg=MAC_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=FONT_TINY, fg=MAC_BLACK,
                     bg=MAC_BG, width=7, anchor="w").pack(side="left")
            s = mac_slider(row, from_=from_, to=to, command=cmd,
                           length=100, orient=tk.HORIZONTAL, resolution=res)
            s.set(init)
            s.pack(side="left")
            lbl = tk.Label(row, text="", font=FONT_TINY,
                            fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w")
            lbl.pack(side="left", padx=(3, 0))
            if cc_key:
                btn = tk.Button(row, text="CC", font=FONT_TINY,
                                fg=MAC_BLACK, bg=MAC_BG,
                                relief="raised", bd=2, width=3,
                                activebackground=MAC_HILIGHT)
                btn.config(command=lambda k=cc_key, b=btn: self._start_learn(k, b))
                btn.pack(side="left", padx=(2, 0))
                self._cc_buttons[cc_key] = btn
            return s, lbl

        self._lfo_rate_sl,  self._lfo_rate_lbl  = make_row(
            "Rate:",  1, 100, 20, self._on_lfo_rate, cc_key="lfo_rate")
        self._lfo_depth_sl, self._lfo_depth_lbl = make_row(
            "Depth:", 0, 100, 50, self._on_lfo_depth, cc_key="lfo_depth")
        self._glide_sl, self._glide_lbl = make_row(
            "Glide:", 0, 1000, 0, self._on_glide, cc_key="glide")
        self._z0_sl, self._z0_lbl = make_row(
            "Z0:", LATENT_MIN, LATENT_MAX, 0.0, self._on_z0_scan, res=0.01,
            cc_key="z0")
        self._z1_sl, self._z1_lbl = make_row(
            "Z1:", LATENT_MIN, LATENT_MAX, 0.0, self._on_z1_scan, res=0.01,
            cc_key="z1")

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

    def _on_glide(self, val):
        ms = float(val)
        self.synth.set_glide(ms)
        if ms == 0:
            self._glide_lbl.config(text="off")
        else:
            self._glide_lbl.config(text=f"{ms:.0f} ms")

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
    # Arpeggiator panel
    # ------------------------------------------------------------------

    def _build_arpeggiator_panel(self, parent):
        _, body = self._make_collapsible_panel(
            parent, "◆ Arpeggiator", pady=(0, 6), start_open=False)

        inner = tk.Frame(body, bg=MAC_BG)
        inner.pack(padx=6, pady=(0, 4), fill="x")

        # Row 1: ARP toggle + Steps buttons
        row1 = tk.Frame(inner, bg=MAC_BG)
        row1.pack(fill="x", pady=(0, 2))

        self._arp_btn = mac_button(row1, "ARP: OFF", self._toggle_arp, width=8)
        self._arp_btn.pack(side="left", padx=(0, 6))

        tk.Label(row1, text="Steps:", font=FONT_TINY,
                 fg=MAC_BLACK, bg=MAC_BG).pack(side="left")

        self._arp_step_btns = []
        for i in range(1, ARP_MAX_STEPS + 1):
            b = mac_button(row1, str(i), lambda n=i: self._set_arp_steps(n), width=2)
            b.pack(side="left", padx=1)
            self._arp_step_btns.append(b)

        # Row 2: Order radio buttons
        row2 = tk.Frame(inner, bg=MAC_BG)
        row2.pack(fill="x", pady=2)

        tk.Label(row2, text="Order:", font=FONT_TINY,
                 fg=MAC_BLACK, bg=MAC_BG, width=7, anchor="w").pack(side="left")

        self._arp_order_var = tk.StringVar(value="up")
        for label, val in [("Up", "up"), ("Down", "down"),
                            ("U-D", "up-down"), ("Rand", "random")]:
            tk.Radiobutton(
                row2, text=label, variable=self._arp_order_var, value=val,
                font=FONT_TINY, fg=MAC_BLACK, bg=MAC_BG,
                activebackground=MAC_BG, selectcolor=MAC_BG,
                command=self._on_arp_order,
            ).pack(side="left", padx=2)

        # BPM / Gate sliders
        def make_row(label, from_, to, init, cmd, res=0):
            row = tk.Frame(inner, bg=MAC_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=FONT_TINY, fg=MAC_BLACK,
                     bg=MAC_BG, width=7, anchor="w").pack(side="left")
            s = mac_slider(row, from_=from_, to=to, command=cmd,
                           length=110, orient=tk.HORIZONTAL, resolution=res)
            s.set(init)
            s.pack(side="left")
            lbl = tk.Label(row, text="", font=FONT_TINY,
                           fg=MAC_BLACK, bg=MAC_BG, width=7, anchor="w")
            lbl.pack(side="left", padx=(3, 0))
            return s, lbl

        self._arp_bpm_sl,  self._arp_bpm_lbl  = make_row(
            "BPM:",  ARP_MIN_BPM, ARP_MAX_BPM, ARP_DEFAULT_BPM,
            self._on_arp_bpm, res=1)
        self._arp_gate_sl, self._arp_gate_lbl = make_row(
            "Gate:", 5, 100, int(ARP_DEFAULT_GATE * 100),
            self._on_arp_gate, res=1)

        # Divider
        tk.Frame(inner, bg=MAC_SHADOW, height=1).pack(fill="x", pady=(4, 2))

        # Step rows — each has a label, z0 slider, z1 slider, capture button
        self._arp_step_labels = []
        self._arp_step_z0_sls = []
        self._arp_step_z1_sls = []

        for i in range(ARP_MAX_STEPS):
            row = tk.Frame(inner, bg=MAC_BG)
            row.pack(fill="x", pady=1)

            lbl = tk.Label(row, text=f"Step {i+1}:", font=FONT_TINY,
                           fg=MAC_BLACK, bg=MAC_BG, width=7, anchor="w")
            lbl.pack(side="left")
            self._arp_step_labels.append(lbl)

            tk.Label(row, text="z0", font=FONT_TINY,
                     fg=MAC_BLACK, bg=MAC_BG).pack(side="left")

            z0_sl = mac_slider(
                row, from_=LATENT_MIN, to=LATENT_MAX,
                command=lambda v, idx=i: self._on_arp_step_latent(idx),
                length=75, orient=tk.HORIZONTAL, resolution=0.01)
            z0_sl.set(0.0)
            z0_sl.pack(side="left")
            self._arp_step_z0_sls.append(z0_sl)

            tk.Label(row, text=" z1", font=FONT_TINY,
                     fg=MAC_BLACK, bg=MAC_BG).pack(side="left")

            z1_sl = mac_slider(
                row, from_=LATENT_MIN, to=LATENT_MAX,
                command=lambda v, idx=i: self._on_arp_step_latent(idx),
                length=75, orient=tk.HORIZONTAL, resolution=0.01)
            z1_sl.set(0.0)
            z1_sl.pack(side="left")
            self._arp_step_z1_sls.append(z1_sl)

            mac_button(row, "◉", lambda idx=i: self._grab_step_pos(idx),
                       width=2).pack(side="left", padx=(3, 0))

        # Apply defaults and set initial visual state
        self._on_arp_bpm(ARP_DEFAULT_BPM)
        self._on_arp_gate(int(ARP_DEFAULT_GATE * 100))
        self._set_arp_steps(ARP_MAX_STEPS)   # all 4 steps active initially
        self._highlight_arp_step(0)

    # ------------------------------------------------------------------
    # Arpeggiator callbacks
    # ------------------------------------------------------------------

    def _toggle_arp(self):
        enabled = not self.synth._arp_enabled
        self.synth.set_arp_enabled(enabled)
        self._arp_btn.config(text="ARP: ON " if enabled else "ARP: OFF")

    def _set_arp_steps(self, n: int):
        self.synth.set_arp_steps(n)
        for i, b in enumerate(self._arp_step_btns):
            b.config(relief="sunken" if (i + 1) <= n else "raised")

    def _on_arp_order(self):
        self.synth.set_arp_order(self._arp_order_var.get())

    def _on_arp_bpm(self, val):
        bpm = float(val)
        self.synth.set_arp_bpm(bpm)
        self._arp_bpm_lbl.config(text=f"{bpm:.0f}")

    def _on_arp_gate(self, val):
        frac = float(val) / 100.0
        self.synth.set_arp_gate(frac)
        self._arp_gate_lbl.config(text=f"{int(float(val))}%")

    def _on_arp_step_latent(self, idx: int):
        """Called when either z0 or z1 slider for a step changes."""
        x = float(self._arp_step_z0_sls[idx].get())
        y = float(self._arp_step_z1_sls[idx].get())
        self.synth.set_step_latent(idx, x, y)

    def _grab_step_pos(self, idx: int):
        """Capture the current XY pad position into step idx."""
        x = self._latent_x
        y = self._latent_y
        self._arp_step_z0_sls[idx].set(x)
        self._arp_step_z1_sls[idx].set(y)
        self.synth.set_step_latent(idx, x, y)

    def _highlight_arp_step(self, active_idx: int):
        """Bold the active step label; dim all others."""
        for i, lbl in enumerate(self._arp_step_labels):
            if i == active_idx:
                lbl.config(font=("Monaco", 8, "bold"))
            else:
                lbl.config(font=FONT_TINY)

    # ------------------------------------------------------------------
    # I/O panel
    # ------------------------------------------------------------------

    def _build_io_panel(self, parent):
        _, body = self._make_collapsible_panel(parent, "◆ I/O", pady=0)

        inner = tk.Frame(body, bg=MAC_BG)
        inner.pack(padx=6, pady=(0, 4), fill="x")

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
        _gain_cc_btn = tk.Button(gain_row, text="CC", font=FONT_TINY,
                                  fg=MAC_BLACK, bg=MAC_BG,
                                  relief="raised", bd=2, width=3,
                                  activebackground=MAC_HILIGHT)
        _gain_cc_btn.config(
            command=lambda b=_gain_cc_btn: self._start_learn("gain", b))
        _gain_cc_btn.pack(side="left", padx=(2, 0))
        self._cc_buttons["gain"] = _gain_cc_btn

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

        # Keyboard row
        kbd_row = tk.Frame(inner, bg=MAC_BG)
        kbd_row.pack(fill="x", pady=(4, 0))
        tk.Label(kbd_row, text="Kbd:", font=FONT_TINY,
                 fg=MAC_BLACK, bg=MAC_BG, width=9, anchor="w").pack(side="left")
        self._kbd_btn = mac_button(kbd_row, "KBD: OFF", self._toggle_kbd, width=8)
        self._kbd_btn.pack(side="left")
        self._kb_info_var = tk.StringVar(value=f"Oct:{KB_DEFAULT_OCTAVE}  Vel:{KB_DEFAULT_VELOCITY}")
        tk.Label(kbd_row, textvariable=self._kb_info_var,
                 font=FONT_TINY, fg=MAC_SHADOW, bg=MAC_BG).pack(side="left", padx=(6, 0))

        # CC clear row
        cc_clear_row = tk.Frame(inner, bg=MAC_BG)
        cc_clear_row.pack(fill="x", pady=(6, 0))
        mac_button(cc_clear_row, "CLEAR CC", self._clear_cc_map).pack(side="left")
        tk.Label(cc_clear_row, text=" clears all MIDI CC assignments",
                 font=FONT_TINY, fg=MAC_SHADOW, bg=MAC_BG).pack(side="left")

        # Apply defaults
        self._on_gain(80)
        self._refresh_midi_ports()

    # ------------------------------------------------------------------
    # Keyboard piano input
    # ------------------------------------------------------------------

    def _setup_keyboard_input(self):
        self.root.bind("<KeyPress>",     self._on_kb_press)
        self.root.bind("<KeyRelease>",   self._on_kb_release)
        self.root.bind_all("<Button-1>", self._refocus_root)
        self.root.focus_set()

    def _refocus_root(self, event):
        self.root.after(1, self.root.focus_set)

    def _toggle_kbd(self):
        self._kb_active = not self._kb_active
        self._kbd_btn.config(text="KBD: ON " if self._kb_active else "KBD: OFF")
        if not self._kb_active:
            self._release_all_kb_notes()

    def _on_kb_press(self, event):
        keysym = event.keysym.lower()

        # M key always toggles regardless of active state
        if keysym == "m":
            self._toggle_kbd()
            return

        if not self._kb_active:
            return

        # Octave shift
        if keysym == "z":
            if self._kb_octave > 0:
                self._release_all_kb_notes()
                self._kb_octave -= 1
                self._update_kb_display()
            return
        if keysym == "x":
            if self._kb_octave < 7:
                self._release_all_kb_notes()
                self._kb_octave += 1
                self._update_kb_display()
            return

        # Velocity
        if keysym == "c":
            self._kb_velocity = max(10, self._kb_velocity - KB_VELOCITY_STEP)
            self._update_kb_display()
            return
        if keysym == "v":
            self._kb_velocity = min(127, self._kb_velocity + KB_VELOCITY_STEP)
            self._update_kb_display()
            return

        # Note on
        if keysym in KB_NOTE_MAP and keysym not in self._kb_held:
            self._kb_held.add(keysym)
            midi = min(127, 12 * (self._kb_octave + 1) + KB_NOTE_MAP[keysym])
            self.synth.note_on(midi, self._kb_velocity)

    def _on_kb_release(self, event):
        keysym = event.keysym.lower()
        if keysym in self._kb_held:
            self._kb_held.discard(keysym)
            midi = min(127, 12 * (self._kb_octave + 1) + KB_NOTE_MAP[keysym])
            self.synth.note_off(midi)

    def _release_all_kb_notes(self):
        for keysym in list(self._kb_held):
            midi = min(127, 12 * (self._kb_octave + 1) + KB_NOTE_MAP[keysym])
            self.synth.note_off(midi)
        self._kb_held.clear()

    def _update_kb_display(self):
        self._kb_info_var.set(f"Oct:{self._kb_octave}  Vel:{self._kb_velocity}")

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

        # Update waveform name at 1/5 rate (10 Hz is plenty for visual)
        if self._lfo_tick % 5 == 0:
            self._update_wave_name(waveform)

    def _decode_and_update(self, x, y):
        waveform = self.synth.set_latent(x, y)
        self.root.after(0, self._update_wave_name, waveform)

    # ------------------------------------------------------------------
    # Waveform display / oscilloscope
    # ------------------------------------------------------------------

    def _osc_timer(self):
        """30 Hz timer: oscilloscope when audio is live, decoded wave otherwise."""
        data = self.synth.get_oscilloscope_data()
        rms  = float(np.sqrt(np.mean(data ** 2)))
        if rms > 0.001:
            self._update_waveform_display(data, auto_scale=True)
        else:
            self._update_waveform_display(self.synth._waveform_to)
        self.root.after(33, self._osc_timer)

    def _waveform_descriptor(self, waveform: np.ndarray) -> str:
        """Two-word spectral descriptor for the decoded waveform."""
        fft = np.abs(np.fft.rfft(waveform))
        fft[0] = 0.0  # strip DC
        total = fft.sum()
        if total < 1e-6:
            return ""
        freqs    = np.arange(len(fft), dtype=np.float32)
        centroid = float((freqs * fft).sum() / total)
        brightness = centroid / len(fft)  # 0–1

        fund  = fft[1] if len(fft) > 1 else 0.0
        richness = 1.0 - fund / total  # 0 = pure sine, 1 = no fundamental

        b = ("Dark" if brightness < 0.08
             else "Warm" if brightness < 0.22
             else "Mid"  if brightness < 0.40
             else "Bright")
        r = ("Pure"    if richness < 0.15
             else "Mild"    if richness < 0.40
             else "Rich"    if richness < 0.70
             else "Complex")
        return f"{b} · {r}"

    def _try_load_latent_index(self):
        if not os.path.exists(LATENT_INDEX_PATH):
            return
        try:
            data = np.load(LATENT_INDEX_PATH, allow_pickle=True)
            self._wave_tree      = KDTree(data["mus"])
            self._wave_filenames = data["filenames"].tolist()
            print(f"Latent index: {len(self._wave_filenames):,} waveforms")
        except Exception as e:
            print(f"Latent index load error: {e}")

    @staticmethod
    def _format_wave_label(filename: str) -> str:
        """
        'AKWF--Akai-MPC/AKWF_0001/AKWF_0001.WAV'     → 'Akai MPC / AKWF_0001'
        'AKWF--SonicWare--Smpltrek/…/AKWF_1939.wav'  → 'SonicWare · Smpltrek / AKWF_1939'
        'AKWF_clavinet_0001.WAV'                       → 'AKWF_clavinet_0001'
        """
        parts = filename.replace("\\", "/").split("/")
        stem  = os.path.splitext(parts[-1])[0]
        if len(parts) >= 2:
            cat = parts[0]
            if cat.upper().startswith("AKWF--"):
                # strip prefix; replace inner '--' with ' · ' before '-' → ' '
                cat = cat[6:].replace("--", " · ").replace("-", " ")
            elif cat.upper().startswith("AKWF_"):
                cat = cat[5:].replace("_", " ").title()
            return f"{cat} / {stem}"
        return stem

    def _update_wave_name(self, waveform: np.ndarray):
        if self._wave_tree is not None:
            _, idx = self._wave_tree.query([self._latent_x, self._latent_y])
            name = self._format_wave_label(self._wave_filenames[idx])
        else:
            name = self._waveform_descriptor(waveform)
        self._wave_name_var.set(name)

    def _update_waveform_display(self, waveform: np.ndarray, auto_scale: bool = False):
        w, h   = WAVE_WIDTH, WAVE_HEIGHT
        margin = 6
        n      = len(waveform)
        step   = n / (w - 2 * margin)

        # For oscilloscope: normalise to peak so quiet signals still fill the view
        if auto_scale:
            peak = float(np.max(np.abs(waveform)))
            scale = 1.0 / peak if peak > 1e-4 else 1.0
        else:
            scale = 1.0

        points = []
        for px in range(w - 2 * margin):
            val = float(waveform[int(px * step) % n]) * scale
            val = max(-1.0, min(1.0, val))
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

    def _refresh_midi_ports(self, preferred_name: str | None = None):
        ports = self.midi.list_ports()
        menu  = self._midi_menu["menu"]
        menu.delete(0, "end")
        if ports:
            for dev_id, name in ports:
                menu.add_command(
                    label=name,
                    command=lambda d=dev_id, n=name: self._select_midi_port(d, n))
            # Pick the preferred port if available, otherwise the first one
            match = next(((d, n) for d, n in ports if n == preferred_name), None)
            dev_id, name = match if match else ports[0]
            self._select_midi_port(dev_id, name, save=False)
        else:
            self._midi_var.set("(no ports)")
            menu.add_command(label="(no ports)", command=lambda: None)

    def _select_midi_port(self, dev_id: int, name: str, save: bool = True):
        """Open a MIDI port and optionally persist the choice."""
        self._midi_var.set(name)
        self.midi.open_port(dev_id)
        if save:
            self._save_settings()

    # ------------------------------------------------------------------
    # MIDI Learn
    # ------------------------------------------------------------------

    def _start_learn(self, key: str, btn: tk.Button):
        """Begin or cancel MIDI learn for the given parameter key."""
        # Cancel if same button clicked while already learning
        if (self._learn_target is not None
                and self._learn_target["key"] == key):
            self.midi._learn_callback = None
            prev_text = self._learn_target.get("prev_text", "CC")
            self._learn_target = None
            btn.config(bg=MAC_BG, text=prev_text)
            return

        # Cancel any previous learn
        if self._learn_target is not None:
            old_btn  = self._learn_target["btn"]
            old_text = self._learn_target.get("prev_text", "CC")
            old_btn.config(bg=MAC_BG, text=old_text)
            self.midi._learn_callback = None

        prev_text = btn.cget("text")
        self._learn_target = {"key": key, "btn": btn, "prev_text": prev_text}
        btn.config(bg="#ffff99", text="...")
        self.midi._learn_callback = lambda cc: self.root.after(
            0, self._finish_learn, cc)

    def _finish_learn(self, cc_num: int):
        """Bind cc_num to the pending learn target and update UI."""
        if self._learn_target is None:
            return

        key = self._learn_target["key"]
        btn = self._learn_target["btn"]
        self._learn_target = None

        # Remove any prior binding for this CC number
        self.midi.cc_map.pop(cc_num, None)

        mapping = self._make_cc_mapping(key)
        if mapping is None:
            btn.config(bg=MAC_BG, text="CC")
            return

        self.midi.cc_map[cc_num] = mapping
        btn.config(bg=MAC_BG, text=str(cc_num))

        # Brief green flash to confirm
        btn.config(bg="#99ff99")
        self.root.after(400, lambda: btn.config(bg=MAC_BG))

        self._save_cc_map()

    def _make_cc_mapping(self, key: str):
        """Build a CCMapping for the given registry key, or None if unknown."""
        root = self.root
        defs = {
            "latent_x": (
                "Latent X", LATENT_MIN, LATENT_MAX,
                lambda v: root.after(
                    0, lambda: self._move_to_latent(v, self._latent_y)),
            ),
            "latent_y": (
                "Latent Y", LATENT_MIN, LATENT_MAX,
                lambda v: root.after(
                    0, lambda: self._move_to_latent(self._latent_x, v)),
            ),
            "gain": (
                "Gain", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._gain_sl.set(v), self._on_gain(v))),
            ),
            "cutoff": (
                "Cutoff", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._cutoff_sl.set(v), self._on_cutoff(v))),
            ),
            "resonance": (
                "Resonance", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._res_sl.set(v), self._on_resonance(v))),
            ),
            "env_amount": (
                "Env Amt", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._env_amt_sl.set(v), self._on_env_amount(v))),
            ),
            "attack": (
                "Attack", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._atk.set(v), self._apply_adsr())),
            ),
            "decay": (
                "Decay", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._dec.set(v), self._apply_adsr())),
            ),
            "sustain": (
                "Sustain", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._sus.set(v), self._apply_adsr())),
            ),
            "release": (
                "Release", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._rel.set(v), self._apply_adsr())),
            ),
            "glide": (
                "Glide", 0, 1000,
                lambda v: root.after(
                    0, lambda: (self._glide_sl.set(v), self._on_glide(v))),
            ),
            "lfo_rate": (
                "LFO Rate", 1, 100,
                lambda v: root.after(
                    0, lambda: (self._lfo_rate_sl.set(v), self._on_lfo_rate(v))),
            ),
            "lfo_depth": (
                "LFO Depth", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._lfo_depth_sl.set(v), self._on_lfo_depth(v))),
            ),
            "z0": (
                "Z0", LATENT_MIN, LATENT_MAX,
                lambda v: root.after(
                    0, lambda: (self._z0_sl.set(v), self._on_z0_scan(v))),
            ),
            "z1": (
                "Z1", LATENT_MIN, LATENT_MAX,
                lambda v: root.after(
                    0, lambda: (self._z1_sl.set(v), self._on_z1_scan(v))),
            ),
            "f_attack": (
                "F.Attack", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._fatk.set(v), self._apply_filter_adsr())),
            ),
            "f_decay": (
                "F.Decay", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._fdec.set(v), self._apply_filter_adsr())),
            ),
            "f_sustain": (
                "F.Sustain", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._fsus.set(v), self._apply_filter_adsr())),
            ),
            "f_release": (
                "F.Release", 0, 100,
                lambda v: root.after(
                    0, lambda: (self._frel.set(v), self._apply_filter_adsr())),
            ),
        }
        if key not in defs:
            return None
        label, min_val, max_val, setter = defs[key]
        return CCMapping(key=key, label=label,
                         min_val=min_val, max_val=max_val, setter=setter)

    def _clear_cc_map(self):
        """Remove all CC assignments and reset button labels."""
        self.midi.cc_map.clear()
        self.midi._learn_callback = None
        self._learn_target = None
        for key, btn in self._cc_buttons.items():
            btn.config(bg=MAC_BG, text="CC" if key not in ("latent_x", "latent_y")
                       else ("CC X" if key == "latent_x" else "CC Y"))
        self._save_cc_map()

    # ------------------------------------------------------------------
    # CC persistence
    # ------------------------------------------------------------------

    def _save_cc_map(self):
        """Write CC assignments to disk as {cc_num_str: key}."""
        data = {str(cc): m.key for cc, m in self.midi.cc_map.items()}
        try:
            with open(CC_MAP_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"CC map save error: {e}")

    def _load_cc_map(self):
        """Restore CC assignments from disk and update button labels."""
        if not os.path.exists(CC_MAP_PATH):
            return
        try:
            with open(CC_MAP_PATH) as f:
                data = json.load(f)
        except Exception as e:
            print(f"CC map load error: {e}")
            return
        for cc_str, key in data.items():
            try:
                cc_num = int(cc_str)
            except ValueError:
                continue
            mapping = self._make_cc_mapping(key)
            if mapping is None:
                continue
            self.midi.cc_map[cc_num] = mapping
            btn = self._cc_buttons.get(key)
            if btn:
                btn.config(text=str(cc_num))

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _on_audio_select(self, selection):
        device_idx = int(selection.split(":")[0])
        self.synth.stop_audio()
        self.synth.start_audio(device=device_idx)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _save_settings(self):
        data = {"midi_port": self._midi_var.get()}
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Settings save error: {e}")

    def _load_settings(self):
        if not os.path.exists(SETTINGS_PATH):
            return
        try:
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Settings load error: {e}")
            return
        saved_port = data.get("midi_port")
        if saved_port:
            self._refresh_midi_ports(preferred_name=saved_port)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self._save_cc_map()
        self._save_settings()
        self.lfo.stop()
        self.synth.arp.stop()
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
