"""
dataset.py — AKWF Single-Cycle Waveform Dataset

Loads Adventure Kid Waveforms (AKWF) .wav files, extracts one clean cycle
per file, resamples to a fixed length, and normalizes to [-1, 1].

Each item returned: Tensor of shape (1, WAVEFORM_LENGTH) — float32.
"""

import os
import math
import torch
import soundfile as sf
import torchaudio.functional
from torch.utils.data import Dataset
from typing import Optional

WAVEFORM_LENGTH = 2048  # fixed output length in samples


class AKWFDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cache_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            data_dir:   Path to folder containing AKWF .wav files.
                        Searched recursively, so subdirectories are fine.
            cache_path: Optional path to a .pt file. If it exists, tensors are
                        loaded from it instead of reprocessing. If it doesn't
                        exist yet, the processed tensors are saved there after
                        the first run.
            verbose:    Print progress and skip counts during loading.
        """
        self.data_dir = data_dir
        self.verbose = verbose

        # Load from cache if available
        if cache_path and os.path.exists(cache_path):
            if verbose:
                print(f"Loading preprocessed dataset from cache: {cache_path}")
            saved = torch.load(cache_path, weights_only=True)
            self.waveforms = saved["waveforms"]   # (N, 1, 2048)
            self.filenames = saved["filenames"]   # list[str]
            if verbose:
                print(f"Loaded {len(self.waveforms)} waveforms from cache.")
            return

        # Discover all .wav files
        wav_paths = self._find_wav_files(data_dir)
        if not wav_paths:
            raise FileNotFoundError(
                f"No .wav files found under '{data_dir}'. "
                "Download the AKWF dataset and place it there."
            )
        if verbose:
            print(f"Found {len(wav_paths)} .wav files. Processing...")

        # Process each file
        waveforms = []
        filenames = []
        skipped = 0

        for i, path in enumerate(wav_paths):
            tensor = self._process_file(path)
            if tensor is None:
                skipped += 1
                continue
            waveforms.append(tensor)
            filenames.append(os.path.relpath(path, data_dir))

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(wav_paths)} files...")

        if not waveforms:
            raise RuntimeError("All files were skipped. Check your data directory.")

        self.waveforms = torch.stack(waveforms)  # (N, 1, 2048)
        self.filenames = filenames

        if verbose:
            print(
                f"Done. {len(self.waveforms)} waveforms loaded, {skipped} skipped."
            )

        # Save cache if requested
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(
                {"waveforms": self.waveforms, "filenames": self.filenames},
                cache_path,
            )
            if verbose:
                print(f"Cache saved to: {cache_path}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns (1, 2048) float32 — no label, unsupervised
        return self.waveforms[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_wav_files(self, root: str) -> list[str]:
        """Recursively collect all .wav file paths under root, sorted."""
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(".wav"):
                    paths.append(os.path.join(dirpath, fname))
        return sorted(paths)

    def _process_file(self, path: str) -> Optional[torch.Tensor]:
        """
        Load a .wav file and return a (1, WAVEFORM_LENGTH) float32 tensor,
        or None if the file should be skipped.

        Steps:
          1. Load with torchaudio (any sample rate, any channel count)
          2. Mix to mono
          3. Find first upward zero crossing → clean loop start
          4. Resample the remaining samples to WAVEFORM_LENGTH
          5. Peak-normalize to [-1, 1]
        """
        try:
            data, sample_rate = sf.read(path, always_2d=True)  # (N, C) float64
        except Exception:
            return None

        # Mix to mono: average across channels → (N,), then convert to float32 tensor
        samples = torch.from_numpy(data.mean(axis=1)).float()  # (N,)

        # Skip files that are too short to be useful
        if samples.shape[0] < 16:
            return None

        # Find first upward zero crossing for a clean loop start point.
        # An upward crossing is where the signal goes from negative to non-negative.
        start = self._first_upward_zero_crossing(samples)

        # Trim to the clean cycle
        samples = samples[start:]

        # Skip if too few samples remain after trimming
        if samples.shape[0] < 16:
            return None

        # Resample to exactly WAVEFORM_LENGTH using torchaudio (anti-aliased)
        # Treat the trimmed samples as if recorded at `sample_rate`, and
        # resample to produce exactly WAVEFORM_LENGTH output samples.
        # We achieve a fixed output length by resampling from len(samples) → WAVEFORM_LENGTH,
        # using the ratio directly via orig_freq / new_freq.
        orig_len = samples.shape[0]
        samples_2d = samples.unsqueeze(0)  # (1, N) required by torchaudio resample

        # torchaudio.functional.resample expects integer freq ratios.
        # We use GCD reduction to keep numbers small and avoid int overflow.
        g = math.gcd(orig_len, WAVEFORM_LENGTH)
        orig_freq = orig_len // g
        new_freq = WAVEFORM_LENGTH // g

        resampled = torchaudio.functional.resample(
            samples_2d,
            orig_freq=orig_freq,
            new_freq=new_freq,
        )  # (1, WAVEFORM_LENGTH) approximately — trim/pad to be exact

        resampled = resampled[0]  # (M,) where M ≈ WAVEFORM_LENGTH

        # Ensure exact length (resample can be off by ±1 due to rounding)
        if resampled.shape[0] > WAVEFORM_LENGTH:
            resampled = resampled[:WAVEFORM_LENGTH]
        elif resampled.shape[0] < WAVEFORM_LENGTH:
            pad = WAVEFORM_LENGTH - resampled.shape[0]
            resampled = torch.nn.functional.pad(resampled, (0, pad))

        # Peak-normalize to [-1, 1]
        peak = resampled.abs().max()
        if peak < 1e-6:
            return None  # Silent / corrupted file — skip
        resampled = resampled / peak

        return resampled.unsqueeze(0).float()  # (1, 2048)

    @staticmethod
    def _first_upward_zero_crossing(samples: torch.Tensor) -> int:
        """
        Return the index of the first upward zero crossing in samples.

        An upward crossing: samples[i-1] < 0 and samples[i] >= 0.
        If none found, returns 0 (use the full waveform as-is).
        """
        for i in range(1, len(samples)):
            if samples[i - 1] < 0 and samples[i] >= 0:
                return i
        return 0


# ------------------------------------------------------------------
# Quick smoke test — run this file directly to validate your dataset
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/akwf"
    cache_path = "data/akwf_cache.pt"

    print(f"Loading dataset from: {data_dir}")
    dataset = AKWFDataset(data_dir=data_dir, cache_path=cache_path, verbose=True)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
    print(f"Sample dtype: {dataset[0].dtype}")
    print(f"Sample min/max: {dataset[0].min():.4f} / {dataset[0].max():.4f}")
    print(f"\nFirst 5 files:")
    for name in dataset.filenames[:5]:
        print(f"  {name}")
