"""
export.py — Export the trained VAE decoder to ONNX

Extracts only the Decoder from a training checkpoint and exports it as a
self-contained ONNX graph. After this, PyTorch is no longer needed at runtime.

Usage:
    python export/export.py --checkpoint model/checkpoints/best.pt
    python export/export.py --checkpoint model/checkpoints/best.pt --output export/decoder.onnx
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from vae import VAE

WAVEFORM_LENGTH = 2048


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_decoder(decoder, output_path: str, latent_dim: int = 2):
    """Export the decoder to ONNX with fixed input/output shapes."""

    decoder.eval()

    # Dummy input — shape and dtype must match exactly what runtime will send
    dummy_z = torch.zeros(1, latent_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        decoder,
        dummy_z,
        output_path,
        export_params=True,          # embed weights in the file
        opset_version=17,
        do_constant_folding=True,    # fold constant ops at export time (faster inference)
        input_names=["latent"],
        output_names=["waveform"],
        # No dynamic_axes — fixed shapes only for real-time safety
    )

    print(f"Exported to: {output_path}")
    print(f"File size:   {os.path.getsize(output_path) / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(decoder, onnx_path: str, latent_dim: int = 2, n_points: int = 10):
    """
    Compare PyTorch decoder output vs ONNX Runtime output on random latent points.
    Max absolute difference should be < 1e-4 (float32 rounding).
    """
    print(f"\nValidating against {n_points} random latent points...")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    decoder.eval()

    max_diff = 0.0

    for i in range(n_points):
        z_np = np.random.randn(1, latent_dim).astype(np.float32)
        z_pt = torch.from_numpy(z_np)

        # PyTorch
        with torch.no_grad():
            pt_out = decoder(z_pt).numpy()  # (1, 1, 2048)

        # ONNX Runtime
        ort_out = session.run(["waveform"], {"latent": z_np})[0]  # (1, 1, 2048)

        diff = np.abs(pt_out - ort_out).max()
        max_diff = max(max_diff, diff)

    status = "PASS" if max_diff < 1e-4 else "FAIL"
    print(f"Max absolute difference: {max_diff:.2e}  [{status}]")

    if status == "FAIL":
        print("WARNING: outputs differ beyond float32 tolerance — check export.")

    return max_diff


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(onnx_path: str, latent_dim: int = 2, n_runs: int = 1000):
    """
    Measure ONNX Runtime inference latency over n_runs calls.
    Target: mean < 1ms, p99 < 2ms.
    """
    print(f"\nBenchmarking ({n_runs} runs)...")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    z = np.zeros((1, latent_dim), dtype=np.float32)

    # Warmup
    for _ in range(20):
        session.run(["waveform"], {"latent": z})

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(["waveform"], {"latent": z})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    print(f"  Mean:  {latencies.mean():.3f} ms")
    print(f"  Median:{np.median(latencies):.3f} ms")
    print(f"  p99:   {np.percentile(latencies, 99):.3f} ms")
    print(f"  Min:   {latencies.min():.3f} ms")
    print(f"  Max:   {latencies.max():.3f} ms")

    target_met = latencies.mean() < 1.0
    print(f"\n  Target (<1ms mean): {'PASS' if target_met else 'FAIL — consider model pruning'}")

    return latencies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export VAE decoder to ONNX")
    parser.add_argument("--checkpoint", default="model/checkpoints/best.pt", help="Path to trained checkpoint")
    parser.add_argument("--output",     default="export/decoder.onnx",       help="Output ONNX path")
    parser.add_argument("--bench-runs", type=int, default=1000,              help="Number of benchmark iterations")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    latent_dim = ckpt["config"].get("latent_dim", 2)
    epoch      = ckpt["epoch"]
    val_loss   = ckpt["val_loss"]
    print(f"  Epoch {epoch}, val_loss={val_loss:.5f}, latent_dim={latent_dim}")

    # Build full VAE, load weights, extract decoder
    vae = VAE(latent_dim=latent_dim)
    vae.load_state_dict(ckpt["model"])
    decoder = vae.decoder
    decoder.eval()

    # Export
    export_decoder(decoder, args.output, latent_dim)

    # Validate ONNX model is well-formed
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    print("ONNX graph check: PASS")

    # Validate outputs match PyTorch
    validate(decoder, args.output, latent_dim)

    # Benchmark
    benchmark(args.output, latent_dim, args.bench_runs)

    print(f"\nDone. Decoder ready at: {args.output}")


if __name__ == "__main__":
    main()
