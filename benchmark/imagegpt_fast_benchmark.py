import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from transformers import ImageGPTImageProcessor, ImageGPTImageProcessorFast


# Default clusters (same as test fixtures) – 2 clusters for simplicity
DEFAULT_CLUSTERS = np.asarray(
    [
        [0.8866443634033203, 0.6618829369544983, 0.3891746401786804],
        [-0.6042559146881104, -0.02295008860528469, 0.5423797369003296],
    ]
)


def create_test_images(batch_size: int = 8, image_size: Tuple[int, int] = (64, 64), num_channels: int = 3):
    """Create a list of random PIL images for benchmarking."""
    images = []
    for _ in range(batch_size):
        img = np.random.randint(0, 255, (image_size[0], image_size[1], num_channels), dtype=np.uint8)
        images.append(Image.fromarray(img))
    return images


def _to_device_tensor(x: torch.Tensor, device: str) -> torch.Tensor:
    if device == "cpu":
        return x
    if device == "cuda" and torch.cuda.is_available():
        return x.to("cuda")
    return x


def benchmark_processor(
    processor,
    images,
    *,
    num_iterations: int = 10,
    device: str = "cpu",
    clusters: Optional[np.ndarray] = None,
):
    """Benchmark a single processor on a given device.

    Returns (times, outputs): times is a list[float] for each iteration; outputs is the last BatchFeature.
    """
    times: list[float] = []

    # Warmup (ignore failures quietly for robustness)
    for _ in range(3):
        try:
            _ = processor(images, clusters=clusters, return_tensors="pt")
        except Exception:
            return None, None

    # Actual timing loop
    for i in range(num_iterations):
        start = time.time()
        try:
            outputs = processor(
                images,
                clusters=clusters,
                return_tensors="pt",
                **({"device": device} if isinstance(processor, ImageGPTImageProcessorFast) else {}),
            )

            # Ensure device sync for fair timing
            if device == "cuda" and torch.cuda.is_available():
                # Move output tensors to device and synchronize
                if "pixel_values" in outputs:
                    outputs["pixel_values"] = _to_device_tensor(outputs["pixel_values"], device)
                if "input_ids" in outputs:
                    outputs["input_ids"] = _to_device_tensor(outputs["input_ids"], device)
                torch.cuda.synchronize()
        except Exception:
            return None, None
        end = time.time()
        times.append(end - start)

    return times, outputs


def verify_equivalence(slow_outputs, fast_outputs) -> bool:
    """Verify outputs are functionally equivalent within expected tolerances.

    ImageGPT with color quantization returns `input_ids` (integers). We'll compare:
    - shapes must match
    - values equality ratio should be very high; allow tiny mismatch due to rare tie/rounding
    """
    if slow_outputs is None or fast_outputs is None:
        return False

    # Prefer input_ids when present (color quantized path)
    if "input_ids" in slow_outputs and "input_ids" in fast_outputs:
        s = slow_outputs["input_ids"].to("cpu")
        f = fast_outputs["input_ids"].to("cpu")
        if s.shape != f.shape:
            return False
        equal = (s == f).float().mean().item()
        # Expect exact or near-exact match; allow a tiny tolerance
        return equal >= 0.999

    # Fallback to pixel_values if not quantizing
    if "pixel_values" in slow_outputs and "pixel_values" in fast_outputs:
        s = slow_outputs["pixel_values"].to(torch.float32).cpu()
        f = fast_outputs["pixel_values"].to(torch.float32).cpu()
        if s.shape != f.shape:
            return False
        return torch.allclose(s, f, rtol=1e-3, atol=1e-1)

    return False


def run_benchmark():
    print("🚀 ImageGPT Image Processor Benchmark (slow vs fast)")
    print("=" * 60)

    # Configurations
    batch_sizes = [1, 4, 8, 16]
    image_sizes = [(18, 18), (64, 64), (128, 128), (256, 256)]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    clusters = DEFAULT_CLUSTERS

    results: dict[str, dict[str, dict[str, float]]] = {}

    for device in devices:
        print(f"\n🖥️  Testing on device: {device}")
        print("-" * 30)
        results[device] = {}

        for batch_size in batch_sizes:
            for image_size in image_sizes:
                config_name = f"{batch_size}x{image_size[0]}x{image_size[1]}"
                print(f"\n🔧 Config: batch_size={batch_size}, image_size={image_size}")

                # Create data
                images = create_test_images(batch_size=batch_size, image_size=image_size)

                # Init processors
                try:
                    slow = ImageGPTImageProcessor(
                        clusters=clusters, size={"height": image_size[0], "width": image_size[1]}
                    )
                    fast = ImageGPTImageProcessorFast(
                        clusters=clusters, size={"height": image_size[0], "width": image_size[1]}
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize processors: {e}")
                    continue

                # Benchmark slow
                print("⏳ Benchmarking slow processor…")
                slow_times, slow_out = benchmark_processor(
                    slow, images, num_iterations=10, device=device, clusters=clusters
                )
                if slow_times is None:
                    print("❌ Slow processor benchmark failed; skipping config")
                    continue

                # Benchmark fast
                print("⚡ Benchmarking fast processor…")
                fast_times, fast_out = benchmark_processor(
                    fast, images, num_iterations=10, device=device, clusters=clusters
                )
                if fast_times is None:
                    print("❌ Fast processor benchmark failed; skipping config")
                    continue

                # Stats
                slow_mean = float(np.mean(slow_times))
                slow_std = float(np.std(slow_times))
                fast_mean = float(np.mean(fast_times))
                fast_std = float(np.std(fast_times))
                speedup = slow_mean / max(fast_mean, 1e-12)

                # Store
                results[device][config_name] = {
                    "slow_mean": slow_mean,
                    "slow_std": slow_std,
                    "fast_mean": fast_mean,
                    "fast_std": fast_std,
                    "speedup": speedup,
                }

                # Print
                print("📊 Results:")
                print(f"   Slow: {slow_mean:.4f}s ± {slow_std:.4f}s")
                print(f"   Fast: {fast_mean:.4f}s ± {fast_std:.4f}s")
                print(f"   Speedup: {speedup:.2f}x")

                # Verify equivalence
                try:
                    ok = verify_equivalence(slow_out, fast_out)
                    if ok:
                        print("✅ Output verification: PASSED")
                        print("   (shape ✓, values ✓)")
                    else:
                        print("❌ Output verification: FAILED (shapes or values differ)")
                except Exception as e:
                    print(f"⚠️  Output verification failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 60)

    for device, device_results in results.items():
        print(f"\n🖥️  {device.upper()}:")
        print("-" * 20)
        if not device_results:
            print("   No results available")
            continue
        speedups = [r["speedup"] for r in device_results.values()]
        print(f"   Average speedup: {np.mean(speedups):.2f}x")
        print(f"   Min speedup: {np.min(speedups):.2f}x")
        print(f"   Max speedup: {np.max(speedups):.2f}x")
        print("\n   Detailed results:")
        for cfg, stats in device_results.items():
            print(f"     {cfg}: {stats['speedup']:.2f}x speedup")

    # Visualization
    if HAS_MATPLOTLIB:
        try:
            create_visualization(results)
        except Exception as e:
            print(f"\n⚠️  Visualization failed: {e}")
    else:
        print("\n📊 Matplotlib not available, skipping visualization")


def create_visualization(results: dict):
    if not HAS_MATPLOTLIB:
        return

    for device, device_results in results.items():
        if not device_results:
            continue
        configs = list(device_results.keys())
        speedups = [device_results[c]["speedup"] for c in configs]
        avg_speedup = float(np.mean(speedups))

        all_configs = configs + ["Average"]
        all_speedups = speedups + [avg_speedup]

        colors = ["steelblue"] * len(configs) + ["orange"]
        alphas = [0.7] * len(configs) + [0.85]

        plt.figure(figsize=(12, 6))
        bars = []
        for i, (cfg, spd, color, alpha) in enumerate(zip(all_configs, all_speedups, colors, alphas)):
            bar = plt.bar(i, spd, color=color, alpha=alpha)
            bars.extend(bar)
        for bar, spd in zip(bars, all_speedups):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{spd:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.xlabel("Configuration (batch_size x height x width)")
        plt.ylabel("Speedup (times faster)")
        plt.title(f"ImageGPT Fast vs Slow Image Processor Speedup - {device.upper()}")
        plt.xticks(range(len(all_configs)), all_configs, rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        filename = f"imagegpt_benchmark_{device}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"   Saved plot: {filename}")
        plt.close()


if __name__ == "__main__":
    run_benchmark()
