"""Transparent dense-model planning estimates, not measured throughput."""
import argparse
import json
import math


def estimate(parameters, tokens, sustained_tflops, gpus, hourly_rate, multiplier=1.0):
    values = [parameters, tokens, sustained_tflops, gpus, multiplier]
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Parameters, tokens, throughput, GPU count and multiplier must be positive.")
    if not math.isfinite(hourly_rate) or hourly_rate < 0:
        raise ValueError("Hourly rate must be finite and nonnegative.")
    if int(gpus) != gpus:
        raise ValueError("GPU count must be an integer.")
    flops = 6 * parameters * tokens
    gpu_hours = flops * multiplier / (sustained_tflops * 1e12 * 3600)
    return {
        "parameters": parameters,
        "text_tokens": tokens,
        "approximate_text_training_flops": flops,
        "sustained_tflops_per_gpu_assumption": sustained_tflops,
        "additional_compute_multiplier_assumption": multiplier,
        "gpus": gpus,
        "gpu_hours": gpu_hours,
        "wall_days": gpu_hours / gpus / 24,
        "price_per_gpu_hour_usd": hourly_rate,
        "gpu_cost_usd": gpu_hours * hourly_rate,
        "notes": "6ND dense text approximation. Sustained throughput includes scaling losses. "
        "Multiplier is a planning allowance, not an image/RL workload measurement. "
        "Excludes storage, CPU preprocessing, data fees, human labeling, taxes, "
        "and teacher generation. No capability prediction.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters", type=int, default=800_000_000)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--sustained-tflops", type=float, default=150)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--hourly-rate", type=float, default=3.99)
    p.add_argument("--multiplier", type=float, default=1.5)
    args = p.parse_args()
    try:
        result = estimate(args.parameters, args.tokens, args.sustained_tflops,
                          args.gpus, args.hourly_rate, args.multiplier)
    except ValueError as e:
        p.error(str(e))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
