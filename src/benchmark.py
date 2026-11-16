import argparse, csv, os, platform, resource, statistics, time
from datetime import datetime
import numpy as np
import torch

from src.data import get_test_loader
from src.models import build_model
from src.utils import set_seed, synchronize

N_WARMUP = 50    # from the warm-up study: stable after <= 35 calls on CPU and MPS
N_ITERS = 300    # timed calls per batch size


def load_model(name, device, variant="fp32"):
    model = build_model(name)
    model.load_state_dict(torch.load(f"models/{name}_{variant}.pt", map_location="cpu"))
    return model.to(device).eval()


def time_calls(model, x, n_iters, device):
    """Latency in ms for each call, in order. No warm-up here."""
    times = []
    with torch.inference_mode():
        for _ in range(n_iters):
            synchronize(device)              # wait until the GPU is idle
            t0 = time.perf_counter()
            model(x)
            synchronize(device)              # wait until this call is done
            times.append((time.perf_counter() - t0) * 1000)
    return times


def peak_rss_mb():
    """Peak resident memory of this process so far (whole process, incl. PyTorch itself)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if platform.system() == "Darwin" else peak / 2**10   # macOS: bytes, Linux: KB


def gpu_mem_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20       # peak allocated by PyTorch
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20      # current, unified memory
    return float("nan")


def evaluate(model, device):
    """Top-1 accuracy (%) on the full CIFAR-10 test set (10,000 images)."""
    correct = total = 0
    with torch.inference_mode():
        for x, y in get_test_loader():
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def benchmark(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    set_seed(42)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = load_model(args.model, device, args.variant)
    size_mb = os.path.getsize(f"models/{args.model}_{args.variant}.pt") / 2**20

    # 1) Latency first, so the test set is not loaded yet when peak RSS is read.
    rows = []
    for b in args.batch_sizes:
        x = torch.randn(b, 3, 32, 32, device=device)
        time_calls(model, x, N_WARMUP, device)                  # warm-up, thrown away
        t = np.array(time_calls(model, x, N_ITERS, device))
        med = float(np.median(t))
        rows.append({
            "batch_size": b,
            "lat_median_ms": med,
            "lat_p95_ms": float(np.percentile(t, 95)),
            "lat_mean_ms": float(t.mean()),
            "lat_std_ms": float(t.std()),
            "throughput_ips": b / (med / 1000),
        })
    rss = peak_rss_mb()
    gpu = gpu_mem_mb(device)

    # 2) Accuracy (same for every batch size).
    acc = evaluate(model, device)

    common = {
        "model": args.model, "variant": args.variant, "device": args.device,
        "run": args.run, "accuracy": acc, "rss_peak_mb": rss, "gpu_mem_mb": gpu,
        "size_mb": size_mb, "threads": args.threads, "warmup": N_WARMUP,
        "iters": N_ITERS, "torch": torch.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/results_{args.device}.csv"
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fields = ["model", "variant", "device", "batch_size", "run", "accuracy",
                  "lat_median_ms", "lat_p95_ms", "lat_mean_ms", "lat_std_ms",
                  "throughput_ips", "rss_peak_mb", "gpu_mem_mb", "size_mb",
                  "threads", "warmup", "iters", "torch", "timestamp"]
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({**common, **r})

    print(f"{args.model} {args.variant} {args.device} run {args.run} | acc {acc:.2f} % | "
          f"RSS {rss:.0f} MB | GPU {gpu:.0f} MB | {size_mb:.2f} MB")
    for r in rows:
        print(f"  b={r['batch_size']:>2}: median {r['lat_median_ms']:.3f} ms | "
              f"p95 {r['lat_p95_ms']:.3f} ms | {r['throughput_ips']:.0f} img/s")
    print(f"-> {path}")


def warmup_study(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    set_seed(42)
    model = load_model(args.model, device)
    x = torch.randn(args.batch_size, 3, 32, 32, device=device)

    times = time_calls(model, x, args.iters, device)   # from the very first call

    os.makedirs("results", exist_ok=True)
    path = (f"results/warmup_{args.model}_{args.device}"
            f"_b{args.batch_size}_run{args.run}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "latency_ms"])
        w.writerows(enumerate(times, start=1))
    print(f"first {times[0]:.2f} ms | steady {statistics.median(times[-100:]):.2f} ms -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="simple_cnn")
    p.add_argument("--variant", default="fp32")        # later: fp16, int8, pruned ...
    p.add_argument("--device", default="cpu")        # always explicit, never get_device()
    p.add_argument("--threads", type=int, default=5)    # = the M5 Pro's 5 Super cores (PyTorch default)
    p.add_argument("--batch-size", type=int, default=1)                  # warm-up study only
    p.add_argument("--batch-sizes", default="1,32",
                   type=lambda s: [int(b) for b in s.split(",")])        # benchmark mode
    p.add_argument("--iters", type=int, default=300)                     # warm-up study only
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--warmup-study", action="store_true")
    args = p.parse_args()
    if args.warmup_study:
        warmup_study(args)
    else:
        benchmark(args)
