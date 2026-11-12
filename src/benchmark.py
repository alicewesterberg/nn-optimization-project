import argparse, csv, os, statistics, time
import numpy as np
import torch

from src.models import build_model
from src.utils import set_seed, synchronize


def load_model(name, device):
    model = build_model(name)
    model.load_state_dict(torch.load(f"models/{name}_fp32.pt", map_location="cpu"))
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
    p.add_argument("--device", default="cpu")        # always explicit, never get_device()
    p.add_argument("--threads", type=int, default=5)    # = the M5 Pro's 5 Super cores (PyTorch default)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--warmup-study", action="store_true")
    args = p.parse_args()
    if args.warmup_study:
        warmup_study(args)