# Neural Network Inference Optimization

Comparing neural network architectures for image classification and measuring how much inference performance can be gained through optimization (quantization, pruning, graph compilation and optimized inference backends), and what it costs in accuracy.

> **Status:** 🚧 Work in progress (week 1 of 12). Results, graphs and conclusions are added as each phase is completed.

---

## 1. Project overview

**Goal:** For each architecture and optimization technique, measure:

| Metric | Unit | How |
| --- | --- | --- |
| Accuracy (top-1), before and after optimization | % | Full CIFAR-10 test set (10,000 images) |
| Inference latency | ms | Median and p95 over 300 runs, after warm-up |
| Throughput | images/s | Batch size 1 and 32 |
| RAM usage | MB | Peak RSS (`psutil`) |
| GPU memory usage | MB | `torch.cuda.max_memory_allocated` / NVML |
| Model size | MB | File size on disk |

The final result is a **trade-off analysis** (Pareto front) between accuracy and speed: *for a given latency budget, which architecture and optimization give the highest accuracy?*

**Research questions**

1. How much faster do models become with FP16, INT8, pruning and optimized backends?
2. How much accuracy is lost for each technique?
3. Do different architectures (classic CNN, mobile CNN) benefit differently from the same optimization?
4. When does an optimization look good on paper (fewer parameters) without making inference faster?

### Project plan (12 weeks)

| Week | Phase | Milestone | Report | Status |
| --- | --- | --- | --- | --- |
| 1–2 | PyTorch and baseline | Own CNN trained on CIFAR-10; stable benchmark script | Report 1: baseline and warm-up | 🚧 |
| 3–4 | Architectures | ResNet-18 and MobileNetV2 compared on accuracy and model size | Report 2: architectures | ⬜ |
| 5–6 | Quantization | FP32 vs FP16 (GPU) and FP32 vs INT8 (CPU) | Report 3: quantization | ⬜ |
| 7–8 | Pruning | Structured pruning + fine-tuning evaluated | Report 4: pruning | ⬜ |
| 9–10 | ONNX and CUDA | PyTorch, `torch.compile`, ONNX Runtime and TensorRT compared | Report 5: backends | ⬜ |
| 11 | Docker and reproducibility | Project runs on another machine from this README | – | ⬜ |
| 12 | GitHub and presentation | Results, graphs, final report and demo published | Final report | ⬜ |

---

## 2. Dataset and architectures

**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), 10 classes, 60,000 colour images (32×32).

| Split | Images | Used for |
| --- | --- | --- |
| Train | 45,000 | Training; INT8 calibration data |
| Validation | 5,000 | Selecting the best checkpoint |
| Test | 10,000 | All reported accuracy numbers |

**Architectures**

| Model | Source | Parameters (approx.) | Input | Why it is included |
| --- | --- | --- | --- | --- |
| SimpleCNN (own) | `src/models.py` | 0.3 M | 32×32 | Baseline built from scratch |
| ResNet-18 | torchvision | 11.7 M | 224×224 | Classic CNN with residual connections |
| MobileNetV2 | torchvision | 3.5 M | 224×224 | Designed for mobile: depthwise separable convolutions |
| *(optional)* ResNet-50, MobileNetV3, EfficientNet-B0, DeiT-Tiny | torchvision / timm | 5–26 M | 224×224 | Extensions if time allows |

---

## 3. Training configuration

> TODO: fill in when training is finished.

| Setting | SimpleCNN | ResNet-18 / MobileNetV2 |
| --- | --- | --- |
| Initialization | Random | ImageNet pre-trained, new 10-class head |
| Input resolution | 32×32 | 224×224 |
| Optimizer | | AdamW |
| Learning rate | | 1e-4, cosine schedule |
| Epochs | | 5 |
| Batch size | | 128 |
| Augmentation | | Random horizontal flip |
| Seed | 42 | 42 |
| Trained on | | Google Colab, Tesla T4 |

---

## 4. Optimization methods

| Method | Device | What it does | Expected effect |
| --- | --- | --- | --- |
| FP16 | GPU | Weights and activations in 16-bit floats; uses Tensor Cores | Lower latency and memory, almost no accuracy loss |
| `torch.compile` | GPU | Compiles the model graph and fuses operations | Lower latency |
| ONNX Runtime | CPU + GPU | Exports the model to ONNX and runs it in an optimized runtime | Lower latency |
| INT8 static quantization | CPU | 8-bit integers, calibrated on training data | ~4× smaller model, faster on CPU, small accuracy loss |
| Unstructured pruning | CPU + GPU | Sets the smallest weights to zero | Fewer non-zero weights, but **no** speed-up (tensors are still dense) |
| Structured pruning | CPU + GPU | Removes whole channels, then fine-tunes | Smaller and faster model |
| TensorRT FP16 | GPU | NVIDIA's inference engine: layer fusion and kernel selection | Largest GPU speed-up |

### Measurement methodology

Correct benchmarking matters more than any single optimization. All measurements follow these rules:

- **Warm-up before measuring.** The first runs include one-time costs (CUDA context, kernel loading, cuDNN autotuning, memory allocation, GPU clock ramp-up, compilation). 50 warm-up iterations are run and discarded.
- **Synchronize the GPU.** GPU calls are asynchronous, so `torch.cuda.synchronize()` (or CUDA events) is used before and after every timed call. Without it, only the kernel launch time is measured.
- **Batch size 1 and 32.** Batch 1 represents real-time latency; batch 32 represents throughput.
- **Inference only.** Data is loaded to the device before timing starts; data loading is never part of latency or throughput.
- **CPU and GPU reported separately.** Speed-up is always computed against the baseline on the same device.
- **One process per variant,** so memory measurements and caches do not leak between runs.

---

## 5. Benchmark results

> TODO: generated from `results/results.csv` by `src/analyze.py`.

### GPU (Tesla T4, Google Colab)

| Model | Variant | Accuracy (%) | Latency b=1 (ms) | Latency b=32 (ms) | Throughput b=32 (img/s) | GPU mem (MB) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

### CPU (Intel Core i5-8257U, 4 threads)

| Model | Variant | Accuracy (%) | Latency b=1 (ms) | Latency b=32 (ms) | Throughput b=32 (img/s) | RAM (MB) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

---

## 6. Performance graphs

> TODO: graphs are saved in `figures/`.

- Accuracy vs latency with Pareto front (CPU and GPU separately)
- Accuracy before and after optimization
- Speed-up per technique and architecture
- Throughput vs batch size
- Model size per variant
- Memory usage
- Warm-up effect: latency per iteration from the first call

<!-- Example: ![Accuracy vs latency (GPU)](figures/pareto_gpu.png) -->

---

## 7. Hardware specifications

| | CPU measurements | GPU measurements |
| --- | --- | --- |
| Machine | MacBook Pro (Intel) | Google Colab |
| CPU | Intel Core i5-8257U @ 1.40 GHz, 4 cores / 8 threads | (see `results/hardware_colab.txt`) |
| RAM | 8 GB | (see `results/hardware_colab.txt`) |
| GPU | – | NVIDIA Tesla T4, 16 GB |
| OS | macOS 15.7.8 | Linux (Colab) |
| Python | 3.11 | (Colab default) |
| PyTorch | 2.2.2 | 2.11.0 + CUDA 13.0 |
| ONNX Runtime | 1.23.2 | TODO |

Full details: `results/hardware_mac.txt` and `results/hardware_colab_*.txt`.

---

## 8. Conclusions and limitations

> TODO: written in week 12.

**Known limitations (so far)**

- CPU and GPU results come from different machines and are never compared directly against each other.
- Colab's free tier may assign different GPUs between sessions; only results from the same GPU type are compared.
- PyTorch versions differ between the Mac (2.2.2, last release for Intel Macs) and Colab (2.11).
- The Intel CPU lacks AVX-512 VNNI, so INT8 speed-ups are expected to be smaller than on newer CPUs.
- Laptop CPU clock speed varies with temperature and power mode; measurements are repeated 3–5 times.

---

## Repository structure

```text
nn-optimization-project/
├── configs/            # one config per experiment
├── data/               # CIFAR-10 (downloaded automatically, not in Git)
├── models/             # saved weights: .pt, .onnx, .engine (not in Git)
├── src/
│   ├── utils.py        # seed, device selection, synchronize
│   ├── check_setup.py  # sanity check of the environment
│   ├── train.py        # training (phase 1–2)
│   ├── benchmark.py    # latency, throughput, memory, size
│   ├── optimize/       # quantize.py, prune.py, export.py
│   └── analyze.py      # tables and graphs
├── results/            # results.csv, hardware_*.txt
├── figures/            # graphs
├── report/             # LaTeX reports
└── requirements.txt
```

## How to reproduce

### Local (Mac, CPU)

```bash
git clone https://github.com/alicewesterberg/nn-optimization-project.git
cd nn-optimization-project
conda create -n nnopt python=3.11 -y
conda activate nnopt
pip install -r requirements.txt
python -m src.check_setup
```

### Google Colab (GPU)

1. Runtime → Change runtime type → **T4 GPU**.
2. Run:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/alicewesterberg/nn-optimization-project.git
%cd nn-optimization-project
!pip install -q timm onnx onnxruntime-gpu pynvml torch-pruning
!python -m src.check_setup
```

### Running the experiments

> TODO: add commands as the scripts are written, e.g.

```bash
python -m src.train --model resnet18 --epochs 5
python -m src.benchmark --model resnet18 --variant fp32_eager --batch-sizes 1,32
python -m src.analyze
```

### Docker

> TODO: added in week 11.

---

## Author

Alice Westerberg · MSc Computer Science and Engineering (AI/ML), Lund University (LTH)
[GitHub](https://github.com/alicewesterberg) · [LinkedIn](https://linkedin.com/in/alice-westerberg)
