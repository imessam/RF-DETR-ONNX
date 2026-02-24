# Benchmarks

The `benchmarks/` directory contains an automated performance measurement suite that evaluates both the Python and C++ implementations across CPU and GPU providers.

---

## How It Works

1. **Discovers models** — scans `models/` for all `.onnx` files
2. **Downloads models** — fetches from the GitHub release if no models are found
3. **Builds C++** — compiles the C++ binaries if needed
4. **Runs inference** — iterates over all models × all providers (CPU, CUDA, TensorRT)
5. **Generates report** — aggregates JSON results into `benchmarks/results/results.md`

---

## Running Benchmarks

```bash
cd benchmarks
./run_benchmarks.sh -n 10 -v
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n <int>` | `10` | Number of iterations per run |
| `-c <float>` | `2.0` | Cooldown period in seconds between runs |
| `-s <float>` | `0.1` | Sleep delay between per-image iterations |
| `-v` | Off | Enable verbose per-iteration output |
| `-u <url>` | *(GitHub release)* | Custom URL to download ONNX models |

---

## Example Output

Results are saved to `benchmarks/results/results.md`. The report shows:

- Model name and architecture
- Provider used (TensorRT / CUDA / CPU)
- Preprocess / ORT run / postprocess timings
- Average FPS

### Sample (RF-DETR Nano, 384×384)

| Provider | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | FPS |
|----------|----------------|--------------|-----------------|-----|
| TensorRT | 2.1 | 4.3 | 1.8 | ~120 |
| CUDA | 2.1 | 12.5 | 1.8 | ~57 |
| CPU | 2.1 | 95.0 | 1.8 | ~10 |

!!! note
    Results vary by hardware (CPU model, GPU generation, system load). The table above is an illustrative example from a laptop with an RTX 40-series GPU.

---

## Generating a Report Standalone

If you already have benchmark result JSON files and just want to regenerate the Markdown report:

```bash
cd benchmarks
python generate_report.py --results-dir results/ --output results/results.md
```

---

## Benchmarking Architecture

```mermaid
flowchart TD
    A[run_benchmarks.sh] --> B{Models found?}
    B -- No --> C[Download from GitHub release]
    B -- Yes --> D[Build C++ binaries]
    C --> D
    D --> E[Run Python benchmarks]
    D --> F[Run C++ benchmarks]
    E --> G[Save JSON results]
    F --> G
    G --> H[generate_report.py]
    H --> I[results/results.md]
```
