# PLATO Forge — GPU Benchmarking Room

Where CUDA experiments become tiles. Every kernel compile, every benchmark run, every optimization — logged and searchable.

## Purpose

This room accumulates GPU knowledge so that the next person asking "what's the fastest way to do X on Y hardware?" gets an instant, data-backed answer instead of running their own experiment.

## Two-Gear System

**Gear 1 (scripts):** Always-on benchmark runners. Compile kernels, measure time, log results.
**Gear 2 (agents):** Read the logs, spot patterns, create optimization tiles, fix clunk signals.

## Hardware Profiles

### Jetson Orin Nano 8GB (JC1)
- 1024 CUDA cores, 8GB unified RAM
- nvcc at /usr/local/cuda-12.6
- Max useful model: Qwen3-32B (4-bit)
- OOM threshold: ~6.5GB Python heap
- Good for: inference, small experiments, prototyping

### RTX 4050 (Forgemaster)
- 2560 CUDA cores, 6GB VRAM
- Good for: training sweeps, batch experiments, large parameter sweeps
- ~120 spells/hour (NoisyBoost, FM constant sweep)

## Seed Tiles

### Compilation
- **Q:** What's the fastest way to compile CUDA on Jetson?
  **A:** Use NVRTC (runtime compilation) for iteration speed. nvcc for production. NVRTC compiles in ~200ms vs nvcc's ~3s for small kernels.

- **Q:** Does C++ interop work on Jetson ARM64?
  **A:** Rust needs a real machine for heavy compilation. C11 compiles everywhere. Python OOMs at ~6.5GB.

- **Q:** How do I avoid OOM on 8GB unified RAM?
  **A:** Keep Python heap under 6.5GB. Use numpy arrays instead of Python lists. Free CUDA memory explicitly with `torch.cuda.empty_cache()`.

### Memory
- **Q:** How much VRAM does a 1M-agent simulation need?
  **A:** ~400MB for positions + velocities (float32). +200MB for perception grids. +50MB for DCS structures. Total: ~650MB fits comfortably.

- **Q:** What's the OOM boundary for agent counts on Jetson?
  **A:** ~4M agents with minimal state. ~2M with full perception grids. ~500K with neural network inference per agent.

### Performance
- **Q:** What's the baseline throughput for a 1K-agent simulation?
  **A:** ~60 FPS on Jetson Orin (1K agents, simple movement + food collection). Drops to ~30 FPS with DCS enabled.

- **Q:** Does batch processing help on Jetson?
  **A:** Minimal. The GPU is already saturated at 1K agents. Batch helps on RTX 4050 (15% token reduction per FM's measurements).

- **Q:** What's the fastest interop for tile I/O?
  **A:** JSON is fine for <10K tiles. Switch to SQLite WAL mode for concurrent reads above that.

## Benchmark Format

Every benchmark result becomes a tile:

```json
{
  "instruction": "What's the throughput for 10K-agent DCS simulation on RTX 4050?",
  "input": "hardware=rtx-4050, agents=10000, dcs=ring-buffer-k1, world=128x128",
  "output": "~45 FPS. DCS adds ~15% overhead. Ring buffer K=1 is optimal.",
  "metadata": {
    "room_id": "forge",
    "source": "benchmark",
    "hardware": "rtx-4050",
    "agents": 10000,
    "fps": 45
  }
}
```

## Cross-Pollination

- **ct-lab**: Constraint theory experiments run here, results exported as tiles
- **plato-chess-dojo**: Chess eval benchmarks shared
- **plato-jetson**: Evennia room mirrors Forge data
- **forgemaster repo**: FM's GPU sweep results feed tiles here

## Boarding

```bash
nc 147.224.38.131 4040  # Connect to PLATO
# Navigate to Forge room (or create it if it doesn't exist yet)
# Ask: "What's the fastest matrix multiply on Jetson?"
# The NPC searches benchmark tiles and returns data-backed answers
```
