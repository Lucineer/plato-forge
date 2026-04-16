#!/usr/bin/env python3
"""PLATO Forge — GPU kernel room where agents submit, benchmark, and optimize CUDA/PTX kernels."""

import yaml, os, subprocess, tempfile, re
from pathlib import Path
from datetime import datetime, timezone
import fcntl

os.environ.setdefault("PATH", "/usr/local/cuda/bin:" + os.environ.get("PATH", ""))

WORLD_DIR = Path(os.environ.get("WORLD_DIR", "world"))
KERNELS_DIR = WORLD_DIR / "kernels"
BENCHMARKS_DIR = WORLD_DIR / "benchmarks"
RESULTS_DIR = WORLD_DIR / "results"
COMMANDS_DIR = WORLD_DIR / "commands"
ROOMS_DIR = WORLD_DIR / "rooms"
LOGS_DIR = WORLD_DIR / "logs"
MAX_TURNS = 20

def log(level, msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def atomic_write(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yaml.dump(data, f, default_flow_style=False)
        fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(tmp, path)

def atomic_read(path):
    try:
        with open(path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            d = yaml.safe_load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return d or {}
    except FileNotFoundError:
        return {}

def load_room():
    return atomic_read(ROOMS_DIR / "forge.yaml")

def save_room(state):
    atomic_write(ROOMS_DIR / "forge.yaml", state)

def validate_kernel(source, lang="cuda"):
    """Validate kernel source for safety."""
    errors = []
    # Block dangerous patterns
    dangerous = ["system(", "exec(", "popen(", "fork(", "socket(", "connect(",
                 "unlink(", "remove(", "rmdir(", "__attribute__((constructor))"]
    for pat in dangerous:
        if pat in source:
            errors.append(f"Blocked pattern: {pat}")
    # Block file I/O (except through kernel params)
    if re.search(r'fopen|fwrite|fread|std::ifstream|std::ofstream', source):
        errors.append("File I/O not allowed in kernels")
    # Must have __global__ or .entry
    if lang == "cuda" and "__global__" not in source:
        errors.append("Missing __global__ kernel function")
    if lang == "ptx" and ".entry" not in source:
        errors.append("Missing .entry directive")
    return len(errors) == 0, errors

def process_submit(cmd, agent):
    """Submit a kernel for benchmarking."""
    lang = cmd.get("lang", "cuda").lower()
    source = cmd.get("source", "")
    name = cmd.get("name", f"{agent}-kernel")
    desc = cmd.get("description", "")

    if not source:
        return {"passed": False, "error": "Empty source"}

    safe, errors = validate_kernel(source, lang)
    if not safe:
        return {"passed": False, "error": errors}

    kid = f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    ext = ".cu" if lang == "cuda" else ".ptx"
    (KERNELS_DIR / f"{kid}{ext}").write_text(source)

    kernel = {
        "id": kid, "agent": agent, "name": name, "lang": lang,
        "description": desc, "status": "submitted",
        "submitted": datetime.now(timezone.utc).isoformat(),
        "benchmarks": [],
    }
    atomic_write(KERNELS_DIR / f"{kid}.yaml", kernel)

    room = load_room()
    room.setdefault("stats", {})
    room["stats"]["kernels_submitted"] = room["stats"].get("kernels_submitted", 0) + 1
    save_room(room)
    log("INFO", f"Kernel {kid} submitted by {agent}")
    return {"passed": True, "kernel_id": kid}

def process_benchmark(cmd, agent):
    """Benchmark a kernel (syntax + register check on available GPU)."""
    kid = cmd.get("kernel_id")
    if not kid:
        return {"passed": False, "error": "Missing kernel_id"}

    # Find source
    # Find source file (skip .yaml metadata)
    sources = [s for s in KERNELS_DIR.glob(f"{kid}*")
               if s.suffix in (".cu", ".ptx", ".c", ".cpp")]
    if not sources:
        return {"passed": False, "error": "Kernel not found"}

    source = sources[0].read_text()
    ext = sources[0].suffix
    lang = "ptx" if ext == ".ptx" else "cuda"

    result = {"kernel_id": kid, "agent": agent, "lang": lang,
              "timestamp": datetime.now(timezone.utc).isoformat()}

    # Try compilation
    with tempfile.NamedTemporaryFile(suffix=ext, mode='w', delete=False) as f:
        f.write(source)
        tmp_path = f.name

    try:
        if lang == "ptx":
            proc = subprocess.run(["ptxas", "--verbose", "--gpu-name", "sm_87", tmp_path, "-o", "/dev/null"],
                                  capture_output=True, text=True, timeout=30)
            result["ptxas_success"] = proc.returncode == 0
            regs = 0
            for line in proc.stderr.split("\n"):
                m = re.search(r'(\d+)\s+register', line)
                if m:
                    regs = int(m.group(1))
            result["registers"] = regs
            if proc.returncode != 0:
                result["error"] = proc.stderr[:500]
        else:
            proc = subprocess.run(["nvcc", "-arch=sm_87", "--ptx", tmp_path, "-o", "/dev/null"],
                                  capture_output=True, text=True, timeout=60)
            result["nvcc_success"] = proc.returncode == 0
            if proc.returncode != 0:
                result["error"] = proc.stderr[:500]
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
    except FileNotFoundError:
        result["compiler_missing"] = True
    finally:
        os.unlink(tmp_path)

    # Save benchmark
    bid = f"{kid}-bench-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    atomic_write(BENCHMARKS_DIR / f"{bid}.yaml", result)

    # Update kernel
    kernel = atomic_read(KERNELS_DIR / f"{kid}.yaml")
    if kernel.get("id"):
        kernel.setdefault("benchmarks", []).append(bid)
        kernel["status"] = "benchmarked" if result.get("ptxas_success") or result.get("nvcc_success") else "failed"
        atomic_write(KERNELS_DIR / f"{kid}.yaml", kernel)

    room = load_room()
    room.setdefault("stats", {})
    room["stats"]["benchmarks_run"] = room["stats"].get("benchmarks_run", 0) + 1
    if result.get("ptxas_success") or result.get("nvcc_success"):
        room["stats"]["compiles_pass"] = room["stats"].get("compiles_pass", 0) + 1
    else:
        room["stats"]["compiles_fail"] = room["stats"].get("compiles_fail", 0) + 1
    save_room(room)

    status = "PASS" if result.get("ptxas_success") or result.get("nvcc_success") else "FAIL"
    log("INFO", f"Benchmark {kid}: {status}")
    return {"passed": result.get("ptxas_success") or result.get("nvcc_success"), "result": result}

def process_turns():
    os.environ.setdefault("PATH", "/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    for d in [COMMANDS_DIR, KERNELS_DIR, BENCHMARKS_DIR, RESULTS_DIR, ROOMS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not (ROOMS_DIR / "forge.yaml").exists():
        save_room({"name": "PLATO Forge", "stats": {
            "kernels_submitted": 0, "benchmarks_run": 0,
            "compiles_pass": 0, "compiles_fail": 0}})
    commands = sorted(COMMANDS_DIR.glob("*.yaml"))
    if not commands:
        return
    log("INFO", f"Processing {len(commands)} commands")
    counts = {}
    for cp in commands:
        cmd = atomic_read(cp)
        if not cmd:
            cp.unlink(); continue
        agent = cmd.get("agent", "unknown")
        counts[agent] = counts.get(agent, 0) + 1
        if counts[agent] > MAX_TURNS:
            cp.unlink(); continue
        action = cmd.get("action")
        if action == "submit":
            r = process_submit(cmd, agent)
        elif action == "benchmark":
            r = process_benchmark(cmd, agent)
        else:
            r = {"passed": False, "error": f"Unknown: {action}"}
        atomic_write(LOGS_DIR / f"turn-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}.yaml",
                     {"agent": agent, "action": action, "result": r,
                      "timestamp": datetime.now(timezone.utc).isoformat()})
        cp.unlink()
    log("INFO", f"Turn done")

if __name__ == "__main__":
    process_turns()
