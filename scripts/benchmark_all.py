"""Benchmark all local models sequentially. Starts/stops llama-server for each.

Usage: .venv/Scripts/python.exe -u scripts/benchmark_all.py
"""
import subprocess, time, sys, os, signal

sys.stdout.reconfigure(encoding="utf-8")

LLAMA_SERVER = r"C:\Users\sakir\ai\llama.cpp\llama-server.exe"
MODELS_DIR = r"C:\Users\sakir\ai\models"
PORT = 8080

# (name, filename, ctx_size, extra_args)
MODELS = [
    ("Qwen3.5-9B-UD", "Qwen3.5-9B-UD-Q4_K_XL.gguf", 32768,
     ["--reasoning", "off", "--reasoning-budget", "0"]),
    ("Qwen3.5-35B-A3B (MoE)", "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf", 8192,
     ["--reasoning", "off", "--reasoning-budget", "0"]),
    ("Qwen3.5-27B", "Qwen3.5-27B.Q4_K_M.gguf", 8192,
     ["--reasoning", "off", "--reasoning-budget", "0"]),
    ("GLM-4.7-Flash", "GLM-4.7-Flash-UD-Q4_K_XL.gguf", 4096,
     ["--reasoning", "off", "--reasoning-budget", "0"]),
    ("Qwen3-Coder-30B (MoE)", "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", 8192,
     ["--override-kv", "tokenizer.ggml.eos_token_id=int:151645",
      "--reasoning", "off", "--reasoning-budget", "0"]),
    ("Apriel-15B-Thinker", "ServiceNow-AI_Apriel-1.6-15b-Thinker-Q4_K_L.gguf", 8192,
     ["--no-jinja", "--chat-template", "chatml"]),
    ("GigaChat3.1-Lightning (MoE)", "GigaChat3.1-Lightning-Uncensored.i1-Q4_K_M.gguf", 8192, []),
    ("gpt-oss-20b (MoE)", "gpt-oss-20b-UD-Q4_K_XL.gguf", 8192,
     ["--reasoning", "off", "--reasoning-budget", "0"]),
    ("gemma-4-26B-A4B (MoE)", "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf", 8192,
     ["--reasoning", "off", "--reasoning-budget", "0"]),
    ("nerdsking-7B", "nerdsking-python-coder-7B-i_Q5_k_m.gguf", 8192, []),
]

import httpx


def ensure_server_dead():
    """Kill llama-server and wait until port 8080 is free."""
    os.system("taskkill /IM llama-server.exe /F >NUL 2>&1")
    # Wait until port is actually free (not just process dead)
    for _ in range(30):
        try:
            httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            time.sleep(1)  # still responding — wait
        except Exception:
            break  # connection refused = port free
    time.sleep(1)  # extra settle time


def wait_healthy(timeout=120):
    """Wait for llama-server to respond healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def query(prompt, max_tokens=100):
    """Send a chat completion request."""
    r = httpx.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    return r.json()


def vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def benchmark_model(name, filename, ctx_size, extra_args):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"=== {name} === SKIPPED (file not found)")
        return None

    print(f"=== {name} ===")
    sys.stdout.flush()

    # Ensure clean state
    ensure_server_dead()

    cmd = [
        LLAMA_SERVER,
        "--model", path,
        "--alias", "local-model",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--ctx-size", str(ctx_size),
        "--flash-attn", "auto",
        "--metrics",
        "--threads", "9",
        "--batch-size", "2048",
        "--ubatch-size", "512",
    ] + extra_args

    load_start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    if not wait_healthy(120):
        print(f"  FAILED TO LOAD (120s timeout)")
        sys.stdout.flush()
        proc.kill()
        proc.wait()
        ensure_server_dead()
        return None

    load_time = time.time() - load_start
    print(f"  Load: {load_time:.0f}s | VRAM: {vram()}")
    sys.stdout.flush()

    result = {"name": name, "load_time": load_time}

    for label, prompt, mt in [
        ("Short", "Write one sentence about coffee.", 50),
        ("Medium",
         "Summarize: 1. DeLonghi 28000TL 4.8stars 2. Philips 15000TL 4.5stars "
         "3. Bosch 12000TL 4.3stars. Top pick and why in 2 sentences.", 200),
    ]:
        try:
            t0 = time.time()
            d = query(prompt, mt)
            wall = time.time() - t0
            t = d.get("timings", {})
            u = d.get("usage", {})
            rr = d["choices"][0]["message"].get("reasoning_content", "")
            gen = t.get("predicted_per_second", 0)
            prompt_tps = t.get("prompt_per_second", 0)
            out = u.get("completion_tokens", 0)
            think = "YES" if rr else "NO"
            print(f"  {label}: gen={gen:.1f} tok/s  prompt={prompt_tps:.1f} tok/s  out={out}  wall={wall:.1f}s  think={think}")
            result[f"{label.lower()}_gen"] = gen
            result[f"{label.lower()}_prompt"] = prompt_tps
            result[f"{label.lower()}_think"] = think
        except Exception as e:
            print(f"  {label}: FAILED {type(e).__name__}")
            result[f"{label.lower()}_gen"] = 0
    sys.stdout.flush()

    # Clean shutdown
    proc.kill()
    proc.wait()
    ensure_server_dead()
    return result


def main():
    print(f"Driver: {subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}")
    print(f"GPU: {subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}")
    print(f"VRAM before: {vram()}")
    print()

    results = []
    for name, fn, ctx, extra in MODELS:
        r = benchmark_model(name, fn, ctx, extra)
        if r:
            results.append(r)
        print()
        sys.stdout.flush()

    print("=" * 80)
    print("SUMMARY (New Driver Benchmark)")
    print("=" * 80)
    print(f"{'Model':<30} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Think':>6} {'Load':>5}")
    print("-" * 70)
    for r in results:
        gen = r.get("medium_gen", r.get("short_gen", 0))
        prompt = r.get("medium_prompt", r.get("short_prompt", 0))
        think = r.get("short_think", "?")
        print(f"{r['name']:<30} {gen:>10.1f} {prompt:>12.1f} {think:>6} {r['load_time']:>4.0f}s")
    print()


if __name__ == "__main__":
    main()
