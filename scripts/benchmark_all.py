"""Benchmark all local models sequentially. Starts/stops llama-server for each."""
import subprocess, time, sys, json, os

LLAMA_SERVER = r"C:\Users\sakir\ai\llama.cpp\llama-server.exe"
MODELS_DIR = r"C:\Users\sakir\ai\models"
PYTHON = r"C:\Users\sakir\Dropbox\Workspaces\kutay\.venv\Scripts\python.exe"
BENCHMARK_SCRIPT = r"C:\Users\sakir\Dropbox\Workspaces\kutay\scripts\benchmark_model.py"

# Model configs: (filename, gpu_layers, ctx_size, extra_args)
# gpu_layers tuned for 8GB VRAM
MODELS = [
    ("Qwen3.5-9B-UD-Q4_K_XL.gguf", 99, 32768, []),
    ("Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf", 99, 8192,
     ["--override-kv", "tokenizer.ggml.eos_token_id=int:151645"]),
    ("Qwen3.5-27B.Q4_K_M.gguf", 99, 8192, []),
    ("GLM-4.7-Flash-UD-Q4_K_XL.gguf", 99, 4096,
     ["--override-kv", "tokenizer.ggml.eos_token_id=int:151645"]),
    ("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", 99, 8192,
     ["--override-kv", "tokenizer.ggml.eos_token_id=int:151645"]),
    ("ServiceNow-AI_Apriel-1.6-15b-Thinker-Q6_K_L.gguf", 99, 16384, []),
    ("gemma-3-27b-it-heretic-v1.2.IQ4_XS.gguf", 99, 8192, []),
    ("nerdsking-python-coder-7B-i_Q5_k_m.gguf", 99, 8192, []),
]

import httpx

def wait_healthy(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get("http://127.0.0.1:8080/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def kill_server():
    os.system("taskkill /IM llama-server.exe /F >NUL 2>&1")
    time.sleep(2)

def get_vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    parts = r.stdout.strip().split(",")
    return int(parts[0].strip()), int(parts[1].strip())

def query_llm(prompt, max_tokens=100):
    r = httpx.post("http://127.0.0.1:8080/v1/chat/completions", json={
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }, timeout=120)
    return r.json()

def benchmark_model(name, path, gpu_layers, ctx_size, extra_args):
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"Config: gpu_layers={gpu_layers}, ctx={ctx_size}")
    print(f"{'='*60}")

    kill_server()

    cmd = [
        LLAMA_SERVER,
        "--model", path,
        "--alias", "local-model",
        "--port", "8080",
        "--host", "127.0.0.1",
        "--n-gpu-layers", str(gpu_layers),
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

    if not wait_healthy(90):
        print(f"  FAILED TO LOAD (timeout 90s)")
        kill_server()
        return None

    load_time = time.time() - load_start
    vram_used, vram_free = get_vram()
    print(f"  Load time: {load_time:.1f}s | VRAM: {vram_used}MB used, {vram_free}MB free")

    results = {"name": name, "load_time": load_time, "vram_used": vram_used}

    # Test 1: Short prompt
    try:
        data = query_llm("Write one sentence about Istanbul.", 50)
        t = data.get("timings", {})
        u = data.get("usage", {})
        reasoning = data["choices"][0]["message"].get("reasoning_content", "")
        content = data["choices"][0]["message"].get("content", "")
        gen_tps = t.get("predicted_per_second", 0)
        prompt_tps = t.get("prompt_per_second", 0)
        results["short_gen_tps"] = gen_tps
        results["short_prompt_tps"] = prompt_tps
        results["thinking"] = bool(reasoning)
        print(f"  Short: gen={gen_tps:.1f} tok/s, prompt={prompt_tps:.1f} tok/s, thinking={'YES' if reasoning else 'NO'}")
        print(f"  Content: {(content or '(empty)')[:80]}")
    except Exception as e:
        print(f"  Short: FAILED - {e}")
        results["short_gen_tps"] = 0

    # Test 2: Medium prompt (agent-like)
    try:
        medium = (
            "You are a helpful assistant. Summarize these search results about coffee machines:\n"
            "1. De'Longhi Dinamica Plus - 28000 TL on Trendyol, 4.8 stars\n"
            "2. Philips EP2246/70 - 15000 TL on Hepsiburada, 4.5 stars\n"
            "3. Bosch TIS30321RW - 12000 TL on Amazon, 4.3 stars\n"
            "Give top 3 with prices and recommendation in 2-3 sentences."
        )
        data = query_llm(medium, 200)
        t = data.get("timings", {})
        u = data.get("usage", {})
        gen_tps = t.get("predicted_per_second", 0)
        prompt_tps = t.get("prompt_per_second", 0)
        out_tokens = u.get("completion_tokens", 0)
        gen_ms = t.get("predicted_ms", 0)
        results["med_gen_tps"] = gen_tps
        results["med_prompt_tps"] = prompt_tps
        results["med_latency"] = gen_ms / 1000
        print(f"  Medium: gen={gen_tps:.1f} tok/s, prompt={prompt_tps:.1f} tok/s, {out_tokens} tokens in {gen_ms/1000:.1f}s")
    except Exception as e:
        print(f"  Medium: FAILED - {e}")
        results["med_gen_tps"] = 0

    kill_server()
    return results

def main():
    all_results = []
    for filename, gpu_layers, ctx_size, extra in MODELS:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"\nSKIPPED: {filename} (not found)")
            continue
        name = filename.replace(".gguf", "")
        result = benchmark_model(name, path, gpu_layers, ctx_size, extra)
        if result:
            all_results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Think':>6} {'VRAM':>6} {'Load':>5}")
    print("-" * 90)
    for r in all_results:
        tps = r.get("med_gen_tps", r.get("short_gen_tps", 0))
        ptps = r.get("med_prompt_tps", r.get("short_prompt_tps", 0))
        think = "YES" if r.get("thinking") else "NO"
        print(f"{r['name']:<45} {tps:>10.1f} {ptps:>12.1f} {think:>6} {r['vram_used']:>5}M {r['load_time']:>4.0f}s")

if __name__ == "__main__":
    main()
