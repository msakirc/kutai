"""Benchmark a single llama-server model. Usage: python benchmark_model.py <model_path> [gpu_layers] [ctx_size]"""
import sys, time, json, subprocess

def query(prompt, max_tokens=100):
    import httpx
    r = httpx.post("http://127.0.0.1:8080/v1/chat/completions", json={
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }, timeout=120)
    return r.json()

def main():
    model_path = sys.argv[1]
    gpu_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    ctx_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

    # Short prompt test
    print(f"\n--- Short prompt (1 sentence) ---")
    data = query("Write one sentence about Istanbul.", 50)
    t = data.get("timings", {})
    u = data.get("usage", {})
    content = data["choices"][0]["message"].get("content", "")
    reasoning = data["choices"][0]["message"].get("reasoning_content", "")
    print(f"Gen: {t.get('predicted_per_second', 0):.1f} tok/s | Prompt: {t.get('prompt_per_second', 0):.1f} tok/s")
    print(f"Tokens: prompt={u.get('prompt_tokens',0)}, output={u.get('completion_tokens',0)}")
    print(f"Thinking: {'YES' if reasoning else 'NO'} ({len(reasoning or '')} chars)")
    print(f"Content: {content[:120]}")

    # Medium prompt test
    print(f"\n--- Medium prompt (~500 token input, 200 output) ---")
    medium_prompt = (
        "You are a helpful assistant. The user asked about coffee machine prices in Turkey. "
        "Here are search results:\n"
        "1. De'Longhi Dinamica Plus ECAM370.95.T - 28000 TL on Trendyol\n"
        "2. Philips EP2246/70 - 15000 TL on Hepsiburada\n"
        "3. Bosch TIS30321RW - 12000 TL on Amazon.com.tr\n"
        "4. De'Longhi Magnifica S ECAM22.110.B - 18000 TL on N11\n"
        "5. Philips 3200 LatteGo EP3246/70 - 22000 TL on MediaMarkt\n"
        "Summarize the top 3 options with prices and a recommendation."
    )
    data = query(medium_prompt, 200)
    t = data.get("timings", {})
    u = data.get("usage", {})
    content = data["choices"][0]["message"].get("content", "")
    reasoning = data["choices"][0]["message"].get("reasoning_content", "")
    print(f"Gen: {t.get('predicted_per_second', 0):.1f} tok/s | Prompt: {t.get('prompt_per_second', 0):.1f} tok/s")
    print(f"Tokens: prompt={u.get('prompt_tokens',0)}, output={u.get('completion_tokens',0)}")
    print(f"Latency: {t.get('predicted_ms', 0)/1000:.1f}s total gen")
    print(f"Thinking: {'YES' if reasoning else 'NO'}")
    print(f"Content preview: {content[:150]}")

    # VRAM
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print(f"\nVRAM: {result.stdout.strip()}")

if __name__ == "__main__":
    main()
