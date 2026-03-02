# run.py
import asyncio
import os
import subprocess
import sys
from dotenv import load_dotenv
load_dotenv()

from config import print_config, DOCKER_CONTAINER_NAME


def check_docker_sandbox():
    """Ensure the Docker sandbox is built and running."""
    print("🐳 Checking Docker sandbox...")

    # Check if image exists
    result = subprocess.run(
        ["docker", "images", "-q", "orchestrator-sandbox"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print("   Building sandbox image...")

        # ── FIX: point build context to the sandbox/ directory ──
        sandbox_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox")
        build = subprocess.run(
            ["docker", "build", "-t", "orchestrator-sandbox", sandbox_dir],
            capture_output=False
        )
        if build.returncode != 0:
            print("   ⚠️  Docker build failed. Shell tool will be unavailable.")
            return False


def check_env():
    """Verify minimum required env vars."""
    required = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_CHAT_ID"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"❌ Missing required env vars: {missing}")
        print("   Create a .env file with at least:")
        print("   TELEGRAM_BOT_TOKEN=...")
        print("   TELEGRAM_ADMIN_CHAT_ID=...")
        sys.exit(1)

    # Check for at least one model provider
    providers = ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "GEMINI_API_KEY", "CEREBRAS_API_KEY", "SAMBANOVA_API_KEY"]
    has_cloud = any(os.getenv(p) for p in providers)

    # Check ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=3)
        has_local = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_local = False

    if not has_cloud and not has_local:
        print("❌ No model providers available!")
        print("   Set at least one of: GROQ_API_KEY, GEMINI_API_KEY, etc.")
        print("   Or install Ollama with at least one model.")
        sys.exit(1)


async def main():
    check_env()
    print_config()
    check_docker_sandbox()

    print("\n🚀 Starting orchestrator...\n")

    # Graceful shutdown event
    import signal
    shutdown_event = asyncio.Event()

    def _signal_handler(sig, frame):
        sig_name = signal.Signals(sig).name
        print(f"\n⚠️  Received {sig_name} — initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    from orchestrator import Orchestrator
    orch = Orchestrator(shutdown_event=shutdown_event)
    await orch.start()


if __name__ == "__main__":
    asyncio.run(main())
