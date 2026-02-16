# run.py
import asyncio
import sys
from config import print_config, MODEL_TIERS, AVAILABLE_KEYS
from orchestrator import Orchestrator

def preflight_check():
    """Verify minimum requirements before starting."""
    print_config()

    if not any(AVAILABLE_KEYS.values()):
        print("❌ FATAL: No API keys configured at all!")
        sys.exit(1)

    if not MODEL_TIERS:
        print("❌ FATAL: No model tiers available!")
        sys.exit(1)

    if "cheap" not in MODEL_TIERS:
        print("⚠️  WARNING: No 'cheap' tier available. All tasks will use paid models.")

    # Quick connectivity test
    print("\n Testing model connectivity...")
    broken_tiers = []

    import litellm
    for tier, config in list(MODEL_TIERS.items()):
        try:
            response = litellm.completion(
                model=config["model"],
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5,
                temperature=0
            )
            print(f"   ✅ {tier}: {config['model']} — working")
        except Exception as e:
            short_error = str(e)[:80]
            print(f"   ❌ {tier}: {config['model']} — {short_error}")
            broken_tiers.append(tier)
            # Remove broken tier

    # Remove broken tiers AFTER iteration
    for tier in broken_tiers:
        print(f"      ↳ Removing '{tier}' tier from active config")
        del MODEL_TIERS[tier]

    if not MODEL_TIERS:
        print("\n❌ FATAL: No working models after connectivity test!")
        sys.exit(1)

    print(f"\n✅ Preflight complete. {len(MODEL_TIERS)} working tier(s).\n")


if __name__ == "__main__":
    preflight_check()
    orchestrator = Orchestrator()
    asyncio.run(orchestrator.start())
