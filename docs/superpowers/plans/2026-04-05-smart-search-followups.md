# Smart Search Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Layer 0→Layer 1 dead zone, harden API reliability demotion logic, and add human-friendly response formatters for common API categories.

**Architecture:** Three independent changes to the smart search pipeline: (1) remove the upper-bound score check in `enrich_context()` so Layer 1 catches Layer 0 failures, (2) raise demotion minimum sample sizes and add consecutive-failure tracking in `api_reliability`, (3) add a `_FORMATTERS` dispatch dict in `fast_resolver.py` for weather/currency/earthquake/pharmacy/fuel/prayer_times categories.

**Tech Stack:** Python 3.10, aiosqlite, pytest, unittest.mock

---

### Task 1: Fix Layer 0 → Layer 1 dead zone

**Files:**
- Modify: `src/core/fast_resolver.py:73-74`
- Test: `tests/test_fast_resolver.py` (create)

The bug: if Layer 0 matches (score ≥ 0.6) but the API call fails, `try_resolve()` returns `None`. Then `enrich_context()` re-scores, sees score ≥ 0.6, and returns `None` at line 73-74 ("Layer 0 would have caught this"). So neither layer produces output.

Fix: remove the upper-bound check. If we reach `enrich_context()`, Layer 0 already failed or returned — the guard is wrong.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fast_resolver.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def _make_fake_api(name="wttr.in", category="weather"):
    api = MagicMock()
    api.name = name
    api.category = category
    api.example_endpoint = "https://wttr.in/Istanbul?format=j1"
    return api


@pytest.mark.asyncio
async def test_enrich_context_works_when_score_above_layer0_threshold():
    """Layer 1 should still enrich even if score >= 0.6 (L0 failed to resolve)."""
    from src.core.fast_resolver import enrich_context

    fake_api = _make_fake_api()
    match = {"api": fake_api, "category": "weather", "score": 0.8}

    with patch("src.core.fast_resolver._find_best_match", new_callable=AsyncMock, return_value=match), \
         patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock, return_value={"temp": "22C"}), \
         patch("src.core.fast_resolver._format_response", return_value="22C in Istanbul"), \
         patch("src.infra.db.log_smart_search", new_callable=AsyncMock), \
         patch("src.infra.db.record_api_call", new_callable=AsyncMock):
        result = await enrich_context({"title": "Istanbul hava durumu"})

    assert result is not None
    assert "22C" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fast_resolver.py::test_enrich_context_works_when_score_above_layer0_threshold -v`
Expected: FAIL — `assert result is not None` fails because score 0.8 ≥ 0.6 triggers the early return.

- [ ] **Step 3: Remove the upper-bound guard**

In `src/core/fast_resolver.py`, delete lines 73-74:

```python
        if match["score"] >= _LAYER0_THRESHOLD:
            return None  # Layer 0 would have caught this
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fast_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_fast_resolver.py src/core/fast_resolver.py
git commit -m "fix(fast_resolver): remove L1 upper-bound guard — fixes dead zone when L0 API call fails"
```

---

### Task 2: Harden API reliability demotion thresholds

**Files:**
- Modify: `src/infra/db.py:429-437` (schema — add `consecutive_failures` column)
- Modify: `src/infra/db.py:2704-2747` (`record_api_call` — raise minimums, track consecutive failures)
- Modify: `src/infra/db.py:2750-2759` (`get_api_reliability` — return new column)
- Modify: `src/infra/db.py:2802-2808` (`unsuspend_api` — reset consecutive_failures)
- Test: `tests/test_api_reliability.py` (create)

The problem: demotion triggers too early (5-10 calls). A single blip on a reliable API triggers `warning`. We raise minimums to 15-20 and add `consecutive_failures` to catch APIs that just went down (3 consecutive failures → `warning` regardless of overall rate).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_reliability.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import aiosqlite


@pytest.fixture
async def mem_db():
    """In-memory SQLite for reliability tests."""
    db = await aiosqlite.connect(":memory:")
    await db.execute("""
        CREATE TABLE api_reliability (
            api_name TEXT PRIMARY KEY,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            consecutive_failures INTEGER DEFAULT 0,
            last_success TEXT,
            last_failure TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
    await db.commit()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_single_failure_does_not_demote(mem_db):
    """An API with 9 successes and 1 failure should stay active."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(9):
            await record_api_call("wttr.in", success=True)
        await record_api_call("wttr.in", success=False)

    cur = await mem_db.execute("SELECT status, consecutive_failures FROM api_reliability WHERE api_name = 'wttr.in'")
    row = await cur.fetchone()
    assert row[0] == "active", f"Expected active, got {row[0]}"
    assert row[1] == 1


@pytest.mark.asyncio
async def test_low_sample_does_not_demote(mem_db):
    """Even with 100% failure, fewer than 15 calls should not demote."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(14):
            await record_api_call("badapi", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'badapi'")
    row = await cur.fetchone()
    assert row[0] == "active", f"Expected active with <15 calls, got {row[0]}"


@pytest.mark.asyncio
async def test_high_failure_rate_demotes_after_threshold(mem_db):
    """API with >=15 calls and <25% success rate gets demoted."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(3):
            await record_api_call("badapi", success=True)
        for _ in range(12):
            await record_api_call("badapi", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'badapi'")
    row = await cur.fetchone()
    assert row[0] == "demoted"


@pytest.mark.asyncio
async def test_consecutive_failures_triggers_warning(mem_db):
    """3 consecutive failures should trigger warning even with good overall rate."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(20):
            await record_api_call("flaky", success=True)
        for _ in range(3):
            await record_api_call("flaky", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'flaky'")
    row = await cur.fetchone()
    assert row[0] == "warning"


@pytest.mark.asyncio
async def test_success_resets_consecutive_failures(mem_db):
    """A success should reset the consecutive failure counter."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        for _ in range(2):
            await record_api_call("recov", success=False)
        await record_api_call("recov", success=True)

    cur = await mem_db.execute("SELECT consecutive_failures FROM api_reliability WHERE api_name = 'recov'")
    row = await cur.fetchone()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_suspended_at_very_low_rate(mem_db):
    """>=20 calls and <10% success → suspended."""
    from src.infra.db import record_api_call
    with patch("src.infra.db.get_db", new_callable=AsyncMock, return_value=mem_db):
        await record_api_call("dead", success=True)
        for _ in range(19):
            await record_api_call("dead", success=False)

    cur = await mem_db.execute("SELECT status FROM api_reliability WHERE api_name = 'dead'")
    row = await cur.fetchone()
    assert row[0] == "suspended"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_reliability.py -v`
Expected: Multiple failures — `consecutive_failures` column doesn't exist, thresholds are wrong.

- [ ] **Step 3: Add `consecutive_failures` column to schema**

In `src/infra/db.py`, modify the `api_reliability` CREATE TABLE (around line 429):

```python
    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_reliability (
            api_name TEXT PRIMARY KEY,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            consecutive_failures INTEGER DEFAULT 0,
            last_success TEXT,
            last_failure TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
```

Also add a migration for existing databases (right after the CREATE TABLE, before the `await db.commit()`):

```python
    # Migration: add consecutive_failures column if missing
    try:
        await db.execute("ALTER TABLE api_reliability ADD COLUMN consecutive_failures INTEGER DEFAULT 0")
    except Exception:
        pass  # column already exists
```

- [ ] **Step 4: Rewrite `record_api_call` with new thresholds and consecutive tracking**

Replace `record_api_call` in `src/infra/db.py` (lines 2704-2747):

```python
async def record_api_call(api_name: str, success: bool):
    """Update api_reliability counters and auto-demote if needed."""
    db = await get_db()
    now = "strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')"
    if success:
        await db.execute(
            f"""INSERT INTO api_reliability (api_name, success_count, last_success, consecutive_failures)
                VALUES (?, 1, {now}, 0)
                ON CONFLICT(api_name) DO UPDATE SET
                    success_count = success_count + 1,
                    consecutive_failures = 0,
                    last_success = {now}""",
            (api_name,),
        )
    else:
        await db.execute(
            f"""INSERT INTO api_reliability (api_name, failure_count, last_failure, consecutive_failures)
                VALUES (?, 1, {now}, 1)
                ON CONFLICT(api_name) DO UPDATE SET
                    failure_count = failure_count + 1,
                    consecutive_failures = consecutive_failures + 1,
                    last_failure = {now}""",
            (api_name,),
        )
    # Auto-demote check
    cur = await db.execute(
        "SELECT success_count, failure_count, consecutive_failures FROM api_reliability WHERE api_name = ?",
        (api_name,),
    )
    row = await cur.fetchone()
    if row:
        total = row[0] + row[1]
        rate = row[0] / max(total, 1)
        consec = row[2]
        if total >= 20 and rate < 0.10:
            status = "suspended"
        elif total >= 15 and rate < 0.25:
            status = "demoted"
        elif total >= 15 and rate < 0.50:
            status = "warning"
        elif consec >= 3:
            status = "warning"
        else:
            status = "active"
        await db.execute(
            "UPDATE api_reliability SET status = ? WHERE api_name = ?",
            (status, api_name),
        )
    await db.commit()
```

- [ ] **Step 5: Update `get_api_reliability` to return `consecutive_failures`**

In `src/infra/db.py`, replace `get_api_reliability` (lines 2750-2759):

```python
async def get_api_reliability(api_name: str) -> dict | None:
    db = await get_db()
    cur = await db.execute(
        "SELECT api_name, success_count, failure_count, status, consecutive_failures FROM api_reliability WHERE api_name = ?",
        (api_name,),
    )
    row = await cur.fetchone()
    if not row:
        return None
    return {"api_name": row[0], "success_count": row[1], "failure_count": row[2], "status": row[3], "consecutive_failures": row[4]}
```

- [ ] **Step 6: Update `unsuspend_api` to reset `consecutive_failures`**

In `src/infra/db.py`, replace `unsuspend_api` (lines 2802-2808):

```python
async def unsuspend_api(api_name: str):
    db = await get_db()
    await db.execute(
        "UPDATE api_reliability SET status = 'active', success_count = 0, failure_count = 0, consecutive_failures = 0 WHERE api_name = ?",
        (api_name,),
    )
    await db.commit()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_api_reliability.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/infra/db.py tests/test_api_reliability.py
git commit -m "fix(reliability): raise demotion minimums to 15-20 calls, add consecutive failure tracking"
```

---

### Task 3: Add category-specific response formatters

**Files:**
- Modify: `src/core/fast_resolver.py:237-248` (replace `_format_response`)
- Test: `tests/test_fast_resolver.py` (add tests)

Replace the generic JSON dump with a `_FORMATTERS` dict dispatching to per-category functions. Start with 6 categories: weather, currency, earthquake, pharmacy, fuel, prayer_times. Fallback stays as truncated JSON.

- [ ] **Step 1: Write formatter tests**

Add to `tests/test_fast_resolver.py`:

```python
def test_format_weather_response():
    from src.core.fast_resolver import _format_response

    raw = {
        "current_condition": [{"temp_C": "22", "weatherDesc": [{"value": "Sunny"}], "humidity": "45", "windspeedKmph": "12"}],
        "nearest_area": [{"areaName": [{"value": "Istanbul"}]}]
    }
    result = _format_response(raw, "weather", "wttr.in")
    assert "22" in result
    assert "Istanbul" in result


def test_format_currency_response():
    from src.core.fast_resolver import _format_response

    raw = {"rates": {"TRY": 38.45}, "base": "USD"}
    result = _format_response(raw, "currency", "exchangerate-api")
    assert "38.45" in result
    assert "USD" in result


def test_format_earthquake_response():
    from src.core.fast_resolver import _format_response

    raw = {"result": [{"mag": "4.2", "location": "Muğla", "date": "2026-04-05 10:30"}]}
    result = _format_response(raw, "earthquake", "kandilli")
    assert "4.2" in result
    assert "Muğla" in result


def test_format_unknown_category_falls_back_to_json():
    from src.core.fast_resolver import _format_response

    raw = {"foo": "bar"}
    result = _format_response(raw, "unknown_category", "some_api")
    assert '"foo"' in result  # JSON formatted


def test_format_string_passthrough():
    from src.core.fast_resolver import _format_response

    result = _format_response("plain text result", "weather", "wttr.in")
    assert result == "plain text result"


def test_format_truncates_long_output():
    from src.core.fast_resolver import _format_response

    raw = "x" * 3000
    result = _format_response(raw, "weather", "wttr.in")
    assert len(result) <= 2003  # 2000 + "..."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fast_resolver.py::test_format_weather_response tests/test_fast_resolver.py::test_format_currency_response tests/test_fast_resolver.py::test_format_earthquake_response -v`
Expected: FAIL — current `_format_response` just does `json.dumps`, no category-aware formatting.

- [ ] **Step 3: Implement the formatters**

Replace `_format_response` in `src/core/fast_resolver.py` (lines 237-248):

```python
def _format_weather(raw: dict) -> str | None:
    try:
        cc = raw["current_condition"][0]
        area = raw.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", "")
        desc = cc.get("weatherDesc", [{}])[0].get("value", "")
        temp = cc.get("temp_C", "?")
        humidity = cc.get("humidity", "?")
        wind = cc.get("windspeedKmph", "?")
        lines = [f"🌡 {city}: {temp}°C, {desc}"]
        lines.append(f"💧 Nem: %{humidity} | 💨 Rüzgar: {wind} km/s")
        # Forecast if available
        forecast = raw.get("weather", [])
        for day in forecast[:3]:
            date = day.get("date", "")
            hi = day.get("maxtempC", "?")
            lo = day.get("mintempC", "?")
            fdesc = day.get("hourly", [{}])[0].get("weatherDesc", [{}])[0].get("value", "")
            lines.append(f"  {date}: {lo}–{hi}°C {fdesc}")
        return "\n".join(lines)
    except (KeyError, IndexError):
        return None


def _format_currency(raw: dict) -> str | None:
    try:
        base = raw.get("base", "?")
        rates = raw.get("rates", {})
        if not rates:
            return None
        lines = [f"💱 {base} kuru:"]
        for currency, value in list(rates.items())[:10]:
            lines.append(f"  {base} → {currency}: {value}")
        return "\n".join(lines)
    except (KeyError, TypeError):
        return None


def _format_earthquake(raw: dict) -> str | None:
    try:
        quakes = raw.get("result", raw.get("earthquakes", []))
        if not quakes:
            return None
        lines = ["🌍 Son depremler:"]
        for q in quakes[:5]:
            mag = q.get("mag", q.get("magnitude", "?"))
            loc = q.get("location", q.get("title", "?"))
            date = q.get("date", q.get("time", ""))
            lines.append(f"  {mag} büyüklük — {loc} ({date})")
        return "\n".join(lines)
    except (KeyError, TypeError):
        return None


def _format_pharmacy(raw: dict) -> str | None:
    try:
        pharmacies = raw if isinstance(raw, list) else raw.get("pharmacies", raw.get("result", []))
        if not pharmacies or not isinstance(pharmacies, list):
            return None
        lines = ["💊 Nöbetçi eczaneler:"]
        for p in pharmacies[:5]:
            name = p.get("name", p.get("eczane", "?"))
            addr = p.get("address", p.get("adres", ""))
            phone = p.get("phone", p.get("telefon", ""))
            line = f"  {name}"
            if addr:
                line += f" — {addr}"
            if phone:
                line += f" (📞 {phone})"
            lines.append(line)
        return "\n".join(lines)
    except (KeyError, TypeError):
        return None


def _format_fuel(raw: dict) -> str | None:
    try:
        prices = raw if isinstance(raw, list) else raw.get("prices", raw.get("result", []))
        if isinstance(prices, dict):
            lines = ["⛽ Güncel yakıt fiyatları:"]
            for fuel_type, price in prices.items():
                lines.append(f"  {fuel_type}: {price} TL")
            return "\n".join(lines)
        if not prices or not isinstance(prices, list):
            return None
        lines = ["⛽ Güncel yakıt fiyatları:"]
        for p in prices[:6]:
            name = p.get("type", p.get("name", "?"))
            price = p.get("price", "?")
            lines.append(f"  {name}: {price} TL")
        return "\n".join(lines)
    except (KeyError, TypeError):
        return None


def _format_prayer_times(raw: dict) -> str | None:
    try:
        times = raw.get("times", raw.get("result", raw))
        if not isinstance(times, dict):
            return None
        lines = ["🕌 Namaz vakitleri:"]
        name_map = {"Imsak": "İmsak", "Gunes": "Güneş", "Ogle": "Öğle",
                     "Ikindi": "İkindi", "Aksam": "Akşam", "Yatsi": "Yatsı"}
        for key, val in times.items():
            label = name_map.get(key, key)
            if isinstance(val, str) and ":" in val:
                lines.append(f"  {label}: {val}")
        return "\n".join(lines) if len(lines) > 1 else None
    except (KeyError, TypeError):
        return None


_FORMATTERS = {
    "weather": _format_weather,
    "currency": _format_currency,
    "earthquake": _format_earthquake,
    "pharmacy": _format_pharmacy,
    "fuel": _format_fuel,
    "prayer_times": _format_prayer_times,
}


def _format_response(raw, category: str, api_name: str) -> str:
    """Format raw API response — category-specific if available, else JSON fallback."""
    if isinstance(raw, str):
        return raw[:2000] + "..." if len(raw) > 2000 else raw

    if isinstance(raw, dict) and category in _FORMATTERS:
        formatted = _FORMATTERS[category](raw)
        if formatted:
            return formatted[:2000] + "..." if len(formatted) > 2000 else formatted

    # Fallback: JSON
    import json
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, indent=2)[:2000]
    return str(raw)[:2000]
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_fast_resolver.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/fast_resolver.py tests/test_fast_resolver.py
git commit -m "feat(fast_resolver): add category-specific formatters for weather, currency, earthquake, pharmacy, fuel, prayer times"
```

---

### Task 4: Run full test suite and verify no regressions

- [ ] **Step 1: Run existing smart_search tests**

Run: `pytest tests/test_smart_search.py tests/test_smart_search_integration.py -v`
Expected: All PASS — no changes to smart_search.py itself.

- [ ] **Step 2: Run broader test suite**

Run: `pytest tests/ -x --timeout=30`
Expected: No regressions.

- [ ] **Step 3: Quick import smoke test**

Run: `python -c "from src.core.fast_resolver import try_resolve, enrich_context, _format_response, _FORMATTERS; print(f'Formatters: {list(_FORMATTERS.keys())}')"`
Expected: Prints `Formatters: ['weather', 'currency', 'earthquake', 'pharmacy', 'fuel', 'prayer_times']`
