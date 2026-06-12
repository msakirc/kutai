"""Plan 3 swap mechanic — CPS chain tests (SP5: await_inline is deleted).

Kickoff writes the chain ledger + enqueues ONE prompt_writer child with
on_complete/on_error continuations; handlers advance a sequential image
chain; finalize applies all HTML rewrites. ``_enqueue_beckman`` stays the
patchable seam — it now captures (spec, kwargs) and returns a task id."""
import json
import os

import pytest

import mr_roboto.swap_placeholder_images as swap_mod
from mr_roboto.swap_placeholder_images import (
    swap_placeholder_images,
    _scan_placeholders,
    _list_html_files,
    _PLACEHOLDER_HOST_RE,
    _coerce_result_dict,
    _extract_prompts,
    _extract_image_path,
    _load_ledger,
    _save_ledger,
    _on_prompts_done,
    _on_prompts_err,
    _on_image_done,
    _on_image_err,
    _finalize,
    ON_PROMPTS_DONE,
    ON_PROMPTS_ERR,
    ON_IMAGE_DONE,
    ON_IMAGE_ERR,
)


def test_placeholder_host_regex():
    assert _PLACEHOLDER_HOST_RE.search("https://placehold.co/64x64/eee/333?text=x")
    assert _PLACEHOLDER_HOST_RE.search("http://placehold.co/256x256")
    assert not _PLACEHOLDER_HOST_RE.search("/assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("https://example.com/real.png")


_HTML = """<!DOCTYPE html>
<html><body class="w-[390px] min-h-[844px]">
  <img src="https://placehold.co/390x220/E07A5F/FFF?text=hero"
       alt="smiling barista handing over a takeaway cup">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat"
       alt="ai-powered task triage dashboard">
  <img src="/assets/already_real.png" alt="something already swapped">
  <img src="https://placehold.co/64x64/264653/FFF?text=u"
       alt="user portrait">
</body></html>"""

_PROMPTS_3 = {
    "_schema_version": "1",
    "prompts": [
        {"placeholder_id": "home__0", "prompt": "coral barista scene"},
        {"placeholder_id": "home__1", "prompt": "slate dashboard"},
        {"placeholder_id": "home__2", "prompt": "teal portrait"},
    ],
}


def test_scan_finds_three(tmp_path):
    p = tmp_path / "home.html"
    p.write_text(_HTML, encoding="utf-8")
    hits = _scan_placeholders(str(p))
    assert len(hits) == 3
    ids = {h["placeholder_id"] for h in hits}
    assert ids == {"home__0", "home__1", "home__2"}
    assert all(h["alt"] for h in hits)
    assert all(h["width"] > 0 and h["height"] > 0 for h in hits)


def test_scan_handles_missing(tmp_path):
    assert _scan_placeholders(str(tmp_path / "missing.html")) == []


def test_scan_handles_no_placeholders(tmp_path):
    p = tmp_path / "empty.html"
    p.write_text("<html><body>no images</body></html>", encoding="utf-8")
    assert _scan_placeholders(str(p)) == []


def test_list_html_recursive(tmp_path):
    """v2 fix: walks subdirectories so multi-screen prototypes work."""
    web = tmp_path / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "home.html").write_text("<html></html>", encoding="utf-8")
    (web / "screens" / "onboarding.html").write_text("<html></html>", encoding="utf-8")
    (web / "screens" / "settings.html").write_text("<html></html>", encoding="utf-8")
    (web / "assets").mkdir(exist_ok=True)
    (web / "assets" / "ignored.png").write_bytes(b"\x89PNG")  # not an HTML
    files = _list_html_files(str(tmp_path))
    names = sorted(os.path.basename(f) for f in files)
    assert names == ["home.html", "onboarding.html", "settings.html"]


# -- tolerant result parsing ---------------------------------------------

def test_coerce_result_dict_shapes():
    assert _coerce_result_dict({"a": 1}) == {"a": 1}
    assert _coerce_result_dict(json.dumps({"a": 1})) == {"a": 1}
    assert _coerce_result_dict(None) == {}
    assert _coerce_result_dict("not json {") == {}
    assert _coerce_result_dict(json.dumps([1, 2])) == {}


def test_extract_prompts_top_level():
    assert _extract_prompts(_PROMPTS_3) == {
        "home__0": "coral barista scene",
        "home__1": "slate dashboard",
        "home__2": "teal portrait",
    }


def test_extract_prompts_nested_result_json_string():
    """Restart-reconcile / posthook shape: artifact nested as a JSON string."""
    assert _extract_prompts({"result": json.dumps(_PROMPTS_3)})["home__1"] == \
        "slate dashboard"
    assert _extract_prompts({"content": json.dumps(_PROMPTS_3)})["home__0"] == \
        "coral barista scene"
    assert _extract_prompts(json.dumps(_PROMPTS_3))["home__2"] == "teal portrait"


def test_extract_prompts_garbage_degrades():
    assert _extract_prompts({}) == {}
    assert _extract_prompts({"content": "not json"}) == {}
    assert _extract_prompts({"prompts": "not-a-list"}) == {}
    assert _extract_prompts(None) == {}


def test_extract_image_path():
    assert _extract_image_path({"path": "/x/y.png"}) == "/x/y.png"
    assert _extract_image_path({"content": "/x/y.png"}) == "/x/y.png"
    assert _extract_image_path({"result": json.dumps({"path": "/x/y.png"})}) == \
        "/x/y.png"
    assert _extract_image_path({"status": "completed"}) is None


# -- test driver ----------------------------------------------------------

class _Capture:
    """Patchable _enqueue_beckman: records (spec, kwargs), returns ids."""

    def __init__(self):
        self.calls: list[dict] = []
        self._next_id = 100

    async def __call__(self, spec, **kwargs):
        assert "await_inline" not in kwargs, "CPS regression: await_inline used"
        self._next_id += 1
        self.calls.append({"spec": spec, "kwargs": kwargs, "id": self._next_id})
        return self._next_id

    def pop_image_calls(self):
        out = [c for c in self.calls if c["spec"].get("agent_type") == "image"]
        self.calls = [c for c in self.calls
                      if c["spec"].get("agent_type") != "image"]
        return out


def _roundtrip(state: dict) -> dict:
    """cont_state survives a DB JSON round-trip — simulate it."""
    return json.loads(json.dumps(state))


@pytest.fixture
def cap(monkeypatch):
    c = _Capture()
    monkeypatch.setattr(swap_mod, "_enqueue_beckman", c)
    return c


@pytest.fixture
def ws(monkeypatch, tmp_path):
    web = tmp_path / ".web"
    web.mkdir()
    (web / "home.html").write_text(_HTML, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    return tmp_path


from PIL import Image as _Image  # noqa: E402


def _write_real_png(path, w=64, h=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _Image.new("RGB", (w, h), (100, 150, 200)).save(path, "PNG")


# -- kickoff ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_swap_no_html_files(monkeypatch, tmp_path, cap):
    web = tmp_path / ".web"; web.mkdir()
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 0
    assert res["html_files_seen"] == 0
    assert res["chain"] == "none"
    assert cap.calls == []


@pytest.mark.asyncio
async def test_swap_no_placeholders(monkeypatch, tmp_path, cap):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text("<html><body>no img</body></html>",
                                   encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    res = await swap_placeholder_images(mission_id=42)
    assert res["chain"] == "none"
    assert res["html_files_seen"] == 1
    assert cap.calls == []


@pytest.mark.asyncio
async def test_kickoff_enqueues_prompt_writer_with_continuations(ws, cap):
    res = await swap_placeholder_images(
        mission_id=42,
        design_tokens={"primary": "#E07A5F"},
        brand_voice="warm, neighborhood coffee shop",
        task_id=777,
    )

    # Queued kickoff shape.
    assert res == {
        "ok": True, "queued": True, "chain": "started",
        "placeholder_count": 3, "html_files_seen": 1,
    }

    # Exactly ONE prompt_writer child, with CPS continuations.
    assert len(cap.calls) == 1
    call = cap.calls[0]
    spec, kwargs = call["spec"], call["kwargs"]
    assert spec["agent_type"] == "prompt_writer"
    assert kwargs["on_complete"] == ON_PROMPTS_DONE
    assert kwargs["on_error"] == ON_PROMPTS_ERR
    assert kwargs["cont_state"] == {"mission_id": 42,
                                    "workspace_path": str(ws)}
    assert kwargs["parent_id"] == 777

    # Constrained-emit safety net armed via task context.
    from src.agents.prompt_writer import PROMPT_WRITER_ARTIFACT_SCHEMA
    ctx = spec["context"]
    assert ctx.get("is_workflow_step") is True
    assert ctx.get("artifact_schema") == PROMPT_WRITER_ARTIFACT_SCHEMA

    # Ledger written: placeholders + status.
    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "prompts_pending"
    assert ledger["mission_id"] == 42
    pids = [p["placeholder_id"] for p in ledger["placeholders"]]
    assert pids == ["home__0", "home__1", "home__2"]
    assert ledger["prompt_map"] == {}
    assert ledger["results"] == {}


@pytest.mark.asyncio
async def test_ledger_lives_outside_served_web_root(ws, cap):
    """The chain ledger holds absolute Windows paths, diffusion prompts and
    exception strings — it must NEVER sit inside <ws>/.web/, which is served
    live by the preview tunnel and copytree'd to a PUBLIC gh-pages repo by
    publish_preview_pages."""
    await swap_placeholder_images(mission_id=42)
    assert (ws / ".swap_state" / "swap_chain.json").is_file()
    assert not (ws / ".web" / ".swap_chain.json").exists()


@pytest.mark.asyncio
async def test_kickoff_without_task_id_omits_parent_id(ws, cap):
    await swap_placeholder_images(mission_id=42)
    assert "parent_id" not in cap.calls[0]["kwargs"]


@pytest.mark.asyncio
async def test_kickoff_enqueue_raises_degrades(ws, monkeypatch):
    async def _boom(spec, **kwargs):
        raise RuntimeError("beckman down")
    monkeypatch.setattr(swap_mod, "_enqueue_beckman", _boom)

    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["chain"] == "none"
    assert res["replaced_count"] == 0
    assert res["skipped_count"] == 3
    assert any("prompt_writer" in e for e in res["errors"])
    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "done"
    assert all(v["status"] == "skipped" for v in ledger["results"].values())


# -- kickoff reentrancy (FIX 1.2a) ------------------------------------------

@pytest.mark.asyncio
async def test_kickoff_refuses_overwrite_while_prompts_pending(ws, cap):
    """Re-pends / restart re-runs / manual retries of 5.35 must be harmless:
    a second kickoff while the chain is mid-flight must NOT overwrite the
    ledger or enqueue a duplicate prompt_writer chain."""
    await swap_placeholder_images(mission_id=42)
    assert len(cap.calls) == 1
    ledger_before = _load_ledger(str(ws))

    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["chain"] == "in_flight"
    assert res["placeholder_count"] == 3
    assert len(cap.calls) == 1  # NO second prompt_writer
    assert _load_ledger(str(ws)) == ledger_before  # untouched


@pytest.mark.asyncio
async def test_kickoff_refuses_overwrite_while_images_pending(ws, cap):
    await swap_placeholder_images(mission_id=42)
    cap.calls.clear()
    ledger = _load_ledger(str(ws))
    ledger["status"] = "images_pending"
    _save_ledger(str(ws), ledger)

    res = await swap_placeholder_images(mission_id=42)
    assert res["chain"] == "in_flight"
    assert res["placeholder_count"] == 3
    assert cap.calls == []
    assert _load_ledger(str(ws))["status"] == "images_pending"


@pytest.mark.asyncio
async def test_kickoff_proceeds_when_previous_chain_done(ws, cap):
    """A finished chain does not block a fresh kickoff (e.g. regenerated
    HTML with new placeholders): the ledger is rebuilt and a new
    prompt_writer child is enqueued."""
    await swap_placeholder_images(mission_id=42)
    cap.calls.clear()
    ledger = _load_ledger(str(ws))
    ledger["status"] = "done"
    _save_ledger(str(ws), ledger)

    res = await swap_placeholder_images(mission_id=42)
    assert res["chain"] == "started"
    assert len(cap.calls) == 1
    assert _load_ledger(str(ws))["status"] == "prompts_pending"


# -- prompts_done / prompts_err handlers -----------------------------------

async def _kickoff(ws, cap, mission_id=42):
    await swap_placeholder_images(mission_id=mission_id)
    call = cap.calls.pop(0)
    return _roundtrip(call["kwargs"]["cont_state"])


@pytest.mark.asyncio
async def test_prompts_done_dict_result_enqueues_first_image(ws, cap):
    state = await _kickoff(ws, cap)
    await _on_prompts_done(101, dict(_PROMPTS_3), state)

    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "images_pending"
    assert ledger["prompt_map"]["home__0"] == "coral barista scene"

    # Sequential chain: exactly ONE image child enqueued.
    assert len(cap.calls) == 1
    call = cap.calls[0]
    spec, kwargs = call["spec"], call["kwargs"]
    assert spec["agent_type"] == "image"
    assert spec["mission_id"] == 42
    ic = spec["context"]["image_call"]
    assert ic["raw_dispatch"] is True
    assert ic["filename_hint"] == "home__0"
    assert ic["out_dir"] == os.path.join(str(ws), ".web", "assets")
    assert kwargs["on_complete"] == ON_IMAGE_DONE
    assert kwargs["on_error"] == ON_IMAGE_ERR
    assert kwargs["cont_state"] == {
        "mission_id": 42, "workspace_path": str(ws), "pid": "home__0",
    }


@pytest.mark.asyncio
async def test_prompts_done_json_string_result(ws, cap):
    """Defensive: result body arrives as a JSON string / nested envelope."""
    state = await _kickoff(ws, cap)
    await _on_prompts_done(101, {"result": json.dumps(_PROMPTS_3)}, state)
    assert _load_ledger(str(ws))["status"] == "images_pending"
    assert len(cap.calls) == 1


@pytest.mark.asyncio
async def test_prompts_done_garbage_degrades(ws, cap):
    """No usable prompts after repair → finalize-with-degrade: every
    placeholder skipped, HTML untouched (placehold.co URLs intact)."""
    state = await _kickoff(ws, cap)
    await _on_prompts_done(101, {"content": "total garbage"}, state)

    assert cap.calls == []  # no image children
    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "done"
    assert ledger["replaced"] == 0
    assert ledger["skipped"] == 3
    assert all(v["status"] == "skipped" for v in ledger["results"].values())
    html = (ws / ".web" / "home.html").read_text(encoding="utf-8")
    assert html.count("placehold.co") == 3


@pytest.mark.asyncio
async def test_prompts_err_degrades(ws, cap):
    state = await _kickoff(ws, cap)
    await _on_prompts_err(101, {"status": "failed", "error": "LLM down"}, state)
    assert cap.calls == []
    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "done"
    assert ledger["skipped"] == 3
    assert any("LLM down" in (v.get("error") or "")
               for v in ledger["results"].values())


@pytest.mark.asyncio
async def test_prompts_done_without_ledger_is_noop(tmp_path, cap):
    await _on_prompts_done(
        101, dict(_PROMPTS_3),
        {"mission_id": 1, "workspace_path": str(tmp_path)},
    )
    assert cap.calls == []


# -- image_done / image_err handlers ----------------------------------------

@pytest.mark.asyncio
async def test_image_done_renames_records_advances(ws, cap):
    state = await _kickoff(ws, cap)
    await _on_prompts_done(101, dict(_PROMPTS_3), state)
    img_call = cap.calls.pop(0)
    ic = img_call["spec"]["context"]["image_call"]

    # Simulate paintress writing the timestamp-suffixed PNG.
    raw = os.path.join(ic["out_dir"], f"{ic['filename_hint']}_mock0.png")
    _write_real_png(raw, ic["width"], ic["height"])

    await _on_image_done(
        img_call["id"], {"path": raw, "content": raw},
        _roundtrip(img_call["kwargs"]["cont_state"]),
    )

    # Renamed to stable <pid>.png.
    assert os.path.isfile(os.path.join(ic["out_dir"], "home__0.png"))
    assert not os.path.exists(raw)
    ledger = _load_ledger(str(ws))
    assert ledger["results"]["home__0"] == {"status": "done",
                                            "asset": "home__0.png"}
    # Advanced: next image child enqueued.
    assert len(cap.calls) == 1
    assert cap.calls[0]["spec"]["context"]["image_call"]["filename_hint"] == \
        "home__1"


@pytest.mark.asyncio
async def test_image_done_no_path_records_error_and_advances(ws, cap):
    state = await _kickoff(ws, cap)
    await _on_prompts_done(101, dict(_PROMPTS_3), state)
    img_call = cap.calls.pop(0)
    await _on_image_done(
        img_call["id"], {"status": "completed"},
        _roundtrip(img_call["kwargs"]["cont_state"]),
    )
    ledger = _load_ledger(str(ws))
    assert ledger["results"]["home__0"]["status"] == "error"
    assert len(cap.calls) == 1  # chain advanced


@pytest.mark.asyncio
async def test_image_err_records_and_advances(ws, cap):
    state = await _kickoff(ws, cap)
    await _on_prompts_done(101, dict(_PROMPTS_3), state)
    img_call = cap.calls.pop(0)
    await _on_image_err(
        img_call["id"], {"status": "failed", "error": "rate-limit"},
        _roundtrip(img_call["kwargs"]["cont_state"]),
    )
    ledger = _load_ledger(str(ws))
    entry = ledger["results"]["home__0"]
    assert entry["status"] == "error"
    assert "rate-limit" in entry["error"]
    assert len(cap.calls) == 1  # advanced to home__1


# -- finalize ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_finalize_rewrites_html(ws, cap):
    """Finalize applies the recorded rewrites: done pids → assets/<pid>.png,
    failed pid keeps its placehold.co URL, real src untouched."""
    state = await _kickoff(ws, cap)
    ledger = _load_ledger(str(ws))
    assets = os.path.join(str(ws), ".web", "assets")
    for pid in ("home__0", "home__2"):
        _write_real_png(os.path.join(assets, f"{pid}.png"))
        ledger["results"][pid] = {"status": "done", "asset": f"{pid}.png"}
    ledger["results"]["home__1"] = {"status": "error",
                                    "error": "image gen failed for home__1"}
    await _finalize(str(ws), ledger)

    html = (ws / ".web" / "home.html").read_text(encoding="utf-8")
    assert html.count('src="assets/home__0.png"') == 1
    assert html.count('src="assets/home__2.png"') == 1
    assert html.count("placehold.co") == 1  # failed pid survives
    assert "/assets/already_real.png" in html  # untouched real src

    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "done"
    assert ledger["replaced"] == 2
    assert ledger["skipped"] == 1
    assert ledger["html_files_changed"] == 1
    assert ledger["shape_check"]["ok"] is True
    assert ledger["shape_check"]["surviving_placeholders"] == 1
    assert any("image gen failed" in e for e in ledger["errors"])
    _ = state  # silence unused


@pytest.mark.asyncio
async def test_double_finalize_is_noop(ws, cap):
    """FIX 1.2b: finalize must be idempotent. A second _finalize on a
    status=done ledger must NOT re-splice the (now stale) scan-time spans
    into the already-rewritten HTML."""
    await _drive_chain(ws, cap)
    html_after = (ws / ".web" / "home.html").read_text(encoding="utf-8")
    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "done"

    await _finalize(str(ws), ledger)

    assert (ws / ".web" / "home.html").read_text(encoding="utf-8") == html_after
    assert _load_ledger(str(ws)) == ledger


@pytest.mark.asyncio
async def test_stale_span_skipped_not_spliced(ws, cap):
    """FIX 1.2c: if the HTML changed between scan and finalize, splicing the
    recorded spans would corrupt the file. Each span is verified against the
    recorded tag text; mismatches are skipped (placehold.co URL survives —
    graceful degrade) with the error recorded in the ledger."""
    state = await _kickoff(ws, cap)
    await _on_prompts_done(101, dict(_PROMPTS_3), state)

    # Mutate the HTML mid-flight: lengthen the SECOND placeholder's tag.
    # home__0 (before the edit) keeps a valid span; home__1's own tag text
    # changed; home__2's span shifted.
    p = ws / ".web" / "home.html"
    html = p.read_text(encoding="utf-8")
    p.write_text(html.replace("text=feat", "text=feature"), encoding="utf-8")

    # Drive the image children to done as usual.
    guard = 0
    while cap.calls:
        guard += 1
        assert guard < 10
        call = cap.calls.pop(0)
        ic = call["spec"]["context"]["image_call"]
        raw = os.path.join(ic["out_dir"], f"{ic['filename_hint']}_raw.png")
        _write_real_png(raw, ic["width"], ic["height"])
        await _on_image_done(
            call["id"], {"path": raw, "content": raw},
            _roundtrip(call["kwargs"]["cont_state"]),
        )

    rewritten = p.read_text(encoding="utf-8")
    # home__0: valid span → spliced.
    assert rewritten.count('src="assets/home__0.png"') == 1
    # home__1 + home__2: stale spans → NOT spliced, placehold.co survives.
    assert rewritten.count("placehold.co") == 2
    assert 'src="assets/home__1.png"' not in rewritten
    assert 'src="assets/home__2.png"' not in rewritten
    # No corrupt splice: the document structure is intact.
    assert rewritten.count("<img") == 4
    assert "/assets/already_real.png" in rewritten

    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "done"
    assert ledger["replaced"] == 1
    assert ledger["skipped"] == 2
    assert ledger["results"]["home__1"]["status"] == "skipped"
    assert ledger["results"]["home__2"]["status"] == "skipped"
    stale_errors = [e for e in ledger["errors"] if "stale span" in e]
    assert any("home__1" in e for e in stale_errors)
    assert any("home__2" in e for e in stale_errors)


# -- full chain --------------------------------------------------------------

async def _drive_chain(ws, cap, *, fail_pids=frozenset(), prompts=None):
    """Drive kickoff → prompts_done → image_done/err until finalize."""
    state = await _kickoff(ws, cap)
    await _on_prompts_done(
        101, prompts if prompts is not None else dict(_PROMPTS_3), state,
    )
    guard = 0
    while cap.calls:
        guard += 1
        assert guard < 50
        call = cap.calls.pop(0)
        ic = call["spec"]["context"]["image_call"]
        st = _roundtrip(call["kwargs"]["cont_state"])
        if ic["filename_hint"] in fail_pids:
            await _on_image_err(
                call["id"], {"status": "failed", "error": "rate-limit"}, st,
            )
            continue
        raw = os.path.join(ic["out_dir"], f"{ic['filename_hint']}_raw.png")
        _write_real_png(raw, ic["width"], ic["height"])
        await _on_image_done(call["id"], {"path": raw, "content": raw}, st)


@pytest.mark.asyncio
async def test_full_chain_writes_assets_and_rewrites_html(ws, cap):
    await _drive_chain(ws, cap)

    assets = ws / ".web" / "assets"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    assert pngs == ["home__0.png", "home__1.png", "home__2.png"]

    rewritten = (ws / ".web" / "home.html").read_text(encoding="utf-8")
    assert "placehold.co" not in rewritten
    assert rewritten.count('src="assets/home__0.png"') == 1
    assert rewritten.count('src="assets/home__1.png"') == 1
    assert rewritten.count('src="assets/home__2.png"') == 1
    assert "/assets/already_real.png" in rewritten

    ledger = _load_ledger(str(ws))
    assert ledger["status"] == "done"
    assert ledger["replaced"] == 3
    assert ledger["skipped"] == 0
    assert ledger["errors"] == []
    assert ledger["shape_check"]["ok"] is True


@pytest.mark.asyncio
async def test_full_chain_per_image_failure_keeps_placeholder(ws, cap):
    await _drive_chain(ws, cap, fail_pids={"home__1"})
    rewritten = (ws / ".web" / "home.html").read_text(encoding="utf-8")
    assert rewritten.count("placehold.co") == 1
    assert rewritten.count('src="assets/') == 2
    ledger = _load_ledger(str(ws))
    assert ledger["replaced"] == 2
    assert ledger["skipped"] == 1
    assert ledger["shape_check"]["ok"] is True


@pytest.mark.asyncio
async def test_full_chain_subdir_html_gets_relative_dotdot_ref(
    monkeypatch, tmp_path, cap,
):
    """Multi-screen: root HTML → "assets/<pid>.png"; subdir screen →
    "../assets/<pid>.png" (flat ref would 404 in a static server)."""
    web = tmp_path / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "home.html").write_text(
        '<html><body>'
        '<img src="https://placehold.co/390x220/E07A5F/FFF?text=hero" alt="hero">'
        '</body></html>',
        encoding="utf-8",
    )
    (web / "screens" / "onboarding.html").write_text(
        '<html><body>'
        '<img src="https://placehold.co/64x64/264653/FFF?text=u" alt="user portrait">'
        '</body></html>',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    prompts = {
        "prompts": [
            {"placeholder_id": "home__0", "prompt": "coral barista"},
            {"placeholder_id": "onboarding__0", "prompt": "teal portrait"},
        ],
    }
    await _drive_chain(tmp_path, cap, prompts=prompts)

    assets = web / "assets"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    assert pngs == ["home__0.png", "onboarding__0.png"]

    home = (web / "home.html").read_text(encoding="utf-8")
    assert 'src="assets/home__0.png"' in home
    onboarding = (web / "screens" / "onboarding.html").read_text(encoding="utf-8")
    assert 'src="../assets/onboarding__0.png"' in onboarding
    assert 'src="assets/onboarding__0.png"' not in onboarding


@pytest.mark.asyncio
async def test_image_task_specs_carry_mission_id(ws, cap):
    """compute_task_hash includes mission_id — without it two concurrent
    missions with same-named HTML files share a dedup hash and the 2nd
    mission's child collapses onto the 1st's."""
    state = await _kickoff(ws, cap, mission_id=99)
    await _on_prompts_done(101, dict(_PROMPTS_3), state)
    seen = []
    while cap.calls:
        call = cap.calls.pop(0)
        seen.append(call["spec"])
        await _on_image_err(
            call["id"], {"status": "failed", "error": "x"},
            _roundtrip(call["kwargs"]["cont_state"]),
        )
    assert seen, "no image task specs captured — chain did not start"
    for spec in seen:
        assert spec.get("mission_id") == 99


# -- registration -------------------------------------------------------------

def test_handlers_registered_at_import():
    from general_beckman.continuations import _HANDLERS
    swap_mod.register_continuations()
    assert _HANDLERS[ON_PROMPTS_DONE] is _on_prompts_done
    assert _HANDLERS[ON_PROMPTS_ERR] is _on_prompts_err
    assert _HANDLERS[ON_IMAGE_DONE] is _on_image_done
    assert _HANDLERS[ON_IMAGE_ERR] is _on_image_err
