import os

from mr_roboto.emit_preview_url import _resolve_preview_root


def test_web_root_with_assets_subdir(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text("<html></html>", encoding="utf-8")
    assets = web / "assets"; assets.mkdir()
    (assets / "home__0.png").write_bytes(b"\x89PNG")
    root = _resolve_preview_root(str(tmp_path))
    assert root == str(web)
    assert os.path.isfile(os.path.join(root, "assets", "home__0.png"))


def test_web_root_with_only_assets(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "assets").mkdir()
    (web / "assets" / "ghost.png").write_bytes(b"\x89PNG")
    root = _resolve_preview_root(str(tmp_path))
    assert root == str(web)
