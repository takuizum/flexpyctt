"""The documentation site: code in the pages runs, and the two languages say the same."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

import flexpyctt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BLOCK_RE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)
PAGES = sorted(p.name.removesuffix(".md") for p in DOCS.glob("*.md") if ".ja." not in p.name)
RUNNABLE = ["index", "guide"]
DOCS_SITE = "https://takuizum.github.io/flexpyctt/"


def _blocks(path: Path) -> list[str]:
    return BLOCK_RE.findall(path.read_text(encoding="utf-8"))


def _code_only(block: str) -> list[str]:
    """Return the block's lines without comments and blank lines."""
    lines = [re.sub(r"\s+#.*$", "", line).rstrip() for line in block.splitlines()]
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


def test_every_page_has_a_japanese_version() -> None:
    assert {"index", "guide", "reference"} <= set(PAGES)
    for page in PAGES:
        assert (DOCS / f"{page}.ja.md").exists(), page


@pytest.mark.parametrize("page", RUNNABLE)
@pytest.mark.parametrize("suffix", [".md", ".ja.md"])
def test_code_in_the_page_runs(page: str, suffix: str, capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks(DOCS / f"{page}{suffix}")
    assert blocks, f"{page}{suffix} has no python blocks"
    namespace: dict[str, object] = {"__name__": f"docs:{page}{suffix}"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flexpyctt.errors.FlexpycttWarning)
        for code in blocks:
            exec(compile(code, f"{page}{suffix}", "exec"), namespace)
    assert capsys.readouterr().out


@pytest.mark.parametrize("page", RUNNABLE)
def test_languages_have_the_same_code(page: str) -> None:
    english, japanese = (
        [_code_only(b) for b in _blocks(DOCS / f"{page}{suffix}")] for suffix in (".md", ".ja.md")
    )
    assert english == japanese


@pytest.mark.parametrize("suffix", [".md", ".ja.md"])
def test_reference_has_an_entry_for_every_public_name(suffix: str) -> None:
    text = (DOCS / f"reference{suffix}").read_text(encoding="utf-8")
    entries = set(re.findall(r"^### `([^`]+)`", text, re.M))
    public = {name for name in flexpyctt.__all__ if name not in ("__version__", "errors")}
    assert public <= entries, sorted(public - entries)
    assert "flexpyctt.errors" in entries
    for name in flexpyctt.errors.__all__:
        assert f"#### `{name}`" in text, name


@pytest.mark.parametrize("page", ["index", "guide", "reference"])
def test_japanese_pages_do_not_link_to_english_pages(page: str) -> None:
    text = (DOCS / f"{page}.ja.md").read_text(encoding="utf-8")
    switcher = f"[English]({page}.md) | 日本語"
    assert text.count(switcher) == 1
    relative = re.findall(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)", text.replace(switcher, ""))
    assert all(link.endswith(".ja.md") for link in relative), relative
    site_links = re.findall(rf"\]\(({re.escape(DOCS_SITE)}[^)\s]*)\)", text)
    assert all(link.startswith(DOCS_SITE + "ja/") for link in site_links), site_links
