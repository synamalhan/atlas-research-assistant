"""Download pinned arXiv PDFs listed in configs/papers.txt and verify titles."""
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://export.arxiv.org/api/query"
PDF = "https://arxiv.org/pdf/{}"
LINE = re.compile(r"^(\S+)\s*#\s*(.+?)\s*\|\s*license:\s*(.+)$")
ATOM = "{http://www.w3.org/2005/Atom}"
DELAY = 3.0  # seconds between requests (arXiv asks for >= 3s)
RETRIES = 4
HEADERS = {"User-Agent": "atlas-research-assistant/0.1 (research corpus downloader)"}
_last = 0.0


def fetch(url: str, params: dict | None = None) -> bytes:
    """GET with politeness delay and exponential backoff on 429/5xx/network errors."""
    global _last
    for attempt in range(RETRIES + 1):
        time.sleep(max(0.0, _last + DELAY - time.monotonic()))
        _last = time.monotonic()
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == RETRIES:
                raise
            time.sleep(DELAY * 2**attempt)
    raise AssertionError("unreachable")


def parse_papers(path: Path) -> list[tuple[str, str, str]]:
    papers = []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            raise ValueError(f"{path}:{n}: cannot parse line: {raw!r}")
        papers.append(m.groups())
    return papers


def norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def fetch_titles(ids: list[str]) -> dict[str, str]:
    """Return {id_with_version: title} for ids arXiv knows about."""
    titles = {}
    for i in range(0, len(ids), 20):
        batch = ids[i : i + 20]
        xml = fetch(API, {"id_list": ",".join(batch), "max_results": len(batch)})
        for e in ET.fromstring(xml).iter(f"{ATOM}entry"):
            url = e.findtext(f"{ATOM}id", "")
            if "/api/errors" in url:
                continue
            titles[url.rsplit("/abs/", 1)[-1]] = " ".join(e.findtext(f"{ATOM}title", "").split())
    return titles


def run(papers_file: Path, out_dir: Path) -> int:
    papers = parse_papers(papers_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        titles = fetch_titles([p[0] for p in papers])
    except requests.RequestException as e:
        print(f"ERROR: could not fetch arXiv metadata: {e}")
        return 1
    done, skipped, failed, mismatched = [], [], [], []
    for pid, want, _license in papers:
        got = titles.get(pid)
        if got is None:
            failed.append((pid, "not found in arXiv metadata"))
            continue
        if norm(got) != norm(want):
            mismatched.append((pid, want, got))
        dest = out_dir / f"{pid}.pdf"
        if dest.exists():
            skipped.append(pid)
            continue
        try:
            data = fetch(PDF.format(pid))
        except requests.RequestException as e:
            failed.append((pid, f"download failed: {e}"))
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dest)  # never leave a partial file that looks complete
        done.append(pid)
        print(f"downloaded {pid}")
    print(
        f"\nSummary: downloaded={len(done)} skipped={len(skipped)} "
        f"failed={len(failed)} mismatched={len(mismatched)}"
    )
    for pid, why in failed:
        print(f"FAILED   {pid}: {why}")
    for pid, want, got in mismatched:
        print(f"MISMATCH {pid}:\n  expected: {want}\n  arXiv:    {got}")
    return 1 if failed or mismatched else 0


if __name__ == "__main__":
    sys.exit(run(ROOT / "configs" / "papers.txt", ROOT / "data" / "raw"))
