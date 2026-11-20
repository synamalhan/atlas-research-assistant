"""Parse all PDFs in data/raw into data/processed; exit non-zero if any paper failed."""
import sys
from pathlib import Path

from atlas.parse import parse_all

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    report = parse_all(ROOT / "data" / "raw", ROOT / "data" / "processed")
    print(f"parsed ok: {report['ok']}  failed: {len(report['failed'])}")
    for f in report["failed"]:
        print(f"FAILED {f['paper_id']}: {f['error']}  salvaged={f['salvaged']}")
    if report["failed"]:
        print("details: data/processed/parse_report.json")
    sys.exit(1 if report["failed"] else 0)
