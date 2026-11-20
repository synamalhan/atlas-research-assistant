import json

import pymupdf
import pytest

from atlas import parse


def make_pdf(path, header="Conf 2026 Header"):
    """Two pages, one column; known heading/body/figure/table/caption text."""
    doc = pymupdf.open()
    # a tiny red PNG used as the figure
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 90), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    for n in (1, 2):
        page = doc.new_page()
        page.insert_text((72, 30), header, fontsize=8)
        page.insert_text((300, 825), str(n), fontsize=8)
    p1, p2 = doc[0], doc[1]
    p1.insert_text((72, 90), "1 Introduction", fontsize=16, fontname="hebo")
    p1.insert_textbox(pymupdf.Rect(72, 110, 520, 160),
                      "We got 42.5% accuracy on Foo-Bench. Results Are Mixed.", fontsize=10)
    p1.insert_image(pymupdf.Rect(72, 200, 252, 335), pixmap=pix)
    p1.insert_textbox(pymupdf.Rect(72, 340, 520, 370), "Figure 1: A red square.", fontsize=9)
    p2.insert_text((72, 90), "2 Results", fontsize=16, fontname="hebo")
    p2.insert_text((72, 130), "Table 1: Scores per model.", fontsize=9)
    x, y = 72, 145
    for r in range(3):
        for c in range(3):
            rect = pymupdf.Rect(x + c * 80, y + r * 24, x + (c + 1) * 80, y + (r + 1) * 24)
            p2.draw_rect(rect)
            p2.insert_text((rect.x0 + 4, rect.y0 + 16), f"r{r}c{c}", fontsize=9)
    p2.insert_textbox(pymupdf.Rect(72, 260, 520, 300), "Another figure-less paragraph.",
                      fontsize=10)
    doc.save(path)


def test_items_ids_pages_captions_sections(tmp_path):
    make_pdf(tmp_path / "2401.00001v1.pdf")
    items, errors, n = parse.parse_pdf(tmp_path / "2401.00001v1.pdf", tmp_path / "figs")
    assert errors == [] and n == 2
    by = {t: [i for i in items if i["type"] == t] for t in ("text", "figure", "table")}
    assert all(i["id"] == f"2401.00001v1:{i['type']}:{k}"
               for t in by for k, i in enumerate(by[t]))
    texts = {i["content"]: i for i in by["text"]}
    # verbatim: case, digits and punctuation untouched
    body = next(t for t in texts if "42.5%" in t)
    assert "Foo-Bench. Results Are Mixed." in body
    assert texts["1 Introduction"]["section"] == "1 Introduction"
    assert texts[body]["section"] == "1 Introduction" and texts[body]["page"] == 1
    assert texts["Another figure-less paragraph."]["section"] == "2 Results"
    assert not any("Conf 2026" in t or t.strip() in ("1", "2") for t in texts)  # header/page no.
    (fig,) = by["figure"]
    assert fig["page"] == 1 and fig["caption"] == "Figure 1: A red square."
    assert (tmp_path / "figs" / "2401.00001v1_0.png").exists()
    (tab,) = by["table"]
    assert tab["page"] == 2 and tab["caption"] == "Table 1: Scores per model."
    assert tab["quality"] == "ok" and "r1c2" in tab["content"]
    assert not any("r1c2" in t for t in texts)  # cells not duplicated as text
    assert fig["source_version"] == "v1"


def test_figure_without_caption(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 90), False)
    pix.set_rect(pix.irect, (0, 0, 200))
    page.insert_image(pymupdf.Rect(72, 100, 252, 235), pixmap=pix)
    page.insert_text((72, 400), "Some text.", fontsize=10)
    doc.save(tmp_path / "a.pdf")
    items, _, _ = parse.parse_pdf(tmp_path / "a.pdf", tmp_path / "f")
    (fig,) = [i for i in items if i["type"] == "figure"]
    assert fig["caption"] is None


def test_two_column_reading_order(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 100, 280, 140), "LEFT-TOP", fontsize=10)
    page.insert_textbox(pymupdf.Rect(50, 300, 280, 340), "LEFT-BOTTOM", fontsize=10)
    page.insert_textbox(pymupdf.Rect(320, 100, 550, 140), "RIGHT-TOP", fontsize=10)
    page.insert_textbox(pymupdf.Rect(320, 300, 550, 340), "RIGHT-BOTTOM", fontsize=10)
    doc.save(tmp_path / "c.pdf")
    items, _, _ = parse.parse_pdf(tmp_path / "c.pdf", tmp_path / "f")
    order = [i["content"] for i in items if i["type"] == "text"]
    assert order == ["LEFT-TOP", "LEFT-BOTTOM", "RIGHT-TOP", "RIGHT-BOTTOM"]


def test_ids_stable_across_runs(tmp_path):
    make_pdf(tmp_path / "2401.00001v1.pdf")
    a = parse.parse_pdf(tmp_path / "2401.00001v1.pdf", tmp_path / "f1")[0]
    b = parse.parse_pdf(tmp_path / "2401.00001v1.pdf", tmp_path / "f2")[0]
    strip = lambda items: [{**i, "image_path": None} for i in items]  # noqa: E731
    assert strip(a) == strip(b)


def test_failures_reported_not_raised(tmp_path, capsys):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    make_pdf(raw / "2401.00001v1.pdf")
    (raw / "2401.00002v1.pdf").write_bytes(b"not a pdf at all")
    (raw / "2401.00003v1.pdf").write_bytes(b"")
    report = parse.parse_all(raw, out, workers=1)
    assert report["ok"] == 1
    assert {f["paper_id"] for f in report["failed"]} == {"2401.00002v1", "2401.00003v1"}
    assert all(f["error"] for f in report["failed"])
    on_disk = json.loads((out / "parse_report.json").read_text())
    assert on_disk == report
    assert (out / "2401.00001v1.jsonl").exists()
    assert not (out / "2401.00002v1.jsonl").exists()


def test_jsonl_fields(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    make_pdf(raw / "2401.00001v1.pdf")
    parse.parse_all(raw, out, workers=1)
    line = json.loads((out / "2401.00001v1.jsonl").read_text().splitlines()[0])
    assert set(line) == {"id", "paper_id", "type", "page", "bbox", "section", "content",
                         "image_path", "caption", "quality", "source_version"}


def test_inspect_flags(tmp_path, capsys):
    import inspect_parse

    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    make_pdf(raw / "2401.00001v1.pdf")
    parse.parse_all(raw, out, workers=1)
    inspect_parse.table(inspect_parse.load(out), raw)
    text = capsys.readouterr().out
    assert "2401.00001v1" in text and "NO-FIGURES" not in text
    inspect_parse.sample(inspect_parse.load(out)["2401.00001v1"], 3)
    assert "Figure 1: A red square." in capsys.readouterr().out
    with pytest.raises(KeyError):
        inspect_parse.load(out)["missing"]
