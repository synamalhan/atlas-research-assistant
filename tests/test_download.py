import download_papers as dp

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">{}</feed>"""
ENTRY = "<entry><id>http://arxiv.org/abs/{}</id><title>{}</title></entry>"


def make_fetch(titles):
    def fake_fetch(url, params=None):
        if url == dp.API:
            return ATOM.format("".join(ENTRY.format(i, t) for i, t in titles.items())).encode()
        return b"%PDF-fake"

    return fake_fetch


def write_list(tmp_path, *lines):
    p = tmp_path / "papers.txt"
    p.write_text("# section\n" + "\n".join(lines) + "\n")
    return p


def test_parse(tmp_path):
    p = write_list(tmp_path, "2401.00001v1  # A Title: Sub | license: x")
    assert dp.parse_papers(p) == [("2401.00001v1", "A Title: Sub", "x")]


def test_success_skip_mismatch_missing(tmp_path, monkeypatch, capsys):
    papers = write_list(
        tmp_path,
        "1.1v1  # Good Paper | license: x",
        "2.2v1  # Already Here | license: x",
        "3.3v1  # Expected Title | license: x",
        "4.4v1  # Ghost | license: x",
    )
    out = tmp_path / "raw"
    out.mkdir()
    (out / "2.2v1.pdf").write_bytes(b"old")
    titles = {"1.1v1": "Good  Paper", "2.2v1": "Already Here", "3.3v1": "Different Title"}
    monkeypatch.setattr(dp, "fetch", make_fetch(titles))
    assert dp.run(papers, out) == 1
    assert (out / "1.1v1.pdf").read_bytes() == b"%PDF-fake"
    assert (out / "2.2v1.pdf").read_bytes() == b"old"  # not re-downloaded
    text = capsys.readouterr().out
    assert "downloaded=2 skipped=1 failed=1 mismatched=1" in text
    assert "MISMATCH 3.3v1" in text and "FAILED   4.4v1" in text


def test_all_good_exits_zero(tmp_path, monkeypatch):
    papers = write_list(tmp_path, "1.1v1  # Good | license: x")
    monkeypatch.setattr(dp, "fetch", make_fetch({"1.1v1": "Good"}))
    assert dp.run(papers, tmp_path / "raw") == 0
