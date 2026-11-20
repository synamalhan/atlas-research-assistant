from atlas.chunk import chunk_paper


def item(paper, n, text, page=1, section="S1"):
    return {"id": f"{paper}:text:{n}", "paper_id": paper, "type": "text", "page": page,
            "section": section, "content": text}


def blocks(paper="p1", n=4):
    # each 20 chars; two joined with "\n\n" = 42 chars
    return [item(paper, i, f"Block {i}: Mixed 3.14 Ab", page=i + 1) for i in range(n)]


def test_overlap_and_sources():
    chunks = chunk_paper(blocks(), size=50, overlap=25)
    assert [c["source_ids"] for c in chunks] == [
        ["p1:text:0", "p1:text:1"], ["p1:text:1", "p1:text:2"], ["p1:text:2", "p1:text:3"]]
    assert chunks[0]["pages"] == [1, 2] and chunks[1]["page"] == 2
    assert chunks[1]["text"].startswith("Block 1: Mixed 3.14 Ab")  # overlap block is verbatim


def test_no_overlap_when_zero():
    chunks = chunk_paper(blocks(), size=50, overlap=0)
    assert [c["source_ids"] for c in chunks] == [
        ["p1:text:0", "p1:text:1"], ["p1:text:2", "p1:text:3"]]


def test_never_crosses_sections_or_papers():
    items = blocks("p1", 2)
    items[1]["section"] = "S2"
    chunks = chunk_paper(items, size=1000, overlap=0)
    assert len(chunks) == 2 and chunks[0]["section"] == "S1" and chunks[1]["section"] == "S2"
    a = chunk_paper(blocks("p1", 2), 1000, 0)
    b = chunk_paper(blocks("p2", 2), 1000, 0)
    assert {s.split(":")[0] for c in a for s in c["source_ids"]} == {"p1"}
    assert b[0]["id"] == "p2:chunk:0"


def test_deterministic_ids_and_verbatim_text():
    a = chunk_paper(blocks(), 50, 25)
    assert a == chunk_paper(blocks(), 50, 25)
    assert [c["id"] for c in a] == ["p1:chunk:0", "p1:chunk:1", "p1:chunk:2"]
    assert a[0]["text"] == "Block 0: Mixed 3.14 Ab\n\nBlock 1: Mixed 3.14 Ab"


def test_oversized_block_kept_whole_and_non_text_ignored():
    big = item("p1", 0, "x" * 500)
    fig = {**item("p1", 1, "cap"), "type": "figure"}
    chunks = chunk_paper([big, fig], size=100, overlap=20)
    assert len(chunks) == 1 and chunks[0]["text"] == "x" * 500
