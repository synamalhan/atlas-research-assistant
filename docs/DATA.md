# Data

Only the ID list (`configs/papers.txt`) is committed. PDFs are **not** in the repo; run
`make download` to fetch them into `data/raw/<id>.pdf` (git-ignored). Each ID is pinned to a
version, so everyone gets the same file.

**Licensing:** arXiv papers carry per-paper licenses. Check each paper's arXiv abstract page
before redistributing any PDF or derived content (text, figures, tables). The `license:` field
in `papers.txt` is a reminder to verify, not a verified value.

The download script compares each paper's arXiv title with the title in `papers.txt` and exits
non-zero on any missing paper, failed download or title mismatch.
