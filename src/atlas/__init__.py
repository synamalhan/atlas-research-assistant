import os

# torch (open_clip, sentence-transformers) and faiss each bundle libomp on macOS; loading both
# crashes (OMP error #15 / segfault) unless duplicates are allowed and threads are limited.
# Must be set before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
