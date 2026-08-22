import os
from pathlib import Path


_TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_ROOT / "arxiv_binn_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_ROOT / "arxiv_binn_cache"))
