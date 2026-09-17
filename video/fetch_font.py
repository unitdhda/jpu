#!/usr/bin/env python3
"""Fetch the OFL-licensed Noto Sans JP font used by the video."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
EXPECTED = "c2f3b4d463500a2ddcd3849cded1fceeb9fd6d1c32e6cbecd568453ba50fc68f"
TARGET = Path(__file__).parent / "assets/fonts/NotoSansJP-Variable.ttf"

payload = urlopen(URL).read()
actual = sha256(payload).hexdigest()
if actual != EXPECTED:
    raise SystemExit(f"Noto Sans JP checksum mismatch: {actual}")
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_bytes(payload)
print(TARGET)
