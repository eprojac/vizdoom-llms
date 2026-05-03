#!/usr/bin/env python3
"""Generate the tiny MAP01 WAD used by scenarios/basic.cfg."""

from __future__ import annotations

import struct
import sys
from pathlib import Path


TEXTMAP = r"""
namespace = "ZDoom";

thing { x = -384.0; y = 0.0; angle = 0; type = 1; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }
thing { x = 384.0; y = 0.0; angle = 180; type = 2; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }
thing { x = -384.0; y = 0.0; angle = 0; type = 11; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }
thing { x = 384.0; y = 0.0; angle = 180; type = 11; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }

thing { x = 0.0; y = 0.0; angle = 0; type = 2012; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }
thing { x = 0.0; y = -248.0; angle = 90; type = 2001; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }
thing { x = -72.0; y = -248.0; angle = 90; type = 2008; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }
thing { x = 72.0; y = -248.0; angle = 90; type = 2008; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }
thing { x = 0.0; y = 248.0; angle = 270; type = 2018; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; coop = true; dm = true; }

vertex { x = -512.0; y = -320.0; }
vertex { x = -512.0; y = 320.0; }
vertex { x = 512.0; y = 320.0; }
vertex { x = 512.0; y = -320.0; }

linedef { v1 = 0; v2 = 1; sidefront = 0; blocking = true; }
linedef { v1 = 1; v2 = 2; sidefront = 1; blocking = true; }
linedef { v1 = 2; v2 = 3; sidefront = 2; blocking = true; }
linedef { v1 = 3; v2 = 0; sidefront = 3; blocking = true; }

sidedef { sector = 0; texturemiddle = "STARTAN3"; }
sidedef { sector = 0; texturemiddle = "STARTAN3"; }
sidedef { sector = 0; texturemiddle = "STARTAN3"; }
sidedef { sector = 0; texturemiddle = "STARTAN3"; }

sector { heightfloor = 0; heightceiling = 128; texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1"; lightlevel = 192; }
""".strip() + "\n"


def make_wad(path: Path) -> None:
    lumps = [
        ("MAP01", b""),
        ("TEXTMAP", TEXTMAP.encode("ascii")),
        ("ENDMAP", b""),
    ]

    data_offset = 12
    payload = bytearray()
    directory = bytearray()
    offset = data_offset

    for name, data in lumps:
        payload.extend(data)
        directory.extend(struct.pack("<II8s", offset, len(data), name.encode("ascii").ljust(8, b"\0")))
        offset += len(data)

    header = struct.pack("<4sII", b"PWAD", len(lumps), data_offset + len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload + directory)


def main() -> None:
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    else:
        out = Path(__file__).resolve().parents[1] / "scenarios" / "basic_1v1.wad"
    make_wad(out)
    print(out)


if __name__ == "__main__":
    main()
