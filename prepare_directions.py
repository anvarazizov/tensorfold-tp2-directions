"""Convert the pinned pqhaz projected F32 tensor using only the standard library."""
import argparse
import hashlib
from pathlib import Path
import struct
from zipfile import ZipFile

SOURCE = "87e81bffab36b046786eb7a3888c8d12c69ddef145235e363e0d4fb2c8fe31b4"
PROJECTED = "98d65378f2167fed790b92ffb8c6b9aa2ac866b128126b0efc7803e10108ea13"
OUTPUT = "097fff6bf88de26730240d1bc20747fb63cc488db050807a6e8efce7452d763c"


def verify(data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch, expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.pt.open("rb") as source:
        verify(source.read(), SOURCE)
        # The storage mapping was inspected for this exact archive. No pickle execution.
        with ZipFile(source) as archive:
            projected = archive.read("directions/data/3")
    verify(projected, PROJECTED)
    skip = 2 * 5120 * 4
    mask = sum(1 << layer for layer in range(2, 40))
    packet = b"TFDIR001" + struct.pack("<Q", mask) + bytes(skip) + projected[skip:]
    verify(packet, OUTPUT)
    with args.output.open("xb") as out:
        out.write(packet)
    print(OUTPUT, args.output)


if __name__ == "__main__":
    main()
