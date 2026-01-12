#!/usr/bin/env python3
import struct
import sys
from pathlib import Path

OFF = 32  # lambda0 int16 (Q8.8) offset in model.bin

def main():
    if len(sys.argv) != 4:
        print('Usage: patch_lambda0_q.py IN_MODEL OUT_MODEL Q', file=sys.stderr)
        print('  lambda0 = Q/256 (Q is int16)', file=sys.stderr)
        sys.exit(2)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    q = int(sys.argv[3])
    if q < -32768 or q > 32767:
        raise ValueError('Q out of int16 range')

    data = bytearray(in_path.read_bytes())
    data[OFF:OFF+2] = struct.pack('<h', q)
    out_path.write_bytes(data)

    print(f'patched: {in_path} -> {out_path}')
    print(f'lambda0 = {q}/256 = {q/256.0:.6f}')

if __name__ == '__main__':
    main()
