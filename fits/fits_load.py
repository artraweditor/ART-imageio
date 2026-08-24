#!/usr/bin/env python3

import subprocess
import argparse
import tempfile
import os


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', nargs='?', default=0, type=int)
    p.add_argument('height', nargs='?', default=0, type=int)
    return p.parse_args()


def main():
    opts = getopts()
    name = None
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as out:
        name = out.name
        out.write('requires 0.99.0\n')
        out.write(f'load "{opts.input}"\n')
        out.write(f'savetif32 "{opts.output[:-4]}"\n')
    try:
        subprocess.run(['siril', '-s', name], check=True)
    finally:
        os.unlink(name)


if __name__ == '__main__':
    main()
        
