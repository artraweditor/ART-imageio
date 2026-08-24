#!/usr/bin/env python3

from PIL import Image
import webp
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../helpers'))
import helpers


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', choices=['read', 'write'], required=True)
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', type=int, default=0, nargs='?')
    p.add_argument('height', type=int, default=0, nargs='?')
    return p.parse_args()


def read(opts):
    src = webp.load_image(opts.input, 'RGB')
    if opts.width and opts.height:
        src.thumbnail((opts.width, opts.height))
    src.save(opts.output)
    helpers.copy_metadata(opts.input, opts.output, True)


def write(opts):
    src = Image.open(opts.input)
    out = webp.WebPPicture.from_pil(src)
    out.save(opts.output)
    helpers.copy_metadata(opts.input, opts.output, True)


def main():
    opts = getopts()
    if opts.mode == 'read':
        read(opts)
    else:
        write(opts)


if __name__ == '__main__':
    main()
