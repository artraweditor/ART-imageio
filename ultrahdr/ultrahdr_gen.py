#!/usr/bin/env python3

import os, sys
import tifffile
import numpy
import subprocess
import argparse
import tempfile
import math
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../helpers'))
import helpers


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('--sdr')
    p.add_argument('hdr')
    p.add_argument('output')
    return p.parse_args()


def save_hdr(data, outname):
    r, g, b = numpy.split(helpers.pq(data.reshape(-1)).reshape(-1, 3), 3, 1)
    d = 2**10-1
    packed = ((b * d).astype(numpy.uint32) << 20) \
        | ((g * d).astype(numpy.uint32) << 10) \
        | ((r * d).astype(numpy.uint32) << 0)
    with open(outname, 'wb') as out:
        out.write(packed.astype('<u4').tobytes())


def save_sdr(data, outname):
    r, g, b = numpy.split(helpers.srgb(data.reshape(-1)).reshape(-1, 3), 3, 1)
    d = 2**8-1
    packed = ((b * d).astype(numpy.uint32) << 16) \
        | ((g * d).astype(numpy.uint32) << 8) \
        | ((r * d).astype(numpy.uint32) << 0)
    with open(outname, 'wb') as out:
        out.write(packed.tobytes())


def read(filename):
    data = tifffile.imread(filename)
    h, w, p = data.shape
    if w & 1:
        data = numpy.delete(data, -1, 1)
    if h & 1:
        data = numpy.delete(data, -1, 0)
    return data


def main():
    opts = getopts()
    hdrdata = read(opts.hdr)
    height, width, planes = hdrdata.shape
    hdrdata = numpy.fmax(hdrdata.reshape(-1), 0)
    if not opts.sdr:
        sdrdata = helpers.tonemap(hdrdata).reshape(-1)
    else:
        sdrdata = read(opts.sdr)
        h, w, p = sdrdata.shape
        assert height == h and width == w and planes == p
        sdrdata = numpy.fmax(sdrdata.reshape(-1), 0)
    with tempfile.TemporaryDirectory() as d:
        save_hdr(hdrdata, os.path.join(d, 'out.hdr'))
        save_sdr(sdrdata, os.path.join(d, 'out.sdr'))
        subprocess.run(['ultrahdr_app', '-m', '0',
                        '-p', os.path.join(d, 'out.hdr'),
                        '-y', os.path.join(d, 'out.sdr'),
                        '-w', str(width), '-h', str(height),
                        '-C', '0', '-t', '2', '-R', '1', '-M', '0',
                        '-z', opts.output], check=True)
    helpers.copy_metadata(opts.hdr, opts.output)

if __name__ == '__main__':
    main()
