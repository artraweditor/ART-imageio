#!/usr/bin/env python3

import os, sys
import tifffile
import numpy
import subprocess
import argparse
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '../helpers'))
import helpers


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    return p.parse_args()


def save_png(data, outname, func):
    data = func(data)
    data *= 65535.0
    helpers.write_png16_rgb(outname, data.astype(numpy.uint16))


def save_hdr(data, outname):
    shape = data.shape
    sRGB_to_rec2020 = \
        numpy.linalg.inv(helpers.rec2020_to_xyz) @ helpers.sRGB_to_xyz
    data = sRGB_to_rec2020 @ data.reshape(-1, 3).transpose()
    save_png(data.transpose().reshape(shape), outname, helpers.pq)


def save_sdr(data, outname):
    data = helpers.tonemap(data)
    save_png(data, outname, helpers.srgb)


def read(filename):
    data = tifffile.imread(filename)
    shape = data.shape
    data = numpy.fmax(data.reshape(-1), 0)
    return data.reshape(shape)


def main():
    opts = getopts()
    data = read(opts.input)
    with tempfile.TemporaryDirectory() as d:
        sdrout = os.path.join(d, 'sdr.png')
        hdrout = os.path.join(d, 'hdr.png')
        save_sdr(data, sdrout)
        helpers.copy_metadata(opts.input, sdrout)
        save_hdr(data, hdrout)
        subprocess.run(['avifgainmaputil', 'combine', 
                        '--ignore-profile',
                        '-q', '80', '--qgain-map', '60',
                        '-d', '10', '--depth-gain-map', '8',
                        '--downscaling', '2',
                        '-y', '444', '--yuv-gain-map', '444',
                        '--cicp-base', '1/13/5',
                        '--cicp-alternate', '9/16/9',
                        '--max-headroom', '0',
                        sdrout, hdrout, opts.output], check=True)


if __name__ == '__main__':
    main()
