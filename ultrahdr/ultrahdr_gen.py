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


lum = helpers.sRGB_to_xyz[1]
to_yuv = numpy.array([lum, lum - [0, 0, 1], [1, 0, 0] - lum],
                     dtype=numpy.float32)
to_rgb = numpy.linalg.inv(to_yuv)


def tonemap(x):
    c = 0 
    a = 1.0 - c
    mid = 0.18
    b = (a / (mid - c)) * (1.0 - ((mid - c) / a)) * mid
    gamma = math.pow((mid + b), 2.0) / (a * b)
    
    def rolloff(x):
        return a * x / (x + b) + c
    def contrast(x):
        return mid * numpy.power(x / mid, gamma)

    x = x.reshape(-1, 3).transpose()

    y, u, v = numpy.split(to_yuv @ x, 3, 0)

    h = numpy.max(y)
    if h <= 1:
        return x.transpose().reshape(-1).copy()

    def tm(a):
        return rolloff(contrast(a))

    hue = numpy.arctan2(u, v)
    rgb = tm(x)
    y, u, v = numpy.split(to_yuv @ rgb, 3, 0)
    sat = numpy.hypot(u, v)

    hue = 0.6 * hue + 0.4 * numpy.arctan2(u, v)

    u = sat * numpy.sin(hue)
    v = sat * numpy.cos(hue)
    oY = y

    yuv = numpy.stack([oY.transpose(), u.transpose(), v.transpose()], -1)
    rgb = to_rgb @ yuv.reshape(-1, 3).transpose()
    rgb = rgb.transpose().reshape(-1)
    return rgb


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
        sdrdata = tonemap(hdrdata)
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
                        '-C', '0', '-t', '2', '-R', '1',
                        '-z', opts.output], check=True)
    helpers.copy_metadata(opts.hdr, opts.output)

if __name__ == '__main__':
    main()
