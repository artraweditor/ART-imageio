#!/usr/bin/env python3

import os, sys
import argparse
import subprocess
import tempfile
import re
import numpy
import tifffile

sys.path.append(os.path.join(os.path.dirname(__file__), '../helpers'))
import helpers


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', choices=['read', 'write'], required=True)
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', type=int, nargs='?')
    p.add_argument('height', type=int, nargs='?')
    p.add_argument('--hdr', action='store_true')
    return p.parse_args()


def get_profile(opts):
    res = subprocess.run(['jxlinfo', opts.input], stdout=subprocess.PIPE,
                         check=True, encoding='utf-8')
    profiles = {
        ('D65', 'sRGB primaries',
         'sRGB transfer function') : ('rec709.icc', helpers.srgb),
        
        ('D65', 'Rec.2100 primaries',
         'PQ transfer function') : ('rec2100.icc', helpers.pq),
        
        ('D65', 'Rec.2100 primaries',
         'HLG transfer function') : ('rec2100.icc', helpers.hlg),
    }
    for line in res.stdout.splitlines():
        if line.startswith('Color space: '):
            bits = line[13:].split(', ')
            if bits[0] == 'RGB':
                key = tuple(bits[1:-1])
                return profiles.get(key)
    return None


def linearize(img, fun):
    shape = img.shape
    img = img.reshape(-1)
    img = fun(img, True)
    return img.reshape(shape)
    

def read(opts):
    fd, name = tempfile.mkstemp(suffix='.ppm')
    os.close(fd)    
    subprocess.run(['djxl', '--bits_per_sample=16',
                    opts.input, name], check=True)
    with open(name, 'rb') as f:
        data = f.read()
        info = re.split(b'\\s+', data, 4)
    os.unlink(name)
    
    img = numpy.frombuffer(info[-1],
                           dtype=numpy.dtype(numpy.uint16).newbyteorder('>'))
    img = img.reshape((int(info[2]), int(info[1]), 3))
    img = img.astype(numpy.float32) / 65535.0

    profile = get_profile(opts)
    if profile:
        img = linearize(img, profile[1])
    tifffile.imwrite(opts.output, img)
    if profile:
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), profile[0]))
        subprocess.run(['exiftool', '-icc_profile<=' + p,
                        '-overwrite_original', opts.output], check=True)
    

def write(opts):
    fd, name = tempfile.mkstemp(suffix='.ppm')
    os.close(fd)

    data = tifffile.imread(opts.input)
    if not opts.hdr:
        data = numpy.fmax(numpy.fmin(data, 1.0), 0.0)
    data *= 65535.0
    data = data.astype(numpy.dtype(numpy.uint16).newbyteorder('>'))
    with open(name, 'wb') as out:
        out.write(b'P6 ')
        out.write(str(data.shape[1]).encode('utf-8'))
        out.write(b' ')
        out.write(str(data.shape[0]).encode('utf-8'))
        out.write(b' ')
        out.write(b'65535\n')
        out.write(data.tobytes('C'))

    colorspace = []
    if opts.hdr:
        colorspace = ['-x', 'color_space=RGB_D65_202_Per_PeQ', '-d', '0.0']
    subprocess.run(['cjxl', '--container=1', name, opts.output] + colorspace,
                   check=True)
    os.unlink(name)
    helpers.copy_metadata(opts.input, opts.output)

        
def main():
    opts = getopts()
    if opts.mode == 'read':
        read(opts)
    else:
        write(opts)


if __name__ == '__main__':
    main()
