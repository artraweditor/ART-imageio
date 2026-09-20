#!/usr/bin/env python3

import os, sys
import argparse
import math
import numpy
import tifffile
import time
import subprocess
import tempfile
from contextlib import contextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), '../helpers'))
import helpers

try:
    import pillow_heif
except ImportError:
    import pi_heif as pillow_heif


def get_version():
    def toint(p):
        try:
            return int(p)
        except ValueError:
            return 0
    return tuple([toint(t) for t in pillow_heif.__version__.split('.')])


if get_version() < (1, 0, 0):
    sys.exit('error: pillow_heif >= 1.0.0 is required '
             '(found %s), please upgrade with: '
             'python3 -m pip install -U pillow_heif' % pillow_heif.__version__)


@contextmanager
def Timer(msg):
    try:
        start = time.time()
        yield
    finally:
        end = time.time()
        print('%s: %.3f s' % (msg, end - start))


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', nargs='?', default=0, type=int)
    p.add_argument('height', nargs='?', default=0, type=int)
    p.add_argument('-m', '--mode', choices=['read', 'write'], default='read')
    p.add_argument('-t', '--transfer', choices=['pq', 'hlg', 'rec709'],
                   default='pq')
    p.add_argument('-q', '--quality', default=80, type=int)
    return p.parse_args()

ACES_AP0_coords = ((0.735, 0.265),
                   (0.0, 1.0),
                   (0.0, -0.077),
                   (0.322, 0.338))

# ACES AP0 v4 ICC profile with linear TRC
ACES_AP0 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ap0.icc'))


def compute_xyz_matrix(key):
    def xyz(xy): return xy[0], xy[1], 1.0 - xy[0] - xy[1]
    r, g, b, w = map(xyz, key)
    w = [w[0]/w[1], 1.0, w[-1]/w[1]]
    m = numpy.array([[r[0], g[0], b[0]],
                     [r[1], g[1], b[1]],
                     [r[2], g[2], b[2]]])
    coeffs = numpy.linalg.solve(m, w)
    return m @ numpy.diag(coeffs)


class NclxProfile:
    def __init__(self, color_primaries, transfer_characteristics,
                 matrix_coefficients, full_range_flag=1, coords=None):
        self.color_primaries = color_primaries
        self.transfer_characteristics = transfer_characteristics
        self.matrix_coefficients = matrix_coefficients
        self.full_range_flag = full_range_flag
        if coords is not None:
            self.red_xy, self.green_xy, self.blue_xy, self.white_xy = coords
        else:
            self.red_xy = self.green_xy = self.blue_xy = self.white_xy = None

    def __str__(self):
        def xy(t): return tuple(map(lambda n: round(n, 3), t)) if t else None
        return f'nclx: {self.color_primaries}/{self.transfer_characteristics}/{self.matrix_coefficients} - r: {xy(self.red_xy)}, g: {xy(self.green_xy)}, b: {xy(self.blue_xy)}, w: {xy(self.white_xy)}'

    def pack(self):
        """nclx parameters in the form expected by pillow_heif when saving"""
        return {
            'color_primaries' : self.color_primaries,
            'transfer_characteristics' : self.transfer_characteristics,
            'matrix_coefficients' : self.matrix_coefficients,
            'full_range_flag' : self.full_range_flag,
            }

    @staticmethod
    def unpack(data):
        """build a NclxProfile out of the dict returned by pillow_heif"""
        coords = None
        try:
            coords = ((data['color_primary_red_x'],
                       data['color_primary_red_y']),
                      (data['color_primary_green_x'],
                       data['color_primary_green_y']),
                      (data['color_primary_blue_x'],
                       data['color_primary_blue_y']),
                      (data['color_primary_white_x'],
                       data['color_primary_white_y']))
        except KeyError:
            pass
        return NclxProfile(data['color_primaries'],
                           data['transfer_characteristics'],
                           data['matrix_coefficients'],
                           data.get('full_range_flag', 1),
                           coords)
# end of class NclxProfile

sRGB_nclx = NclxProfile(1, 13, 5)

rec2100_nclx = NclxProfile(9, 16, 9)


def get_nclx(info):
    try:
        return NclxProfile.unpack(info['nclx_profile'])
    except:
        return None

def getmatrix(nclx):
    if nclx and nclx.red_xy:
        return compute_xyz_matrix([nclx.red_xy, nclx.green_xy, nclx.blue_xy,
                                   nclx.white_xy])
    else:
        return None


def get_profile(info):
    return info.get('icc_profile')


def rec1886(a, inv):
    return numpy.power(numpy.fmax(a, 0.0), 1.0/2.4 if not inv else 2.4)


def hlg(a, inv):
    h_a = 0.17883277
    h_b = 1.0 - 4.0 * 0.17883277
    h_c = 0.5 - h_a * math.log(4.0 * h_a)
    if not inv:
        rgb = a
        #rgb /= 12.0
        rgb = numpy.fmin(numpy.fmax(rgb, 1e-6), 1.0)
        rgb = numpy.where(rgb <= 1.0 / 12.0, numpy.sqrt(3.0 * rgb),
                          h_a * numpy.log(
                              numpy.fmax(12.0 * rgb - h_b, 1e-6)) + h_c)
        return rgb
    else:
        rgb = a
        rgb = numpy.where(rgb <= 0.5, rgb * rgb / 3.0,
                          (numpy.exp((rgb - h_c)/ h_a) + h_b) / 12.0)
        #rgb *= 12.0
        return rgb


def linearize(data, nclx):
    if not nclx:
        return data
    shape = data.shape
    data = data.reshape(-1)
    if nclx.transfer_characteristics in (1, 6, 14, 15):
        # Rec.709
        data = helpers.rec709(data, True)
    elif nclx.transfer_characteristics == 13:
        # sRGB
        data = helpers.srgb(data, True)
    elif nclx.transfer_characteristics == 16:
        # PQ
        data = helpers.pq(data, True)
    elif nclx.transfer_characteristics == 18:
        # HLG
        data = helpers.hlg(data, True)
    else:
        pass
    return data.reshape(shape)


def decode(path):
    # hdr_to_16bit=False keeps the samples in their native range, so that we
    # can normalise them exactly with the bit depth of the image
    heif = pillow_heif.open_heif(path, convert_hdr_to_8bit=False,
                                 hdr_to_16bit=False)
    info = dict(heif.info)
    bit_depth = info.get('bit_depth', 8)
    width, height = heif.size
    print(f'found image: {width}x{height} pixels, {bit_depth} bits')
    with Timer('decoding'):
        rgb = numpy.asarray(heif, dtype=numpy.float32) / (2**bit_depth - 1)
    if rgb.ndim == 2:
        rgb = numpy.dstack([rgb] * 3)
    elif rgb.shape[2] > 3:
        # drop the alpha channel, ART wants plain RGB
        rgb = numpy.ascontiguousarray(rgb[:, :, :3])
    return rgb, info


def resize(rgb, width, height):
    """Downscale rgb so that it fits in a width x height box"""
    h, w = rgb.shape[:2]
    scale = min(width / w, height / h)
    if scale >= 1.0:
        return rgb
    from PIL import Image
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return numpy.dstack(
        [numpy.asarray(Image.fromarray(rgb[:, :, c]).resize(
            (nw, nh), Image.BOX), dtype=numpy.float32)
         for c in range(rgb.shape[2])])


def read(opts):
    rgb, info = decode(opts.input)
    if opts.width and opts.height:
        with Timer('resizing'):
            rgb = resize(rgb, opts.width, opts.height)
    nclx = get_nclx(info)
    profile = None
    del_profile = False
    if nclx:
        print('nclx profile: %s' % nclx)
    else:
        prof = get_profile(info)
        if not prof:
            print('no profile found, assuming sRGB')
            nclx = sRGB_nclx
        else:
            fd, profile = tempfile.mkstemp()
            with open(fd, 'wb') as out:
                out.write(prof)
            del_profile = True
    with Timer('linearization'):
        to_xyz = getmatrix(nclx)
        rgb = linearize(rgb, nclx)
        if to_xyz is not None:
            ap0_to_xyz = compute_xyz_matrix(ACES_AP0_coords)
            to_ap0 = numpy.linalg.inv(ap0_to_xyz) @ to_xyz
            shape = rgb.shape
            rgb = rgb.reshape(-1, 3).transpose()
            rgb = to_ap0 @ rgb
            rgb = rgb.transpose().reshape(*shape).astype(numpy.float32)
            profile = ACES_AP0
    with Timer('saving'):
        tifffile.imwrite(opts.output, rgb)
        if profile is not None:
            subprocess.run(['exiftool', '-icc_profile<=' + profile,
                            '-overwrite_original', opts.output], check=True)
    if del_profile:
        os.unlink(profile)


def write(opts):
    with Timer('loading'):
        data = tifffile.imread(opts.input)
    if data.ndim == 2:
        data = numpy.dstack([data] * 3)
    elif data.shape[2] > 3:
        data = data[:, :, :3]
    height, width = data.shape[:2]
    data = numpy.fmax(data.astype(numpy.float32), 0.0)
    if opts.transfer == 'hlg':
        data = helpers.hlg(data)
    elif opts.transfer == 'pq':
        data = helpers.pq(data)
    else:
        data = helpers.rec709(data)
    data = numpy.clip(data * 65535.0 + 0.5, 0.0, 65535.0).astype(numpy.uint16)
    nclx = NclxProfile(rec2100_nclx.color_primaries,
                       {
                           'pq' : 16,
                           'hlg' : 18,
                           'rec709' : 1,
                       }[opts.transfer],
                       rec2100_nclx.matrix_coefficients)
    with Timer('encoding'):
        pillow_heif.encode('RGB;16', (width, height),
                           numpy.ascontiguousarray(data).tobytes(),
                           opts.output,
                           quality=opts.quality,
                           save_nclx_profile=True,
                           **nclx.pack())


def main():
    opts = getopts()
    if opts.mode == 'write':
        write(opts)
    else:
        read(opts)


if __name__ == '__main__':
    main()
