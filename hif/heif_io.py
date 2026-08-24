#!/usr/bin/env python3

import os, sys
import argparse
import math
import numpy
import tifffile
import struct
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
try:
    open_heif = pillow_heif.open_heif
except AttributeError:
    open_heif = pillow_heif.open    
pillow_heif.register_avif_opener()


def get_version():
    def toint(p):
        try:
            return int(p)
        except ValueError:
            return 0
    return tuple([toint(t) for t in pillow_heif.__version__.split('.')])


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
    def __init__(self, t):
        self.color_primaries = t[1]
        self.transfer_characteristics = t[2]
        self.matrix_coefficients = t[3]
        self.red_xy = t[5], t[6]
        self.green_xy = t[7], t[8]
        self.blue_xy = t[9], t[10]
        self.white_xy = t[11], t[12]

    def __str__(self):
        def xy(t): return tuple(map(lambda n: round(n, 3), t))
        return f'nclx: {self.color_primaries}/{self.transfer_characteristics}/{self.matrix_coefficients} - r: {xy(self.red_xy)}, g: {xy(self.green_xy)}, b: {xy(self.blue_xy)}, w: {xy(self.white_xy)}'

    def pack(self):
        if get_version() >= (0, 9, 1):
            return {
                'color_primaries' : self.color_primaries,
                'transfer_characteristics' : self.transfer_characteristics,
                'matrix_coefficients' : self.matrix_coefficients,
                'full_range_flag' : 1,
                'color_primary_red_x' : self.red_xy[0],
                'color_primary_red_y' : self.red_xy[1],
                'color_primary_green_x' : self.green_xy[0],
                'color_primary_green_y' : self.green_xy[1],
                'color_primary_blue_x' : self.blue_xy[0],
                'color_primary_blue_y' : self.blue_xy[1],
                'color_primary_white_x' : self.white_xy[0],
                'color_primary_white_y' : self.white_xy[1]
            }
        else:
            return struct.pack('BiiiBffffffff',
                               1, self.color_primaries,
                               self.transfer_characteristics,
                               self.matrix_coefficients, 1,
                               self.red_xy[0], self.red_xy[1],
                               self.green_xy[0], self.green_xy[1],
                               self.blue_xy[0], self.blue_xy[1],
                               self.white_xy[0], self.white_xy[1]
                               )
# end of class NclxProfile

sRGB_nclx = NclxProfile(
    (1, 1, 13, 5, 1,
     0.6399999856948853, 0.33000001311302185,
     0.30000001192092896, 0.6000000238418579,
     0.15000000596046448, 0.05999999865889549,
     0.3127000033855438, 0.32899999618530273))

rec2100_nclx = NclxProfile((1, 9, 16, 9, 1,
                            0.708, 0.292,
                            0.170, 0.797,
                            0.131, 0.046,
                            0.3127, 0.3290))

def get_nclx(info):
    try:
        data = info['nclx_profile']
        if get_version() >= (0, 9, 1):
            return NclxProfile((1, data['color_primaries'],
                                data['transfer_characteristics'],
                                data['matrix_coefficients'],
                                1,
                                data['color_primary_red_x'],
                                data['color_primary_red_y'],
                                data['color_primary_green_x'],
                                data['color_primary_green_y'],
                                data['color_primary_blue_x'],
                                data['color_primary_blue_y'],
                                data['color_primary_white_x'],
                                data['color_primary_white_y']))
        else:
            return NclxProfile(struct.unpack('BiiiBffffffff', data))
    except:
        return None

def getmatrix(nclx):
    if nclx:
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


def read(opts):
    heif = open_heif(opts.input, convert_hdr_to_8bit=False)
    width, height = heif.size
    print(f'found image: {width}x{height} pixels, {heif.bit_depth} bits')
    if opts.width and opts.height:
        heif = pillow_heif.thumbnail(heif, max(opts.width, opts.height))
    with Timer('decoding'):
        rgb = numpy.asarray(heif, dtype=numpy.float32) / (2**heif.bit_depth - 1)
        end = time.time()
    nclx = get_nclx(heif.info)
    profile = None
    del_profile = False
    if nclx:
        print('nclx profile: %s' % nclx)
    else:
        prof = get_profile(heif.info)
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
    height, width = data.shape[:2]
    if opts.transfer == 'hlg':
        data = helpers.hlg(data)
    elif opts.transfer == 'pq':
        data = helpers.pq(data)
    else:
        data = helpers.rec709(data)
    data *= 65535.0
    data = data.astype(numpy.uint16)
    with Timer('encoding'):
        heif_file = pillow_heif.from_bytes(mode="RGB;16",
                                           size=(width, height),
                                           data=data.tobytes())
    rec2100_nclx.transfer_characteristics = {
        'pq' : 16,
        'hlg' : 18,
        'rec709' : 1,
        }[opts.transfer]
    heif_file.info['nclx_profile'] = rec2100_nclx.pack()
    with Timer('saving'):
        ffi = pillow_heif.heif.ffi
        lib = pillow_heif.heif.lib
        @staticmethod
        def my_save(ctx, img_list, primary_index, **kwargs):
            enc_options = lib.heif_encoding_options_alloc()
            enc_options = ffi.gc(enc_options, lib.heif_encoding_options_free)
            enc_options.macOS_compatibility_workaround_no_nclx_profile = 0
            for i, img in enumerate(img_list):
                pillow_heif.heif.set_color_profile(img.heif_img, img.info)
                
                p_img_handle = ffi.new("struct heif_image_handle **")
                error = lib.heif_context_encode_image(ctx.ctx,
                                                      img.heif_img,
                                                      ctx.encoder,
                                                      enc_options, p_img_handle)
                pillow_heif.heif.check_libheif_error(error)
                new_img_handle = ffi.gc(p_img_handle[0],
                                        lib.heif_image_handle_release)
                exif = img.info["exif"]
                xmp = img.info["xmp"]
                if i == primary_index:
                    if i:
                        lib.heif_context_set_primary_image(ctx.ctx,
                                                           new_img_handle)
                    if kwargs.get("exif", -1) != -1:
                        exif = kwargs["exif"]
                        if isinstance(exif, Image.Exif):
                            exif = exif.tobytes()
                    if kwargs.get("xmp", -1) != -1:
                        xmp = kwargs["xmp"]
                pillow_heif.heif.set_exif(ctx, new_img_handle, exif)
                pillow_heif.heif.set_xmp(ctx, new_img_handle, xmp)
                pillow_heif.heif.set_metadata(ctx, new_img_handle, img.info)

        pillow_heif.HeifFile._save = my_save
        heif_file.save(opts.output, quality=80)


def main():
    opts = getopts()
    if opts.mode == 'write':
        write(opts)
    else:
        read(opts)


if __name__ == '__main__':
    main()
