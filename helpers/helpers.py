import numpy
import subprocess
import zlib
import struct


sRGB_to_xyz = numpy.array([
    [0.4360747,  0.3850649, 0.1430804],
    [0.2225045,  0.7168786,  0.0606169],
    [0.0139322,  0.0971045,  0.7141733]
    ], dtype=numpy.float32)


rec2020_to_xyz = numpy.array([
    [0.6734241,  0.1656411,  0.1251286],
    [0.2790177,  0.6753402,  0.0456377],
    [-0.0019300,  0.0299784, 0.7973330]
    ], dtype=numpy.float32)


# 203 nits is Operational Target as Standardized by ITU-R BT.2408
def pq(a, inv=False, sdr_peak_nits=203.0):
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 107.0 / 128.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    scaling = 10000.0 / sdr_peak_nits
    if not inv:
        a /= scaling
        aa = numpy.power(a, m1)
        res = numpy.power((c1 + c2 * aa)/(1.0 + c3 * aa), m2)
    else:
        p = numpy.power(a, 1.0/m2)
        aa = numpy.fmax(p-c1, 0.0) / (c2 - c3 * p)
        res = numpy.power(aa, 1.0/m1)
        res *= scaling
    return res


def hlg(a, inv=False, sdr_peak_nits=203.0):
    h_a = 0.17883277
    h_b = 1.0 - 4.0 * 0.17883277
    h_c = 0.5 - h_a * numpy.log(4.0 * h_a)
    scaling = 1000.0 / sdr_peak_nits
    if not inv:
        rgb = a
        rgb /= scaling
        rgb = numpy.fmin(numpy.fmax(rgb, 1e-6), 1.0)
        rgb = numpy.where(rgb <= 1.0 / 12.0, numpy.sqrt(3.0 * rgb),
                          h_a * numpy.log(
                              numpy.fmax(12.0 * rgb - h_b, 1e-6)) + h_c)
        return rgb
    else:
        rgb = a
        rgb = numpy.where(rgb <= 0.5, rgb * rgb / 3.0,
                          (numpy.exp((rgb - h_c)/ h_a) + h_b) / 12.0)
        rgb *= scaling
        return rgb


def srgb(a, inv=False, clip=True):
    if not inv:
        a = numpy.fmax(a, 0.0)
        if clip:
            a = numpy.fmin(a, 1.0)
        return numpy.where(a <= 0.0031308,
                           12.92 * a,
                           1.055 * numpy.power(a, 1.0/2.4)-0.055)
    else:
        return numpy.where(a <= 0.04045, a / 12.92,
                           numpy.power((a + 0.055) / 1.055, 2.4))


def rec709(a, inv=False, clip=True):
    if not inv:
        a = numpy.fmax(a, 0.0)
        if clip:
            a = numpy.fmin(a, 1.0)
        return numpy.where(a < 0.018,
                           4.5 * a,
                           1.099 * numpy.power(a, 0.45) - 0.099)
    else:
        return numpy.where(a < 0.081,
                           a / 4.5,
                           numpy.power((a + 0.099) / 1.099, 1.0/0.45))


def copy_metadata(src, dst, include_icc=False):
    subprocess.run(['exiftool', '-tagsFromFile', src, '-all'] +
                   (['-icc_profile'] if include_icc else []) +
                   ['-overwrite_original', dst],
                   check=True)


def write_png16_rgb(path, a, level=0, rows_per_chunk=64):
    """Write a (h, w, 3) uint16 RGB array as a 16-bit PNG. Stdlib only.

    level=0 -> stored deflate blocks (fastest, ~6 bytes/pixel).
    rows_per_chunk controls the streaming block size / peak memory.
    """
    a = numpy.asarray(a)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected (h, w, 3) RGB array, got {a.shape}")
    if a.dtype.kind not in "ui":
        raise ValueError(f"expected an integer dtype, got {a.dtype}")

    a = numpy.ascontiguousarray(a, dtype=">u2") # PNG is big-endian
    h, w = a.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("image has a zero-length axis")
    rowbytes = w * 3 * 2

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + \
            struct.pack(">I", zlib.crc32(c))

    n = max(1, rows_per_chunk)
    block = numpy.empty((n, rowbytes + 1), numpy.uint8) # reused across blocks
    block[:, 0] = 0 # filter type 0 (None)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)))

        co = zlib.compressobj(level)
        for r0 in range(0, h, n):
            rows = a[r0:r0 + n]
            k = len(rows)
            block[:k, 1:] = rows.view(numpy.uint8).reshape(k, rowbytes)
            piece = co.compress(block[:k].tobytes())
            if piece:
                f.write(chunk(b"IDAT", piece))
        tail = co.flush()
        if tail:
            f.write(chunk(b"IDAT", tail))

        f.write(chunk(b"IEND", b""))


_lum = sRGB_to_xyz[1]
_to_yuv = numpy.array([_lum, _lum - [0, 0, 1], [1, 0, 0] - _lum],
                      dtype=numpy.float32)
_to_rgb = numpy.linalg.inv(_to_yuv)

def tonemap(x):
    c = 0
    a = 1.0 - c
    mid = 0.18
    b = (a / (mid - c)) * (1.0 - ((mid - c) / a)) * mid
    gamma = numpy.power((mid + b), 2.0) / (a * b)
    
    def rolloff(x):
        return a * x / (x + b) + c
    def contrast(x):
        return mid * numpy.power(x / mid, gamma)

    x = x.reshape(-1, 3).transpose()

    iy, u, v = numpy.split(_to_yuv @ x, 3, 0)

    h = numpy.max(iy)
    if h <= 1:
        return x.transpose().reshape(-1).copy()

    def tm(a):
        return rolloff(contrast(a))

    hue = numpy.arctan2(u, v)
    rgb = tm(x)
    y, u, v = numpy.split(_to_yuv @ rgb, 3, 0)
    sat = numpy.hypot(u, v) * numpy.where(iy > 0, numpy.sqrt(y / iy), 1.0)

    hue = 0.6 * hue + 0.4 * numpy.arctan2(u, v)

    u = sat * numpy.sin(hue)
    v = sat * numpy.cos(hue)
    oY = y

    yuv = numpy.stack([oY.transpose(), u.transpose(), v.transpose()], -1)
    rgb = _to_rgb @ yuv.reshape(-1, 3).transpose()
    rgb = rgb.transpose().reshape(-1)
    return rgb
