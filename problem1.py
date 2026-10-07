"""Problem 1 - homography estimation with the DLT (points and lines), bilinear warping, texture transfer.

Run from the folder that holds basketball-court.ppm and pacers3.png:   python problem1.py
Writes top_down_points.ppm, top_down_lines.ppm and texture-transfer.ppm.
"""
import numpy as np
from PIL import Image                                    # image reading / writing only


def dlt(p, q):
    """(a) DLT: 3x3 H with q ~ H p, from n >= 4 homogeneous correspondences (rows of p and q)."""
    # q x (H p) = 0 for every pair, i.e. ([q]x kron p^T) h = 0: three equations in h, two of them independent
    A = np.vstack([np.kron([[0, -w, v], [w, 0, -u], [-v, u, 0]], x) for x, (u, v, w) in zip(p, q)])
    return np.linalg.svd(A)[2][-1].reshape(3, 3)         # h = singular vector of the smallest singular value


def normalize(p):
    """Hartley normalisation T: centroid of the points -> origin, mean distance to it -> sqrt(2)."""
    c = p[:, :2].mean(0)
    s = np.sqrt(2) / np.linalg.norm(p[:, :2] - c, axis=1).mean()
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def lines(p):
    """Homogeneous lines through consecutive corners of a polygon: l_i = p_i x p_(i+1)."""
    return np.cross(p, np.roll(p, -1, 0))


def homography(p, q, use_lines=False):
    """H with q ~ H p from the corner points, or from the lines joining them, by the normalised DLT."""
    Tp, Tq = normalize(p), normalize(q)
    p, q = p @ Tp.T, q @ Tq.T
    # (c) points map as q ~ H p, lines as l_p ~ H^T l_q, so the same DLT run on the lines returns H^T
    H = dlt(lines(q), lines(p)).T if use_lines else dlt(p, q)
    return np.linalg.inv(Tq) @ H @ Tp


def bilinear(img, x, y):
    """(b) Bilinear interpolation of img at the real-valued pixel positions (x, y)."""
    x0 = np.clip(np.floor(x).astype(int), 0, img.shape[1] - 2)      # top-left neighbour
    y0 = np.clip(np.floor(y).astype(int), 0, img.shape[0] - 2)
    a, b = (x - x0)[..., None], (y - y0)[..., None]                 # offsets from it, in [0, 1]
    return ((1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x0 + 1]
            + (1 - a) * b * img[y0 + 1, x0] + a * b * img[y0 + 1, x0 + 1])


def transfer(src, H, dst, quad):
    """(d) Texture transfer: every pixel x of dst inside the quad (corners clockwise) takes src at H x."""
    y, x = np.indices(dst.shape[:2])
    inside = np.all([a * x + b * y + c >= 0 for a, b, c in lines(quad)], 0)
    x, y = x[inside], y[inside]
    u, v, w = [a * x + b * y + c for a, b, c in H]
    dst[inside] = np.rint(bilinear(src, u / w, v / w))
    return dst


load = lambda name: np.array(Image.open(name).convert("RGB"))
court, pacers = load("basketball-court.ppm"), load("pacers3.png")

# The four court corners (homogeneous pixel coordinates, clockwise, same order in every image) picked in
# basketball-court.ppm and pacers3.png, and where they go in the 940 x 500 top-down view (94 x 50 ft court)
c_court = np.array([(22, 193, 1), (246, 51, 1), (402, 74, 1), (279, 279, 1)], float)
c_pacers = np.array([(217, 1027, 1), (1422, 1019, 1), (1546, 1397, 1), (50, 1399, 1)], float)
c_top = np.array([(0, 0, 1), (939, 0, 1), (939, 499, 1), (0, 499, 1)], float)

# TASK 1 - top-down view of the court, H estimated from points and from lines
for name, use_lines in (("points", False), ("lines", True)):
    H = homography(c_top, c_court, use_lines)             # top-down pixel -> source pixel (inverse warping)
    print(f"H from {name}:\n{H / H[2, 2]}")
    top = transfer(court, H, np.zeros((500, 940, 3), np.uint8), c_top)     # into a blank 940 x 500 image
    Image.fromarray(top).save(f"top_down_{name}.ppm")

# TASK 2 - Pacers court texture painted onto the court of basketball-court.ppm
out = transfer(pacers, homography(c_court, c_pacers), court.copy(), c_court)
Image.fromarray(out).save("texture-transfer.ppm")
