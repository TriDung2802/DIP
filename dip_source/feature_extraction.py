import cv2
import numpy as np

ANGLE_MIN, ANGLE_MAX = 25, 65


def _adaptive_params(shape, min_line_length, max_line_gap):
    """Tham so Hough ti le theo kich thuoc ROI (None -> tu dong)."""
    h, w = shape[:2]
    diag = float(np.hypot(h, w))
    if min_line_length is None:
        min_line_length = max(20, int(0.12 * diag))
    if max_line_gap is None:
        max_line_gap = max(10, int(0.03 * diag))
    threshold = max(30, int(0.5 * min_line_length))
    return min_line_length, max_line_gap, threshold, diag


def _merge_collinear(segs, angle_tol, offset_tol, gap_tol):
    """Gop cac doan thang cung huong, cung vi tri, gan nhau thanh 1 duong.

    segs: mang (N, 4) float [x1, y1, x2, y2]. Tra ve list (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = segs.T
    length = np.hypot(x2 - x1, y2 - y1)
    theta = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    used = np.zeros(len(segs), dtype=bool)
    merged = []

    for i in np.argsort(-length):          # seed = doan dai nhat con lai
        if used[i]:
            continue
        dth = np.abs(theta - theta[i])
        dth = np.minimum(dth, 180.0 - dth)
        t = np.radians(theta[i])
        ux, uy = np.cos(t), np.sin(t)
        off = np.abs(-(mx - x1[i]) * uy + (my - y1[i]) * ux)

        idx = np.flatnonzero(~used & (dth <= angle_tol) & (off <= offset_tol))
        used[idx] = True

        # chieu cac dau mut len huong cua seed, sau do quet de noi cac doan
        p1 = (x1[idx] - x1[i]) * ux + (y1[idx] - y1[i]) * uy
        p2 = (x2[idx] - x1[i]) * ux + (y2[idx] - y1[i]) * uy
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        order = np.argsort(lo)
        lo, hi = lo[order], hi[order]

        start, end = lo[0], hi[0]
        chains = []
        for a, b in zip(lo[1:], hi[1:]):
            if a <= end + gap_tol:
                end = max(end, b)
            else:
                chains.append((start, end))
                start, end = a, b
        chains.append((start, end))

        for s, e in chains:
            merged.append((x1[i] + s * ux, y1[i] + s * uy,
                           x1[i] + e * ux, y1[i] + e * uy))
    return merged


def seatbelt_line_extraction(binary_img, min_line_length=None, max_line_gap=None):
    if binary_img is None:
        return []

    min_line_length, max_line_gap, threshold, diag = _adaptive_params(
        binary_img.shape, min_line_length, max_line_gap)

    lines = cv2.HoughLinesP(
        binary_img,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return []

    # Vector hoa: tinh goc/do dai cho tat ca doan cung luc, khong lap Python
    segs = lines.reshape(-1, 4).astype(np.float64)
    dx = np.abs(segs[:, 2] - segs[:, 0])
    dy = np.abs(segs[:, 3] - segs[:, 1])
    angle = np.degrees(np.arctan2(dy, dx))
    segs = segs[(angle >= ANGLE_MIN) & (angle <= ANGLE_MAX)]
    if len(segs) == 0:
        return []

    # Gop cac manh vun cua cung 1 mep day thanh 1 duong duy nhat
    merged = _merge_collinear(
        segs,
        angle_tol=8.0,
        offset_tol=max(4.0, 0.015 * diag),
        gap_tol=float(max_line_gap),
    )

    valid_lines = []
    for x1, y1, x2, y2 in merged:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_line_length:
            continue
        ang = float(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
        if not (ANGLE_MIN <= ang <= ANGLE_MAX):
            continue
        valid_lines.append({
            "coords": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            "angle": ang,
            "length": length
        })

    valid_lines.sort(key=lambda l: l["length"], reverse=True)
    return valid_lines
