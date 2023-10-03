import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import numpy as np


def cos_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (np.vdot(a, b) / d) if d > 0 else 0.0


def array_to_rectangles(arr: np.array, rect_w: int, rect_h: int):
    _, w, _ = arr.shape
    return np.concatenate(np.hsplit(arr, w // rect_w)).reshape(
        -1, rect_w, rect_h, 3
    )


def open_image(p, *, mode=None, size=None):
    img = Image.open(p)
    if mode is not None and img.mode != mode:
        img = img.convert(mode=mode)
    if size is not None and img.size != size:
        w, h = img.size
        if size[0] / size[1] != w / h:
            raise Exception(f"Aspect ratio mismatch {p}; {size} vs {img.size}")
        img = img.resize(size)
    return img


def image_loader_factory(mode):
    def load_images(image_paths, size):
        for p in image_paths:
            try:
                yield open_image(p, mode=mode, size=size)
            except Exception as e:
                print(e, file=sys.stderr)
                yield None

    return load_images


def get_by_extension(p: Path, extensions: List[str] | Tuple[str]):
    to_visit = [p]
    while len(to_visit):
        for item in to_visit.pop(0).iterdir():
            if item.is_dir():
                to_visit.append(item)
            elif item.is_file() and any(
                item.name.endswith(ext) for ext in extensions
            ):
                yield item


def is_prime(p: int):
    i = 2
    while i**2 <= p:
        if p % i == 0:
            return False
        i += 1
    return True


def get_factors(n):
    factors = []
    while not is_prime(n):
        for factor in range(2, n):
            x = n / factor
            if int(x) == x:
                factors.append(factor)
                n = int(x)
                break
    factors.append(n)
    return factors


def get_nearest_rect(area, count):
    factors = get_factors(area)
    tgt = (area / count) ** 0.5
    if int(tgt) == tgt:
        return (int(tgt), int(tgt))
    elif len(factors) < 2:
        raise Exception("Not enough factors")
    else:
        w = h = ax = 1
        idx = 0
        while idx < len(factors):
            n = factors[idx]
            nax = n * ax
            if nax > tgt:
                if w == 1:
                    w = nax
                    ax = 1
                else:
                    h = ax
                    break
            else:
                ax = nax
            idx += 1
        return (w, h)
