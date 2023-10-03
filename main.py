import os
import sys
from pathlib import Path
from multiprocessing import Pool, freeze_support
from dataclasses import dataclass
from typing import Callable
from operator import itemgetter
from shutil import rmtree
import gzip

from tqdm import tqdm
import numpy as np
import click
from image_matcher.utils import (
    array_to_rectangles,
    image_loader_factory,
    cos_sim,
    get_by_extension,
    get_nearest_rect,
)
from image_matcher.hierarchical_clustering import (
    create_hierarchy,
    flatten_to_top,
)


@dataclass
class Step:
    name: str
    action: Callable


def img_as_array(img):
    return (np.asarray(img, dtype=np.short) / 2).astype(np.byte)


def get_tfidf(a, corp, tile_w, tile_h, min_similarity=0.83):
    total_terms = len(a)
    minimum = (1 - min_similarity) * 127
    N = len(corp)
    tf = np.array(
        [
            np.count_nonzero(
                (
                    np.abs(a[:, np.newaxis] - b)
                    .reshape(-1, total_terms, tile_w * tile_h * 3)
                    .mean(axis=2)
                    <= minimum
                ),
                axis=1,
            )
            / total_terms
            for b in corp
        ]
    )
    n_docs = np.count_nonzero(np.rot90(tf, axes=(1, 0)), axis=1)
    return tf * np.log((N + 1) / (n_docs + 1))


def pool_get_tfidf(work):
    idx, image, corp_images, tile_w, tile_h, min_similarity = work
    return (idx, get_tfidf(image, corp_images, tile_w, tile_h, min_similarity))


def do_tfidf(kwargs):
    (
        test_images,
        image_size,
        tile_count,
        tile_size,
        min_similarity,
        paths_output,
        arrays_output,
        limit,
    ) = itemgetter(
        "test_images",
        "image_size",
        "tile_count",
        "tile_size",
        "min_similarity",
        "paths_output",
        "arrays_output",
        "limit",
    )(
        kwargs
    )
    image_extensions = (".jpg", ".png", ".bmp")
    image_loader = image_loader_factory("RGB")
    corp_image_paths = list(
        get_by_extension(Path(test_images), image_extensions)
    )
    if limit:
        corp_image_paths = corp_image_paths[:limit]
    area = image_size[0] * image_size[1]
    if tile_count:
        tile_size = get_nearest_rect(area, tile_count)
    elif not tile_size:
        tile_size = image_size
    print(
        f"Using {area / np.prod(tile_size)} tiles of size {tile_size}",
        file=sys.stderr,
    )
    # Only keep paths of images that are loaded successfully
    corp_image_paths, tiled_corp_images = zip(
        *[
            (
                corp_image_paths[i],
                array_to_rectangles(img_as_array(img), *tile_size),
            )
            for i, img in enumerate(image_loader(corp_image_paths, image_size))
            if img is not None
        ]
    )

    freeze_support()
    with Pool() as p, open(paths_output, "w", encoding="utf-8") as pout, open(
        arrays_output, "wb"
    ) as aout:
        for idx, tfidf_scores in tqdm(
            p.imap_unordered(
                pool_get_tfidf,
                (
                    (i, image, tiled_corp_images, *tile_size, min_similarity)
                    for i, image in enumerate(tiled_corp_images)
                ),
                chunksize=max(
                    int(len(corp_image_paths) / (os.cpu_count() or 1) / 4), 1
                ),
            ),
            desc="Getting tfidf for corpus",
            total=len(tiled_corp_images),
        ):
            try:
                print(corp_image_paths[idx], file=pout)
                np.save(aout, tfidf_scores)
            except Exception as e:
                print(e, file=sys.stderr)


def pool_cos_sim(work):
    p_idx, c_idx, p_arrays, c_arrays = work
    return p_idx, c_idx, cos_sim(p_arrays, c_arrays)


def calc_cos_sim(kwargs):
    arrays_output, similarity_output = itemgetter(
        "arrays_output", "similarity_output"
    )(kwargs)
    arrays = []
    with open(arrays_output, "rb") as ain:
        try:
            while True:
                arrays.append(np.load(ain))
        except EOFError:
            pass
    with Pool() as p, gzip.open(similarity_output, "wb") as fout:
        work_args = list(
            (p_idx, c_idx, p_arrays, c_arrays)
            for c_idx, c_arrays in enumerate(arrays)
            for p_idx, p_arrays in enumerate(arrays)
        )
        for p_idx, c_idx, sim in tqdm(
            p.imap(
                pool_cos_sim,
                work_args,
                chunksize=max(
                    int(len(work_args) / (os.cpu_count() or 1) / 2), 1
                ),
            ),
            desc="Calculating cosine similarities",
            total=len(work_args),
        ):
            fout.write(f"{p_idx},{c_idx},{sim}\n".encode("utf-8"))


def symlink_results(kwargs):
    (
        cos_similarity,
        paths_output,
        similarity_output,
        results_output,
    ) = itemgetter(
        "cos_similarity", "paths_output", "similarity_output", "results_output"
    )(
        kwargs
    )
    with open(paths_output, "r", encoding="utf-8") as pin:
        paths = [line.strip() for line in pin]
    with gzip.open(similarity_output, "rb") as fin:
        scores = [
            (int(v[0]), int(v[1]), float(v[2]))
            for v in (
                line.decode("utf-8").strip().split(",")
                for line in tqdm(fin, desc="Loading scores")
            )
        ]
    hierarchy = create_hierarchy(scores, cos_similarity)
    top_level = flatten_to_top(hierarchy)

    results = Path(results_output)
    rmtree(results, ignore_errors=True)
    results.mkdir(exist_ok=True, parents=True)
    for node in tqdm(top_level, desc="Symlinking results"):
        p_idx = node.name
        p_path = paths[p_idx]
        dst_dir = results / Path(p_path).with_suffix("").name
        dst_dir.mkdir(exist_ok=True, parents=True)
        for child_node in [node] + node.children:
            c_idx = child_node.name
            c_path = paths[c_idx]
            src = Path(c_path)
            dst = dst_dir / src.name
            while dst.exists():
                name = dst.with_suffix("").name
                dst = dst_dir / Path(f"{name} - Copy").with_suffix(dst.suffix)
            dst.symlink_to(src)


def new_step(f: Callable):
    return Step(f.__name__.lstrip("_").replace("_", "-"), f)


STEPS = [new_step(f) for f in [do_tfidf, calc_cos_sim, symlink_results]]

STEP_NAMES = [step.name for step in STEPS]


@click.command(
    help=(
        "Group images from TEST_IMAGES by tfidf cosine similarity at "
        "IMAGE_SIZE"
    )
)
@click.argument(
    "test-images",
    type=click.Path(exists=True),
)
@click.argument(
    "image-size",
    type=int,
    nargs=2,
)
@click.option("--limit", type=int, help="Limit the number of images compared")
@click.option(
    "--tile-count",
    type=int,
    help=(
        "Target tile count. This is used to approximate the dimensions of "
        "the rectangle to use. The approximation will most likely result in "
        "a greater tile count."
    ),
)
@click.option(
    "--tile-size",
    type=int,
    nargs=2,
    help="Force this tile size (note: image must divide evenly by this area)",
)
@click.option(
    "--min-similarity",
    type=float,
    default=0.83,
    help=(
        "Minimum similarity for rectangles to be considered identical in "
        "tfidf"
    ),
)
@click.option(
    "--start-at",
    type=click.Choice(STEP_NAMES),
    help="Skip steps before this one",
)
@click.option(
    "--stop-at", type=click.Choice(STEP_NAMES), help="Break at this step"
)
@click.option(
    "--cos-similarity",
    type=float,
    default=0.70,
    help="Minimum cosine similarity for clustering",
)
@click.option(
    "--paths-output",
    type=str,
    default="paths.txt",
    help="Save image paths to this file",
)
@click.option(
    "--arrays-output",
    type=str,
    default="arrays.npy",
    help="Save tfidf results to this file",
)
@click.option(
    "--similarity-output",
    type=str,
    default="similarities.csv.gz",
    help="Save similarity scores to this file",
)
@click.option(
    "--results-output",
    type=str,
    default="results_cluster",
    help="Put cluster results here",
)
def main(start_at, stop_at, **kwargs):
    skip_to = start_at
    for step in STEPS:
        if step.name == stop_at:
            break
        if skip_to:
            if step.name != skip_to:
                print(f"==> Skipping {step.name}", file=sys.stderr)
                continue
            else:
                skip_to = None
        print(f"==> Starting {step.name}", file=sys.stderr)
        step.action(kwargs)
        print(f"==> Finished {step.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
