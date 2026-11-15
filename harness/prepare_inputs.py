#!/usr/bin/env python3
"""Build the read-only input tree of the benchmark from a local download of the
GeoAnalystBench data.

    python3 harness/prepare_inputs.py --source /path/to/GeoAnalystBench   # build
    python3 harness/prepare_inputs.py --verify                            # full SHA-256 check

<source>/<id>/dataset  ->  benchmark/inputs/<id>/dataset   (files 0444, directories 0555)

Only the data folder of each task is copied: the reference solutions that the download also
contains stay outside what a container can mount. Files in benchmark/supplements/<id>/dataset
(or --supplements), if present, are copied on top of the downloaded files; they complete the
tasks whose download lacks a dataset or part of it.

benchmark/manifest.json holds the size and SHA-256 of every input file. A task whose built
files differ from its manifest entry is reported and makes the exit code 1; a task without an
entry is added to the manifest. --verify compares the existing tree with the manifest and
checks that no input file is writable.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

MANIFEST = os.path.join(paths.BENCH, "manifest.json")
SUPPLEMENTS = os.path.join(paths.BENCH, "supplements")
# name of the data folder inside a task folder of the download: usually "dataset"
DATA_FOLDERS = ("dataset", "datasets", "data")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = {"size": os.path.getsize(p), "sha256": sha256(p)}
    return out


def set_writable(d, writable):
    for root, dirs, files in os.walk(d):
        for f in files:
            os.chmod(os.path.join(root, f), 0o644 if writable else 0o444)
        os.chmod(root, 0o755 if writable else 0o555)


def task_ids(d):
    return {t for t in os.listdir(d) if t.isdigit()} if d and os.path.isdir(d) else set()


def build(source, supplements):
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    differ = []
    for tid in sorted(task_ids(source) | task_ids(supplements), key=int):
        src = next((os.path.join(source, tid, n) for n in DATA_FOLDERS
                    if os.path.isdir(os.path.join(source, tid, n))), "")
        sup = os.path.join(supplements, tid, "dataset")
        if not os.path.isdir(src) and not os.path.isdir(sup):
            print(f"T{tid}: no data, skipped")
            continue
        # final tree = downloaded files, then supplements on top (supplements win on name clashes)
        want = {**(tree(src) if os.path.isdir(src) else {}), **(tree(sup) if os.path.isdir(sup) else {})}
        if tid in manifest and manifest[tid] != want:
            differ.append(tid)
        dst = os.path.join(paths.INPUTS, tid, "dataset")
        if manifest.get(tid) == want and os.path.isdir(dst):
            continue
        if os.path.exists(os.path.join(paths.INPUTS, tid)):
            set_writable(os.path.join(paths.INPUTS, tid), True)
            shutil.rmtree(os.path.join(paths.INPUTS, tid))
        os.makedirs(dst)
        for base in (src, sup):
            if os.path.isdir(base):
                shutil.copytree(base, dst, dirs_exist_ok=True)
        set_writable(os.path.join(paths.INPUTS, tid), False)
        manifest.setdefault(tid, want)
        print(f"T{tid}: {len(want)} files" + (" (+supplements)" if os.path.isdir(sup) else ""))
    json.dump(manifest, open(MANIFEST, "w"), indent=1, sort_keys=True)
    print(f"manifest: {len(manifest)} tasks -> {MANIFEST}")
    if differ:
        print(f"MISMATCH with the manifest: {differ}")
    return not differ


def verify():
    manifest = json.load(open(MANIFEST))
    bad = []
    for tid, files in manifest.items():
        d = os.path.join(paths.INPUTS, tid, "dataset")
        now = tree(d) if os.path.isdir(d) else {}
        if now != files:
            bad.append(tid)
        for root, dirs, fs in os.walk(d):
            for f in fs:
                if os.stat(os.path.join(root, f)).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                    bad.append(f"{tid}:writable:{f}")
    print("inputs verified OK" if not bad else f"MISMATCH: {bad}")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build or verify the read-only input tree.")
    ap.add_argument("--source", help="folder of the GeoAnalystBench download (one sub-folder per task id)")
    ap.add_argument("--supplements", default=SUPPLEMENTS,
                    help="folder with <id>/dataset files copied on top (default benchmark/supplements)")
    ap.add_argument("--verify", action="store_true", help="compare the existing tree with the manifest")
    a = ap.parse_args()
    if not a.verify and not a.source:
        ap.error("--source is required unless --verify is given")
    sys.exit(0 if (verify() if a.verify else build(a.source, a.supplements)) else 1)
