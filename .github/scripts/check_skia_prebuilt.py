#!/usr/bin/env python3
"""Check that rust-skia publishes a prebuilt binary for anyrender_skia's skia-bindings features.

Mirrors skia-bindings' binary-cache key (`<hash>-<target>-<features>`) using `cargo tree` and
`cargo metadata`, then sends a HEAD request to the download URL. Nothing is compiled.

Usage: check_skia_prebuilt.py <target>...
"""

import json
import pathlib
import subprocess
import sys
import urllib.request

PACKAGE = "anyrender_skia"
URL_TEMPLATE = "https://github.com/rust-skia/skia-binaries/releases/download/{tag}/skia-binaries-{key}.tar.gz"

# skia-bindings cargo features that are part of the key (`Features::from_cargo_env`),
# with the key replacements from `feature::KEY_REPLACEMENTS`.
KEY_FEATURES = {
    "pdf": "pdf",
    "gl": "gl",
    "egl": "egl",
    "wayland": "wayland",
    "x11": "x11",
    "vulkan": "vulkan",
    "metal": "metal",
    "d3d": "d3d",
    "ganesh": "ganesh",
    "graphite": "graphite",
    "textlayout": "textlayout",
    "svg": "svg",
    "skottie": "skottie",
    "webp-encode": "webpe",
    "webp-decode": "webpd",
    "embed-freetype": "ftembed",
    "freetype-woff2": "ftwoff2",
    "jpeg-encode": "jpege",
    "jpeg-decode": "jpegd",
}
# Dropped as redundant on targets that don't use FreeType (`platform::redundant_features`).
FREETYPE_FEATURES = {"embed-freetype", "freetype-woff2"}


def cargo(*args):
    return subprocess.run(["cargo", *args], check=True, capture_output=True, text=True).stdout


def skia_bindings(target):
    metadata = json.loads(cargo("metadata", "--format-version", "1", "--filter-platform", target))
    [package] = [p for p in metadata["packages"] if p["name"] == "skia-bindings"]
    vcs_info = pathlib.Path(package["manifest_path"]).parent / ".cargo_vcs_info.json"
    sha = json.loads(vcs_info.read_text())["git"]["sha1"]
    return package["version"], sha[:20]


def key_features(target):
    out = cargo(
        "tree", "-q", "-p", PACKAGE, "--target", target,
        "-i", "skia-bindings", "-e", "normal", "--depth", "0", "-f", "{f}",
    )
    features = set(out.strip().split(","))
    if "windows" in target or "apple" in target:
        features -= FREETYPE_FEATURES
    return sorted(KEY_FEATURES[f] for f in features if f in KEY_FEATURES)


def check(target):
    tag, half_hash = skia_bindings(target)
    key = "-".join([half_hash, target, *key_features(target)])
    url = URL_TEMPLATE.format(tag=tag, key=key)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            ok = response.status == 200
    except urllib.error.HTTPError:
        ok = False
    print(f"{'ok' if ok else 'MISSING'}: {url}")
    return ok


if __name__ == "__main__":
    results = [check(target) for target in sys.argv[1:]]
    sys.exit(0 if results and all(results) else 1)
