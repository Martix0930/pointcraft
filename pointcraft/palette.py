"""Minecraft block palette with average sRGB colors.

Used by BlockMapper to convert per-cell RGB to nearest MC block ID via LAB distance.
Colors approximated from MC 1.21 texture atlases / community palettes.

Extend by appending to PALETTE or by constructing BlockPalette(your_entries).
"""
from __future__ import annotations
import numpy as np
from skimage import color as skcolor


# (block_id, (R, G, B))   — minecraft 1.21.1
PALETTE = [
    # ---- Stone / concrete neutrals
    ("minecraft:stone",                  (125, 125, 125)),
    ("minecraft:cobblestone",            (122, 122, 122)),
    ("minecraft:smooth_stone",           (158, 158, 158)),
    ("minecraft:andesite",               (135, 135, 135)),
    ("minecraft:diorite",                (188, 188, 187)),
    ("minecraft:granite",                (149, 105,  79)),
    ("minecraft:deepslate",              ( 78,  78,  82)),

    # ---- Concrete (saturated, modern building)
    ("minecraft:white_concrete",         (207, 213, 214)),
    ("minecraft:light_gray_concrete",    (125, 125, 115)),
    ("minecraft:gray_concrete",          ( 54,  57,  61)),
    ("minecraft:black_concrete",         (  8,  10,  15)),
    ("minecraft:brown_concrete",         ( 96,  59,  31)),
    ("minecraft:red_concrete",           (142,  33,  33)),
    ("minecraft:orange_concrete",        (224, 100,   1)),
    ("minecraft:yellow_concrete",        (240, 175,  21)),
    ("minecraft:lime_concrete",          ( 94, 168,  24)),
    ("minecraft:green_concrete",         ( 73,  91,  36)),
    ("minecraft:cyan_concrete",          ( 21, 119, 136)),
    ("minecraft:light_blue_concrete",    ( 36, 137, 199)),
    ("minecraft:blue_concrete",          ( 45,  47, 143)),
    ("minecraft:purple_concrete",        (100,  31, 156)),
    ("minecraft:magenta_concrete",       (169,  48, 159)),
    ("minecraft:pink_concrete",          (213, 101, 142)),

    # ---- Terracotta (earth tones, traditional architecture)
    ("minecraft:terracotta",             (152,  95,  67)),
    ("minecraft:white_terracotta",       (210, 178, 161)),
    ("minecraft:light_gray_terracotta",  (135, 107,  98)),
    ("minecraft:gray_terracotta",        ( 58,  42,  36)),
    ("minecraft:brown_terracotta",       ( 77,  51,  35)),
    ("minecraft:red_terracotta",         (143,  61,  47)),
    ("minecraft:orange_terracotta",      (162,  84,  38)),
    ("minecraft:yellow_terracotta",      (186, 133,  35)),
    ("minecraft:cyan_terracotta",        ( 86,  91,  91)),
    ("minecraft:blue_terracotta",        ( 74,  59,  91)),

    # ---- Wood / log
    ("minecraft:oak_planks",             (162, 130,  78)),
    ("minecraft:spruce_planks",          (114,  84,  48)),
    ("minecraft:dark_oak_planks",        ( 66,  43,  20)),

    # ---- Vegetation
    ("minecraft:oak_leaves",             ( 52,  84,  27)),
    ("minecraft:spruce_leaves",          ( 35,  53,  31)),
    ("minecraft:grass_block",            (115, 142,  70)),
    ("minecraft:moss_block",             ( 89, 109,  39)),

    # ---- Ground
    ("minecraft:dirt",                   (134,  96,  67)),
    ("minecraft:sand",                   (219, 207, 163)),
    ("minecraft:gravel",                 (134, 130, 127)),
    ("minecraft:smooth_sandstone",       (220, 207, 161)),
]


class BlockPalette:
    """LAB-nearest-neighbor block lookup over a palette."""

    def __init__(self, entries=PALETTE, exclude=()):
        excl = set(exclude)
        self.ids = [b for b, _ in entries if b not in excl]
        rgb = np.array([c for b, c in entries if b not in excl], dtype=np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        self.lab = lab.astype(np.float32)
        # Side-channel lookup so the viewer can show "what color does block X look like"
        self.rgb_lookup = {b: np.array(c, dtype=np.uint8) for b, c in entries}

    def nearest(self, rgb_arr: np.ndarray) -> list[str]:
        """rgb_arr: (N, 3) uint8 -> list of block IDs (length N)."""
        rgb01 = rgb_arr.astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
        d = ((lab[:, None, :] - self.lab[None, :, :]) ** 2).sum(axis=2)
        idx = d.argmin(axis=1)
        return [self.ids[i] for i in idx]
