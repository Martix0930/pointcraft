"""CityGML LOD2 -> typed surfaces in EPSG:6677 (M0 Phase C1, decision D5).

Parses PLATEAU LOD2 building surfaces tagged by **semantic type**
(``bldg:RoofSurface`` / ``bldg:WallSurface`` / ``bldg:GroundSurface``), reprojects
the geometry from the delivered **EPSG:6697 (lat/lon)** to **EPSG:6677** (the LiDAR
CRS), and returns typed polygon rings ready for shell voxelization (C5).
**No voxelization happens here** — this module only produces typed 6677 surfaces.

Why this replaces the OBJ + normal-heuristic path (D5): semantics come straight
from the CityGML surface type, so a building base is ``GroundSurface`` (not a
mislabelled roof) and roofs/walls are correctly typed at the source.

Axis-order gotcha (verified against tile 09LD1848 / grid 53394622):
  - ``gml:posList`` in EPSG:6697 is ordered ``lat lon z`` (``srsDimension=3``).
  - reproject with pyproj ``always_xy=True``, feeding ``(lon, lat)`` so the output
    is ``(easting, northing)``; calibration of the grid envelope reproduced the
    known 6677 extent X[-5324,-4146] Y[-35173,-34191].
  - output ``easting = x``, ``northing = y`` matches docs/02_DATA_CONTRACT.md and
    the LAS tile coordinates.
  - ``z`` (absolute elevation) passes through unchanged (D3; the transform is
    horizontal-only, both CRSs are JGD2011-based).

Pure stdlib XML parsing (``xml.etree``) + pyproj for reprojection; no learning.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .target import FACADE_LABEL, ROOF_LABEL

log = logging.getLogger(__name__)

#: Semantic label for a building footprint base (docs/02_DATA_CONTRACT.md, D5).
#: Note: this is the *building* ground face, sparse in PLATEAU LOD2 (many
#: buildings are open-bottomed), NOT terrain ground (which needs a DEM).
GROUND_LABEL = 1

#: CityGML LOD2 surface element local-name -> semantic label (D5).
SURFACE_TYPE_TO_LABEL: dict[str, int] = {
    "RoofSurface": ROOF_LABEL,      # 3
    "WallSurface": FACADE_LABEL,    # 4
    "GroundSurface": GROUND_LABEL,  # 1
}

_GML_NS = "http://www.opengis.net/gml"
_DEFAULT_SRC_EPSG = 6697
_DEFAULT_DST_EPSG = 6677


@dataclass(frozen=True)
class TypedSurfaces:
    """LOD2 surfaces tagged by semantic type, reprojected to ``crs`` (EPSG:6677).

    ``polygons[i]`` is the exterior ring of one surface as an ``(n_i, 3)`` float64
    array of world ``(x=easting, y=northing, z=elevation)`` coordinates. Interior
    rings (holes) are dropped — irrelevant at the 1 m shell resolution.
    ``labels[i]`` is the semantic class of ``polygons[i]`` (3 roof / 4 facade /
    1 ground), aligned by index.
    """

    polygons: list[np.ndarray]
    labels: np.ndarray  # (P,) int64
    crs: str

    def __len__(self) -> int:
        return len(self.polygons)

    def counts(self) -> dict[int, int]:
        """Number of polygons per semantic label."""
        if self.labels.size == 0:
            return {}
        vals, cnts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}


def _local(tag: str) -> str:
    """Strip the XML namespace from an element tag -> local name."""
    return tag.rsplit("}", 1)[-1]


def _parse_poslist(text: str, srs_dim: int) -> np.ndarray | None:
    """Parse a ``gml:posList`` text blob into an ``(n, srs_dim)`` float array.

    Returns ``None`` if the ring is empty or not a clean multiple of ``srs_dim``.
    """
    vals = np.fromstring(text, sep=" ", dtype=np.float64)
    if vals.size == 0 or vals.size % srs_dim != 0:
        return None
    return vals.reshape(-1, srs_dim)


def parse_citygml(
    path,
    *,
    src_epsg: int = _DEFAULT_SRC_EPSG,
    dst_epsg: int = _DEFAULT_DST_EPSG,
) -> TypedSurfaces:
    """Parse one CityGML LOD2 building file into reprojected typed surfaces.

    Args:
        path:     path to a ``*_bldg_*.gml`` file (EPSG:6697 lat/lon).
        src_epsg: source CRS of the GML coordinates (default 6697). Verified
                  against the file's declared ``srsName`` (warns on mismatch).
        dst_epsg: target CRS to reproject into (default 6677, the LiDAR CRS).

    Returns:
        ``TypedSurfaces`` with exterior rings in ``EPSG:{dst_epsg}`` and per-ring
        semantic labels. ``z`` is passed through unchanged (horizontal transform).
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True
    )

    polygons: list[np.ndarray] = []
    labels: list[int] = []

    declared_epsg: int | None = None
    n_surfaces = 0
    n_skipped_rings = 0

    poslist_tag = f"{{{_GML_NS}}}posList"
    exterior_tag = f"{{{_GML_NS}}}exterior"

    context = ET.iterparse(path, events=("end",))
    root = None
    for _event, el in context:
        if root is None:
            # The first 'end' belongs to a deep leaf; capture the document root
            # lazily below via the Envelope instead. (root kept for completeness.)
            pass
        local = _local(el.tag)

        if local == "Envelope" and declared_epsg is None:
            srs = el.get("srsName", "")
            # e.g. ".../EPSG/0/6697" or "EPSG:6697"
            digits = "".join(c for c in srs.rsplit("/", 1)[-1] if c.isdigit())
            if digits:
                declared_epsg = int(digits)
                if declared_epsg != src_epsg:
                    log.warning(
                        "CityGML srsName declares EPSG:%d but src_epsg=%d was "
                        "requested; trusting the requested src_epsg.",
                        declared_epsg,
                        src_epsg,
                    )
            continue

        if local in SURFACE_TYPE_TO_LABEL:
            label = SURFACE_TYPE_TO_LABEL[local]
            # Each semantic surface holds one or more Polygons; take exterior rings.
            for exterior in el.iter(exterior_tag):
                posl = exterior.find(f".//{poslist_tag}")
                if posl is None or not posl.text:
                    continue
                srs_dim = int(posl.get("srsDimension", 3))
                ring = _parse_poslist(posl.text, srs_dim)
                if ring is None or ring.shape[0] < 3:
                    n_skipped_rings += 1
                    continue
                lat = ring[:, 0]
                lon = ring[:, 1]
                z = ring[:, 2] if srs_dim >= 3 else np.zeros(ring.shape[0])
                # always_xy=True -> feed (lon, lat) -> (easting, northing)
                x, y = transformer.transform(lon, lat)
                polygons.append(
                    np.column_stack([x, y, z]).astype(np.float64)
                )
                labels.append(label)
            n_surfaces += 1
            el.clear()
        elif local == "cityObjectMember":
            # Free the whole building subtree (lodXSolid geometry etc.).
            el.clear()

    labels_arr = np.asarray(labels, dtype=np.int64)
    out = TypedSurfaces(polygons=polygons, labels=labels_arr, crs=f"EPSG:{dst_epsg}")
    log.info(
        "CityGML %s: %d semantic surfaces -> %d exterior rings %s "
        "(skipped %d degenerate); reprojected EPSG:%d->EPSG:%d",
        getattr(path, "name", str(path)),
        n_surfaces,
        len(polygons),
        out.counts(),
        n_skipped_rings,
        declared_epsg if declared_epsg is not None else src_epsg,
        dst_epsg,
    )
    return out


def load_citygml(paths, **kwargs) -> TypedSurfaces:
    """Parse and merge one or more CityGML files into a single ``TypedSurfaces``.

    Accepts a single path or an iterable of paths. All files are reprojected to the
    same ``dst_epsg`` (default 6677); their rings/labels are concatenated.
    """
    import os

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    all_polys: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    crs = f"EPSG:{kwargs.get('dst_epsg', _DEFAULT_DST_EPSG)}"
    for p in paths:
        ts = parse_citygml(p, **kwargs)
        all_polys.extend(ts.polygons)
        all_labels.append(ts.labels)
        crs = ts.crs
    labels_arr = (
        np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)
    )
    return TypedSurfaces(polygons=all_polys, labels=labels_arr, crs=crs)
