"""M0 C1/C5 tests: CityGML parse + 6697->6677 reprojection, and typed-surface
shell voxelization (D5). No large data — a tiny inline GML + synthetic surfaces.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pointcraft.data import (
    GROUND_LABEL,
    parse_citygml,
    voxelize_citygml_target,
)
from pointcraft.data.target import FACADE_LABEL, ROOF_LABEL
from pointcraft.voxelization import VoxelGrid

# A known Tokyo point in EPSG:6697 (lat, lon, z) — within the PLATEAU tiles.
_LAT, _LON, _Z = 35.6850, 139.7800, 30.0

# Minimal CityGML: one Building with one RoofSurface holding one triangular ring.
# posList is "lat lon z" (srsDimension=3), matching the real PLATEAU files.
_TINY_GML = f"""<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel
  xmlns:core="http://www.opengis.net/citygml/2.0"
  xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
  xmlns:gml="http://www.opengis.net/gml">
  <gml:boundedBy>
    <gml:Envelope srsName="http://www.opengis.net/def/crs/EPSG/0/6697" srsDimension="3">
      <gml:lowerCorner>{_LAT} {_LON} {_Z}</gml:lowerCorner>
      <gml:upperCorner>{_LAT+0.001} {_LON+0.001} {_Z+1}</gml:upperCorner>
    </gml:Envelope>
  </gml:boundedBy>
  <core:cityObjectMember>
    <bldg:Building gml:id="b1">
      <bldg:boundedBy>
        <bldg:RoofSurface>
          <bldg:lod2MultiSurface>
            <gml:MultiSurface>
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList srsDimension="3">{_LAT} {_LON} {_Z} {_LAT} {_LON+0.0005} {_Z} {_LAT+0.0005} {_LON} {_Z} {_LAT} {_LON} {_Z}</gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
            </gml:MultiSurface>
          </bldg:lod2MultiSurface>
        </bldg:RoofSurface>
      </bldg:boundedBy>
    </bldg:Building>
  </core:cityObjectMember>
</core:CityModel>
"""


@pytest.fixture()
def tiny_gml(tmp_path):
    p = tmp_path / "tiny_bldg_6697.gml"
    p.write_text(_TINY_GML, encoding="utf-8")
    return str(p)


def test_parse_tags_roof_and_sets_crs(tiny_gml):
    ts = parse_citygml(tiny_gml)
    assert ts.crs == "EPSG:6677"
    assert len(ts) == 1
    assert ts.labels.tolist() == [ROOF_LABEL]  # 3, from bldg:RoofSurface
    assert ts.polygons[0].shape[1] == 3


def test_reprojection_axis_order_matches_pyproj(tiny_gml):
    """posList is (lat, lon, z); output must be (easting=x, northing=y, z).

    Verifying the always_xy / lon-lat feed: the parsed first vertex must equal a
    direct pyproj 6697->6677 transform of (lon, lat), with z passed through.
    """
    pyproj = pytest.importorskip("pyproj")
    ts = parse_citygml(tiny_gml)
    ring = ts.polygons[0]

    t = pyproj.Transformer.from_crs("EPSG:6697", "EPSG:6677", always_xy=True)
    ex, ny = t.transform(_LON, _LAT)  # feed (lon, lat) -> (easting, northing)

    assert ring[0, 0] == pytest.approx(ex, abs=1e-3)   # x = easting
    assert ring[0, 1] == pytest.approx(ny, abs=1e-3)   # y = northing
    assert ring[0, 2] == pytest.approx(_Z, abs=1e-9)   # z unchanged (horizontal xform)
    # Sanity: reprojected easting/northing land in the Tokyo plane-CS IX range.
    assert -8000 < ex < 4000 and -38000 < ny < -27000


def _quad(x0, x1, y0, y1, z0, z1, *, plane):
    """Axis-aligned quad ring (closed) in one of the xy/xz/yz planes."""
    if plane == "xy":      # horizontal (roof/ground) at z=z0
        return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z0]], float)
    if plane == "yz":      # vertical wall at x=x0
        return np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1], [x0, y0, z0]], float)
    raise ValueError(plane)


def test_voxelize_citygml_target_labels_and_occupancy():
    grid = VoxelGrid.from_bounds([0, 0, 0, 5, 5, 6], 1.0)
    surfaces = SimpleNamespace(
        polygons=[
            _quad(0, 4, 0, 4, 0, 0, plane="xy"),   # ground at z=0
            _quad(0, 4, 0, 4, 4, 4, plane="xy"),   # roof   at z=4
            _quad(0, 0, 0, 4, 0, 4, plane="yz"),   # wall   at x=0, z 0..4
        ],
        labels=np.array([GROUND_LABEL, ROOF_LABEL, FACADE_LABEL], dtype=np.int64),
    )
    coords, occ, sem = voxelize_citygml_target(surfaces, grid)

    # Contract dtypes/shapes + occupancy.
    assert coords.dtype == np.int32 and coords.shape[1] == 3
    assert occ.dtype == np.uint8 and np.all(occ == 1)
    assert sem.dtype == np.int64 and sem.shape[0] == coords.shape[0]
    # Only the three surface labels appear (semantics from type, not geometry).
    assert set(np.unique(sem).tolist()) <= {GROUND_LABEL, ROOF_LABEL, FACADE_LABEL}
    # Each surface type produced at least one voxel at the expected height band.
    k = coords[:, 2]
    assert np.any((sem == GROUND_LABEL) & (k == 0))   # ground at base
    assert np.any((sem == ROOF_LABEL) & (k == 4))     # roof on top
    assert np.any((sem == FACADE_LABEL) & (k > 0) & (k < 4))  # wall fills mid-height
    # Coords are unique and in-bounds.
    assert np.unique(coords, axis=0).shape[0] == coords.shape[0]
    assert grid.in_bounds(coords).all()


def test_voxelize_citygml_target_empty():
    grid = VoxelGrid.from_bounds([0, 0, 0, 4, 4, 4], 1.0)
    surfaces = SimpleNamespace(polygons=[], labels=np.zeros((0,), dtype=np.int64))
    coords, occ, sem = voxelize_citygml_target(surfaces, grid)
    assert coords.shape == (0, 3) and occ.shape == (0,) and sem.shape == (0,)
