"""Investigate why LOD2 face coverage is so low."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from pointcraft.data.lod2 import parse_obj, parse_mtl

tile_dir = r"D:\Desktop\实习\三维GIS\LOD2\53394611"
obj_files = [f for f in os.listdir(tile_dir) if f.endswith(".obj")]
obj_path = os.path.join(tile_dir, obj_files[0])

verts, uvs, faces, mtllib = parse_obj(obj_path)
mtl_path = os.path.join(tile_dir, mtllib)
mat_to_tex = parse_mtl(mtl_path)

print(f"Tile: {os.path.basename(tile_dir)}")
print(f"  vertices: {len(verts):,}")
print(f"  uvs: {len(uvs):,}")
print(f"  faces (triangulated): {len(faces):,}")
print(f"  materials in .mtl: {len(mat_to_tex)}")
print()

# Per face: material assignment
no_mat = sum(1 for _, _, m in faces if m is None)
no_tex = sum(1 for _, _, m in faces if m is not None and mat_to_tex.get(m) is None)
has_mat = sum(1 for _, _, m in faces if m is not None and mat_to_tex.get(m) is not None)
print(f"Face material status:")
print(f"  none (no usemtl):           {no_mat:,} ({100*no_mat/len(faces):.1f}%)")
print(f"  mat ref'd but no map_Kd:    {no_tex:,} ({100*no_tex/len(faces):.1f}%)")
print(f"  mat + texture OK:           {has_mat:,} ({100*has_mat/len(faces):.1f}%)")

# Show first few materials' texture filenames
print()
print("First 10 materials in MTL:")
for i, (m, t) in enumerate(list(mat_to_tex.items())[:10]):
    print(f"  {m} -> {t}")

# Check what 'no material' faces look like — Z range, area
import numpy as np
print()
print("Geometry summary by category:")
for label, predicate in [
    ("no-material",          lambda m: m is None),
    ("mat-but-no-texture",   lambda m: m is not None and mat_to_tex.get(m) is None),
    ("mat-and-texture",      lambda m: m is not None and mat_to_tex.get(m) is not None),
]:
    sel = [f for f in faces if predicate(f[2])]
    if not sel:
        continue
    zs = np.array([verts[list(f[0])][:, 2].mean() for f in sel])
    print(f"  {label}: {len(sel):,} faces, Z mean={zs.mean():.1f}, "
          f"Z range [{zs.min():.1f}, {zs.max():.1f}]")
