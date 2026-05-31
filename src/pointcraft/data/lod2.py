"""LOD2 textured OBJ -> top-down color raster (z-buffered).

For each LOD2 face:
    - vertices project to XY plane (drop Z but keep for z-buffer test)
    - compute mean RGB of face from its UV region in its material texture
    - rasterize the 2D triangle into a numpy image at given resolution
    - keep pixel only if face's max Z > existing zbuf

Usage:
    raster = LOD2Rasterizer(tile_dirs=[...]).rasterize(origin_xy, size_xy_px, pixels_per_meter)
    -> (H, W, 3) uint8 image  +  (H, W) float32 z-buffer
"""
import os
import glob
import numpy as np
from PIL import Image
from skimage.draw import polygon as draw_polygon


def parse_mtl(path):
    """material_name -> texture_filename (relative)"""
    mats = {}
    cur = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("newmtl "):
                cur = line.split(None, 1)[1].strip()
                mats[cur] = None
            elif line.startswith("map_Kd ") and cur is not None:
                mats[cur] = line.split(None, 1)[1].strip()
    return mats


def parse_obj(path):
    """Returns:
        verts:     (Nv, 3) float
        uvs:       (Nu, 2) float
        faces:     list of (v_idx_tuple3, vt_idx_tuple3, material_name)
        mtllib:    path to mtl file (relative)
    """
    verts, uvs, faces = [], [], []
    mtllib = None
    cur_mat = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                toks = line.split()[1:]
                v_idx, vt_idx = [], []
                for t in toks:
                    parts = t.split("/")
                    v_idx.append(int(parts[0]) - 1)
                    vt_idx.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1)
                # triangulate fan if quad/n-gon
                for i in range(1, len(v_idx) - 1):
                    faces.append(((v_idx[0], v_idx[i], v_idx[i+1]),
                                  (vt_idx[0], vt_idx[i], vt_idx[i+1]),
                                  cur_mat))
            elif line.startswith("usemtl "):
                cur_mat = line.split(None, 1)[1].strip()
            elif line.startswith("mtllib "):
                mtllib = line.split(None, 1)[1].strip()
    return (np.array(verts, dtype=np.float64),
            np.array(uvs, dtype=np.float32),
            faces, mtllib)


def face_mean_color(uv_tri, tex_img, black_thresh=15):
    """Robust RGB color for a UV triangle in a texture image.

    PLATEAU texture atlases have huge black padding between real chunks.
    Naive mean over the UV polygon pulls colors toward black. We:
      1) sample pixels inside the UV polygon
      2) drop near-black pixels (sum of RGB < black_thresh*3)
      3) fall back to median if too many were dropped
      4) use centroid pixel for degenerate (tiny) triangles
    """
    H, W = tex_img.shape[:2]
    px = uv_tri[:, 0] * W
    py = (1.0 - uv_tri[:, 1]) * H
    rr, cc = draw_polygon(py, px, shape=(H, W))
    if len(rr) == 0:
        cy = int(np.clip(py.mean(), 0, H - 1))
        cx = int(np.clip(px.mean(), 0, W - 1))
        return tex_img[cy, cx, :3].astype(np.float32)
    samples = tex_img[rr, cc, :3].astype(np.float32)
    brightness = samples.sum(axis=1)
    keep = brightness >= black_thresh * 3
    if keep.sum() >= max(4, int(0.1 * len(samples))):
        return samples[keep].mean(axis=0)
    # Mostly black -> median over all samples (still better than mean)
    return np.median(samples, axis=0)


class LOD2Rasterizer:
    def __init__(self, tile_dirs):
        self.tile_dirs = tile_dirs
        self.tiles = []
        self._load()

    def _load(self):
        for d in self.tile_dirs:
            obj_files = glob.glob(os.path.join(d, "*.obj"))
            if not obj_files:
                continue
            obj_path = obj_files[0]
            verts, uvs, faces, mtllib = parse_obj(obj_path)
            mtl_path = os.path.join(d, mtllib) if mtllib else None
            mat_to_tex_name = parse_mtl(mtl_path) if mtl_path else {}
            tex_cache = {}
            tex_dir = os.path.join(d, "materials_textures")
            for mat, fname in mat_to_tex_name.items():
                if fname is None:
                    continue
                fpath = os.path.join(d, fname) if os.path.isabs(fname) else os.path.join(d, fname)
                # Some files reference materials_textures/xxx.jpg directly
                if not os.path.exists(fpath):
                    fpath = os.path.join(tex_dir, os.path.basename(fname))
                if os.path.exists(fpath):
                    tex_cache[mat] = np.asarray(Image.open(fpath).convert("RGB"))
            self.tiles.append({
                "verts": verts, "uvs": uvs, "faces": faces, "tex": tex_cache,
                "name": os.path.basename(d),
            })
            print(f"  loaded {os.path.basename(d)}: "
                  f"{len(verts):,} v, {len(faces):,} f, {len(tex_cache)} tex")

    def rasterize(self, origin_x, origin_y, width_m, height_m, ppm=1.0):
        """Top-down z-buffered raster.
        origin_x, origin_y: world coords of (0, 0) pixel (image lower-left)
        Returns: rgb (H, W, 3) uint8, zbuf (H, W) float32, valid (H, W) bool
        """
        W = int(round(width_m * ppm))
        H = int(round(height_m * ppm))
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        zbuf = np.full((H, W), -np.inf, dtype=np.float32)
        valid = np.zeros((H, W), dtype=bool)

        total_faces = sum(len(t["faces"]) for t in self.tiles)
        print(f"  rasterizing {total_faces:,} faces -> {W}x{H} image @ {ppm} px/m")

        n_drawn = 0
        n_skipped = 0
        for ti, tile in enumerate(self.tiles):
            verts = tile["verts"]
            uvs = tile["uvs"]
            faces = tile["faces"]
            tex = tile["tex"]
            # Pre-cache face colors per material to avoid recomputing for repeated UVs
            for vi, vti, mat in faces:
                if mat is None or mat not in tex:
                    n_skipped += 1
                    continue
                v3 = verts[list(vi)]  # (3, 3)
                # Project to image pixel coords
                px = (v3[:, 0] - origin_x) * ppm
                py = (v3[:, 1] - origin_y) * ppm
                # quick reject if outside
                if px.max() < 0 or px.min() >= W or py.max() < 0 or py.min() >= H:
                    continue
                zmax = float(v3[:, 2].max())
                # UV color
                if vti[0] < 0:
                    continue
                uv_tri = uvs[list(vti)]
                color = face_mean_color(uv_tri, tex[mat])
                # Rasterize triangle
                rr, cc = draw_polygon(py, px, shape=(H, W))
                if len(rr) == 0:
                    continue
                mask = zmax > zbuf[rr, cc]
                if not mask.any():
                    continue
                rr2 = rr[mask]; cc2 = cc[mask]
                rgb[rr2, cc2] = color.astype(np.uint8)
                zbuf[rr2, cc2] = zmax
                valid[rr2, cc2] = True
                n_drawn += 1
        print(f"  drew {n_drawn:,} faces (skipped {n_skipped:,} no-material)")
        return rgb, zbuf, valid

    def colored_point_samples(self, origin_x, origin_y, width_m, height_m,
                              spacing=0.5, max_samples=4_000_000, seed=0,
                              dark_thresh=30):
        """Sample the textured mesh surface (walls + roofs) into colored 3D points.

        Only faces overlapping the [origin, origin+size] XY window are sampled.
        Returns: pts (M, 3) float32 world XYZ, cols (M, 3) uint8 RGB.
        Used to reconstruct facade colors per voxel (3D nearest-color lookup).
        """
        rng = np.random.default_rng(seed)
        x0, x1 = origin_x, origin_x + width_m
        y0, y1 = origin_y, origin_y + height_m
        pts_chunks, col_chunks = [], []
        total = 0
        for tile in self.tiles:
            verts, uvs, faces, tex = tile["verts"], tile["uvs"], tile["faces"], tile["tex"]
            for vi, vti, mat in faces:
                if mat is None or mat not in tex or vti[0] < 0:
                    continue
                v3 = verts[list(vi)]
                if (v3[:, 0].max() < x0 or v3[:, 0].min() > x1 or
                        v3[:, 1].max() < y0 or v3[:, 1].min() > y1):
                    continue
                uv = uvs[list(vti)]
                e1 = v3[1] - v3[0]
                e2 = v3[2] - v3[0]
                area = 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
                n = int(np.clip(area / (spacing * spacing), 1, 6000))
                # uniform barycentric samples
                u1 = rng.random(n)
                u2 = rng.random(n)
                r1 = np.sqrt(u1)
                b0 = (1.0 - r1)[:, None]
                b1 = (r1 * (1.0 - u2))[:, None]
                b2 = (r1 * u2)[:, None]
                world = b0 * v3[0] + b1 * v3[1] + b2 * v3[2]          # (n, 3)
                uvp = b0 * uv[0] + b1 * uv[1] + b2 * uv[2]            # (n, 2)
                timg = tex[mat]
                Ht, Wt = timg.shape[:2]
                tx = np.clip((uvp[:, 0] * Wt).astype(np.int32), 0, Wt - 1)
                ty = np.clip(((1.0 - uvp[:, 1]) * Ht).astype(np.int32), 0, Ht - 1)
                cols = timg[ty, tx, :3]
                # Drop near-black texels (window glass / atlas padding) so walls
                # aren't dominated by black; keep the lit facade color.
                keep = cols.astype(np.int32).sum(axis=1) >= dark_thresh
                if not keep.any():
                    continue
                world = world[keep]
                cols = cols[keep]
                pts_chunks.append(world.astype(np.float32))
                col_chunks.append(cols.astype(np.uint8))
                total += int(keep.sum())
                if total >= max_samples:
                    break
            if total >= max_samples:
                break
        if not pts_chunks:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        pts = np.concatenate(pts_chunks, axis=0)
        cols = np.concatenate(col_chunks, axis=0)
        print(f"  facade samples: {len(pts):,} colored points")
        return pts, cols


