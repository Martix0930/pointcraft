"""exp_004 §2a — Build combined candidate table.

Merges outputs/g1/tile_scan.csv (density/complexity) with
outputs/g1/ground_class2_sweep.csv (ground coverage) for the 39
coverage_ok tiles.  Adds sheet(row,col) and disjoint_from_2814
(Chebyshev ≥ 2 in the parsed 3-digit-row × 1-digit-col grid).

Writes outputs/g1/combined_candidates.csv and prints the pick sheet.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import pandas as pd


def _parse_sheet(tile_id: str) -> tuple[int, int]:
    """Return (row_indicator, col) from '09LD1823' → (182, 3)."""
    digits = tile_id.replace("09LD", "")   # '1823'
    return int(digits[:3]), int(digits[3])  # (182, 3)


REF_ROW, REF_COL = _parse_sheet("09LD2814")  # (281, 4)


def _cheb(tile_id: str) -> int:
    r, c = _parse_sheet(tile_id)
    return max(abs(r - REF_ROW), abs(c - REF_COL))


def main() -> None:
    scan_path   = os.path.join(REPO, "outputs", "g1", "tile_scan.csv")
    ground_path = os.path.join(REPO, "outputs", "g1", "ground_class2_sweep.csv")
    out_path    = os.path.join(REPO, "outputs", "g1", "combined_candidates.csv")

    scan   = pd.read_csv(scan_path)
    ground = pd.read_csv(ground_path)[["tile", "ground_cov_footprint"]].rename(
        columns={"tile": "tile_id"}
    )

    # coverage_ok only
    scan = scan[scan["coverage_ok"] == True].copy()

    df = scan.merge(ground, on="tile_id", how="left")

    # Sheet coordinates and disjointness
    df[["sheet_row", "sheet_col"]] = df["tile_id"].apply(
        lambda t: pd.Series(_parse_sheet(t))
    )
    df["cheb_dist"] = df["tile_id"].apply(_cheb)
    df["disjoint_from_2814"] = df["cheb_dist"] >= 2

    # Trim to spec columns (+ cheb_dist for inspection)
    cols = [
        "tile_id", "footprint_ratio", "surf_per_ha",
        "non_flat_roof_ratio", "height_std_m",
        "ground_cov_footprint",
        "sheet_row", "sheet_col",
        "disjoint_from_2814", "cheb_dist",
    ]
    df = df[cols].sort_values("surf_per_ha", ascending=True).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}\n")

    # ── Pretty-print pick sheet ────────────────────────────────────────────────
    print("=" * 100)
    print("COMBINED CANDIDATE TABLE  (coverage_ok=True, sorted by surf_per_ha asc)")
    print("  * = held-out (excluded)   X = Cheb<2 excluded   ~ = ground cov < 50%")
    print("=" * 100)
    hdr = (f"{'tile_id':12s}  {'ftpr':6s}  {'s/ha':7s}  {'nfr':6s}  "
           f"{'h_std':6s}  {'gnd%':6s}  {'rc':6s}  {'dj':5s}  {'flag'}")
    print(hdr)
    print("-" * 100)

    HELD_OUT = "09LD2814"
    for _, row in df.iterrows():
        flag = ""
        if row.tile_id == HELD_OUT:
            flag += "*held-out "
        elif not row.disjoint_from_2814:
            flag += "X cheb<2 "
        if row.ground_cov_footprint < 50.0:
            flag += "~gnd<50% "
        rc = f"({int(row.sheet_row)},{int(row.sheet_col)})"
        print(
            f"{row.tile_id:12s}  {row.footprint_ratio:6.3f}  {row.surf_per_ha:7.1f}  "
            f"{row.non_flat_roof_ratio:6.3f}  {row.height_std_m:6.2f}  "
            f"{row.ground_cov_footprint:6.1f}  {rc:8s}  {str(row.disjoint_from_2814):5s}  {flag}"
        )

    print("=" * 100)
    print(f"\nSummary: {len(df)} coverage_ok tiles total")
    disjoint = df[df.disjoint_from_2814 & (df.tile_id != HELD_OUT)]
    good_gnd = disjoint[disjoint.ground_cov_footprint >= 50.0]
    print(f"  Disjoint from 2814 (Cheb≥2, excl held-out): {len(disjoint)}")
    print(f"  Disjoint + ground_cov ≥ 50%: {len(good_gnd)}")
    print()
    print("RECOMMENDED CANDIDATES (disjoint, gnd≥50%, sorted by surf_per_ha):")
    print(good_gnd[["tile_id", "footprint_ratio", "surf_per_ha",
                     "non_flat_roof_ratio", "height_std_m",
                     "ground_cov_footprint", "sheet_row", "sheet_col"]].to_string(index=False))


if __name__ == "__main__":
    main()
