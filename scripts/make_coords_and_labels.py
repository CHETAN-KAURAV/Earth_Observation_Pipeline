#!/usr/bin/env python3
"""
make_coords_and_labels.py

- Scans data/rgb for image files whose filenames contain latitude/longitude.
- Builds data/image_coords.csv (filename,lat,lon).
- Uses data/land_cover.tif to extract a 128x128 class patch centered at each image coordinate.
- Assigns label = mode(class) for the patch, maps ESA codes -> 11 labels (example mapping included).
- Saves outputs/labels.csv and train/test splits to outputs/.

No path edits required if you keep:
  data/rgb/            <- your images
  data/land_cover.tif  <- landcover raster (ESA WorldCover 2021)
The script creates data/image_coords.csv and outputs/ automatically.
"""
import re
import csv
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from sklearn.model_selection import train_test_split

# -------------------
# Configuration (no need to change)
# -------------------
IMAGES_DIR = Path("data/rgb")
OUT_CSV = Path("data/image_coords.csv")
LANDCOVER_TIF = Path("data/land_cover.tif")
OUTDIR = Path("outputs")
PATCH_SIZE = 128  # pixels
NODATA_FILL = 0

# Example ESA -> 11-label mapping (adjust if your assignment specifies a different mapping)
ESA_TO_11 = {
    10: 'Tree', 20: 'Shrub', 30: 'Grass', 40: 'Cropland', 50: 'Built-up',
    60: 'Bare', 70: 'Snow', 80: 'Water', 90: 'HerbaceousWetland',
    95: 'Mangroves', 100: 'SparseVegetation'
}

# -------------------
# Helpers
# -------------------
FLOAT_RE = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

def parse_coords_from_name(name):
    """
    Extract numeric tokens from filename stem.
    Returns list of numeric strings (may include ints or floats).
    """
    stem = Path(name).stem
    return FLOAT_RE.findall(stem)

def to_float_safe(s):
    try:
        return float(s)
    except:
        return None

def decide_latlon_order(a, b):
    """
    Heuristic:
      - If first number abs>90 and second <=90 => assume (lon,lat) => swap to (lat,lon)
      - If second number abs>90 and first <=90 => assume (lat,lon) (keep)
      - else assume (lat,lon)
    """
    if a is None or b is None:
        return None, None
    if abs(a) > 90 and abs(b) <= 90:
        return b, a
    return a, b

def build_image_coords(images_dir=IMAGES_DIR, out_csv=OUT_CSV, exts=('.png','.jpg','.jpeg','.tif','.tiff')):
    images_dir = Path(images_dir)
    out_csv = Path(out_csv)   # FIX: ensure Path object
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir.resolve()}")
    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    rows = []
    skipped = []
    for f in files:
        nums = parse_coords_from_name(f.name)
        if len(nums) < 2:
            skipped.append((f.name, 'not_enough_numbers'))
            continue
        a = to_float_safe(nums[0])
        b = to_float_safe(nums[1])
        lat, lon = decide_latlon_order(a, b)
        if lat is None or lon is None:
            skipped.append((f.name, 'invalid_numbers'))
            continue
        # sanity check ranges; if not OK try swapping
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            # try swapping
            lat2, lon2 = lon, lat
            if -90 <= lat2 <= 90 and -180 <= lon2 <= 180:
                lat, lon = lat2, lon2
            else:
                skipped.append((f.name, 'out_of_range'))
                continue
        rows.append({'filename': f.name, 'lat': lat, 'lon': lon})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename','lat','lon'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} entries to {out_csv.resolve()}")
    if skipped:
        print(f"Skipped {len(skipped)} files. Examples (up to 10):")
        for s in skipped[:10]:
            print("  ", s[0], "->", s[1])
    return out_csv

def extract_patch_from_raster(lc_path, lon, lat, out_size=PATCH_SIZE, nodata_fill=NODATA_FILL):
    """Return patch numpy array (out_size x out_size) of the landcover classes centered at lon,lat (EPSG:4326)."""
    lc_path = Path(lc_path)
    if not lc_path.exists():
        raise FileNotFoundError(f"Landcover raster not found: {lc_path.resolve()}")
    with rasterio.open(lc_path) as src:
        # transform lat/lon to raster crs if needed
        raster_epsg = None
        try:
            raster_epsg = src.crs.to_epsg()
        except Exception:
            raster_epsg = None
        if raster_epsg is None:
            # assume same as input
            x, y = lon, lat
        else:
            if raster_epsg != 4326:
                transformer = Transformer.from_crs('epsg:4326', f'epsg:{raster_epsg}', always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat
        # get row/col
        try:
            col, row = src.index(x, y)
        except Exception:
            # if outside extent, return nodata patch
            return np.full((out_size, out_size), nodata_fill, dtype=np.int32)
        half = out_size // 2
        col_start = col - half
        row_start = row - half
        window = Window(col_start, row_start, out_size, out_size)
        arr = src.read(1, window=window, boundless=True, fill_value=nodata_fill)
        return arr

def compute_mode_label(patch, nodata_value=NODATA_FILL):
    flat = patch.flatten()
    valid = flat[flat != nodata_value]
    if valid.size == 0:
        return None, 0.0
    # use bincount for integer classes (works if labels are non-negative small ints)
    try:
        counts = np.bincount(valid.astype(np.int64))
        top = counts.argmax()
        cnt = counts[top]
        frac = cnt / flat.size
        return int(top), float(frac)
    except Exception:
        # fallback
        c = Counter(valid.tolist())
        val, cnt = c.most_common(1)[0]
        return int(val), float(cnt / flat.size)

def build_labels(image_coords_csv, landcover_tif=LANDCOVER_TIF, outdir=OUTDIR, out_size=PATCH_SIZE, nodata_val=NODATA_FILL, esa_map=ESA_TO_11):
    image_coords_csv = Path(image_coords_csv)   # FIX: ensure Path
    landcover_tif = Path(landcover_tif)         # FIX: ensure Path
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not image_coords_csv.exists():
        raise FileNotFoundError(f"Image coords CSV not found: {image_coords_csv.resolve()}")
    df = pd.read_csv(image_coords_csv)
    rows = []
    for i, r in df.iterrows():
        fname = r['filename']
        lat = float(r['lat'])
        lon = float(r['lon'])
        patch = extract_patch_from_raster(landcover_tif, lon, lat, out_size=out_size, nodata_fill=nodata_val)
        label_val, dominance = compute_mode_label(patch, nodata_value=nodata_val)
        if label_val is None:
            label_name = 'uncertain'
        else:
            label_name = esa_map.get(label_val, f'esa_{label_val}')
        # if dominance very low, mark uncertain
        if dominance < 0.2:
            label_name = 'uncertain'
        rows.append({'filename': fname, 'lat': lat, 'lon': lon, 'label_val': label_val, 'dominance': dominance, 'label_name': label_name})
        if (i+1) % 200 == 0:
            print(f"Processed {i+1}/{len(df)} images")
    out_df = pd.DataFrame(rows)
    labels_csv = outdir / 'labels.csv'
    out_df.to_csv(labels_csv, index=False)
    print(f"Saved labels to {labels_csv.resolve()} ({len(out_df)} rows)")

    # Train/test split (60/40). Use stratify on label_name but if too many 'uncertain' may fail -> fallback
    try:
        train, test = train_test_split(out_df, test_size=0.4, random_state=42, stratify=out_df['label_name'])
    except Exception:
        train, test = train_test_split(out_df, test_size=0.4, random_state=42)
    train.to_csv(outdir / 'train_labels.csv', index=False)
    test.to_csv(outdir / 'test_labels.csv', index=False)
    print(f"Saved train/test splits: {len(train)} train, {len(test)} test")
    return labels_csv, outdir / 'train_labels.csv', outdir / 'test_labels.csv'

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Build image_coords.csv from data/rgb and generate labels from land_cover.tif")
    parser.add_argument('--images_dir', default=str(IMAGES_DIR), help="Directory containing RGB images (default: data/rgb)")
    parser.add_argument('--landcover', default=str(LANDCOVER_TIF), help="Path to land_cover.tif (default: data/land_cover.tif)")
    parser.add_argument('--out_csv', default=str(OUT_CSV), help="Output CSV for image coordinates (default: data/image_coords.csv)")
    parser.add_argument('--outdir', default=str(OUTDIR), help="Output directory for labels (default: outputs)")
    args = parser.parse_args()

    print("1) Building image_coords.csv from images in:", args.images_dir)
    csvpath = build_image_coords(images_dir=args.images_dir, out_csv=args.out_csv)
    print("\n2) Building labels from landcover:", args.landcover)
    build_labels(csvpath, landcover_tif=args.landcover, outdir=args.outdir)
    print("\nDone. Files created:")
    print(" -", Path(args.out_csv).resolve())
    print(" -", Path(args.outdir).resolve(), "/ labels.csv, train_labels.csv, test_labels.csv")

if __name__ == "__main__":
    main()
