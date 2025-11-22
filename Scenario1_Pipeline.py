#!/usr/bin/env python3
"""
Scenario1_Pipeline.py — corrected full pipeline (Q1 + helpers for Q2/Q3)

Usage:
  python Scenario1_Pipeline.py grid --shapefile data/delhi_ncr_region_utm44n.shp --outdir outputs
  python Scenario1_Pipeline.py prepare_labels --images_csv data/image_coords.csv --landcover data/land_cover.tif --outdir outputs
  python Scenario1_Pipeline.py train --labels_csv outputs/labels.csv --images_dir data/rgb --outdir outputs --epochs 5

Defaults are set to the dataset layout you confirmed:
  shapefile base: delhi_ncr_region_utm44n.*
  landcover: data/land_cover.tif
  images folder: data/rgb
  image coords csv: data/image_coords.csv

This script:
 - creates a 60x60 km grid (EPSG:32644)
 - saves grid polygons and separate centroid & corner GeoJSONs
 - creates a static PNG plot and optional geemap HTML (satellite basemap)
 - filters images by grid and saves counts + filtered CSV
 - prepares labels from landcover (mode) and train/test split
 - simple ResNet18 training routine (toy default)

Dependencies:
 geopandas, rasterio, pyproj, numpy, pandas, matplotlib, shapely, geemap (optional), torch, torchvision, scikit-learn, pillow
"""
import argparse
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

# Torch optional parts (only required for train)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------
# Defaults (use your confirmed names)
# ----------------------
DEFAULT_SHAPEFILE = "data/delhi_ncr_region_utm44n.shp"
DEFAULT_LC = "data/land_cover.tif"
DEFAULT_IMAGES_DIR = "data/rgb"
DEFAULT_IMAGE_COORDS = "data/image_coords.csv"
DEFAULT_OUTDIR = "outputs"

# ----------------------
# Utilities
# ----------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------
# Q1: Grid creation + saving centroids/corners + basemap overlay + filter images
# ----------------------
def create_grid_from_shapefile(shapefile_path,
                               cell_size_m=60000,
                               target_epsg=32644,
                               out_dir=DEFAULT_OUTDIR,
                               save_basemap_html=True,
                               images_csv=DEFAULT_IMAGE_COORDS):
    """
    Read shapefile, reproject to target_epsg (meters), create uniform grid cells sized cell_size_m,
    save:
      - outputs/grid_60km.geojson (polygons)
      - outputs/grid_centroids.geojson (points)
      - outputs/grid_corners.geojson (points)
      - outputs/grid_plot.png (static)
      - outputs/grid_satellite_map.html (interactive, optional - requires geemap)
      - outputs/filtered_image_coords.csv and outputs/q1_counts.txt (if images_csv exists)
    Returns (grid_gdf, centroids_gdf, corners_gdf)
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    shp = Path(shapefile_path)
    if not shp.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp}")

    gdf = gpd.read_file(shp)
    print(f"Loaded shapefile: {shp} with {len(gdf)} features")

    # Reproject to UTM/metric CRS for 60km grid
    gdf_utm = gdf.to_crs(epsg=target_epsg)
    xmin, ymin, xmax, ymax = gdf_utm.total_bounds
    print("bounds (UTM):", xmin, ymin, xmax, ymax)

    # Build grid cells
    xs = np.arange(math.floor(xmin), math.ceil(xmax), cell_size_m)
    ys = np.arange(math.floor(ymin), math.ceil(ymax), cell_size_m)
    grid_polys = [Polygon([(x, y), (x + cell_size_m, y),
                           (x + cell_size_m, y + cell_size_m), (x, y + cell_size_m)])
                  for x in xs for y in ys]
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_polys}, crs=f"EPSG:{target_epsg}")

    # Intersect with study area: use union_all() prefered; fallback if not available
    try:
        union_geom = gdf_utm.union_all()
    except Exception:
        union_geom = gdf_utm.unary_union
    grid_gdf['intersects'] = grid_gdf.geometry.apply(lambda p: p.intersects(union_geom))
    grid_gdf = grid_gdf[grid_gdf['intersects']].copy().reset_index(drop=True)
    grid_gdf.drop(columns=['intersects'], inplace=True)

    # Compute bounds tuple column (minx,miny,maxx,maxy)
    bounds_df = grid_gdf.geometry.bounds
    grid_gdf['bounds'] = bounds_df.apply(lambda r: (r['minx'], r['miny'], r['maxx'], r['maxy']), axis=1)

    # Create separate centroids GeoDataFrame (so grid_gdf keeps single geometry column)
    centroids = [geom.centroid for geom in grid_gdf.geometry]
    centroids_gdf = gpd.GeoDataFrame({'geometry': centroids}, crs=grid_gdf.crs)

    # Create corners GeoDataFrame
    corner_rows = []
    for i, geom in enumerate(grid_gdf.geometry):
        try:
            coords = list(geom.exterior.coords)[:4]
        except Exception:
            minx, miny, maxx, maxy = geom.bounds
            coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        for j, (x, y) in enumerate(coords):
            corner_rows.append({'grid_id': i, 'corner_id': j, 'geometry': Point(x, y)})
    corners_gdf = gpd.GeoDataFrame(corner_rows, crs=grid_gdf.crs)

    # Save grid polygons (single geometry column)
    grid_out = Path(out_dir) / 'grid_60km.geojson'
    # Drop any accidental extra geometry-like columns just in case
    to_drop = [c for c in grid_gdf.columns if c != 'geometry' and getattr(grid_gdf[c], 'dtype', None) == 'geometry']
    if to_drop:
        grid_gdf = grid_gdf.drop(columns=to_drop)
    grid_gdf.to_file(grid_out, driver='GeoJSON')
    print(f"Saved grid polygons -> {grid_out}")

    # Save centroids + corners
    centroids_out = Path(out_dir) / 'grid_centroids.geojson'
    centroids_gdf.to_file(centroids_out, driver='GeoJSON')
    print(f"Saved centroids -> {centroids_out}")

    corners_out = Path(out_dir) / 'grid_corners.geojson'
    corners_gdf.to_file(corners_out, driver='GeoJSON')
    print(f"Saved corners -> {corners_out}")

    # Static plot (UTM)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_utm.plot(ax=ax, facecolor='none', edgecolor='black')
    grid_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=0.5)
    centroids_gdf.plot(ax=ax, markersize=6, color='blue', label='centroids')
    corners_gdf.plot(ax=ax, markersize=2, color='green', label='corners')
    ax.set_title('Delhi-NCR with 60x60 km grid (EPSG:%d)' % target_epsg)
    plt.axis('equal')
    out_png = Path(out_dir) / 'grid_plot_with_corners.png'
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved static plot -> {out_png}")

    # Optional: interactive basemap HTML using geemap (converted to 4326)
    html_out = Path(out_dir) / 'grid_satellite_map.html'
    try:
        import geemap
        grid_4326 = grid_gdf.to_crs(epsg=4326)
        area_4326 = gdf_utm.to_crs(epsg=4326)
        m = geemap.Map(center=[area_4326.geometry.centroid.y.mean(), area_4326.geometry.centroid.x.mean()], zoom=10)
        m.add_basemap("SATELLITE")
        m.add_gdf(area_4326, layer_name="Study area")
        m.add_gdf(grid_4326, layer_name="60km grid")
        centroids_4326 = centroids_gdf.to_crs(epsg=4326)
        m.add_gdf(centroids_4326, layer_name="centroids")
        m.to_html(str(html_out))
        print(f"Saved interactive basemap -> {html_out}")
    except Exception as e:
        print("geemap not available or failed — skipping HTML basemap. Install geemap to enable. (Error:", e, ")")

    # If images CSV exists, filter images by grid and save counts
    images_csv_path = Path(images_csv)
    if images_csv_path.exists():
        filtered_df, before, after = filter_images_within_grid(images_csv_path, grid_gdf, images_crs=4326, grid_crs=target_epsg)
        filtered_out = Path(out_dir) / 'filtered_image_coords.csv'
        filtered_df[['filename', 'lon', 'lat']].to_csv(filtered_out, index=False)
        counts_txt = Path(out_dir) / 'q1_counts.txt'
        with open(counts_txt, 'w') as fh:
            fh.write(f"images_before={before}\nimages_after={after}\n")
        print(f"Saved filtered CSV -> {filtered_out} and counts -> {counts_txt}")
    else:
        print(f"Images CSV not found at {images_csv_path}; skipping filtering step.")

    return grid_gdf, centroids_gdf, corners_gdf

def latlon_to_point_gdf(df_coords, lon_col='lon', lat_col='lat', crs_epsg=4326):
    gdf = gpd.GeoDataFrame(df_coords.copy())
    gdf['geometry'] = gdf.apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
    gdf.set_crs(epsg=crs_epsg, inplace=True)
    return gdf

def filter_images_within_grid(images_csv, grid_gdf, images_crs=4326, grid_crs=32644):
    df = pd.read_csv(images_csv)
    # support both (lat,lon) and (lon,lat) column orders robustly
    if {'filename','lat','lon'}.issubset(df.columns):
        pass
    elif {'filename','lon','lat'}.issubset(df.columns):
        df = df.rename(columns={'lon':'lon','lat':'lat'})
    else:
        raise ValueError("images_csv must contain filename,lat,lon (or filename,lon,lat)")

    pts = latlon_to_point_gdf(df, lon_col='lon', lat_col='lat', crs_epsg=images_crs)
    pts_utm = pts.to_crs(epsg=grid_crs)
    pts_utm = pts_utm.sjoin(grid_gdf[['geometry']], how='left', predicate='intersects')
    pts_utm.rename(columns={'index_right':'grid_index'}, inplace=True)
    before = len(df)
    after = pts_utm['grid_index'].notna().sum()
    keep = pts_utm[pts_utm['grid_index'].notna()].copy()
    return keep, before, after

# ----------------------
# Q2: Label construction from landcover
# ----------------------
def extract_label_from_raster(lc_raster_path, lon, lat, out_size=128, raster_crs=4326, nodata_val=0):
    lc_raster_path = Path(lc_raster_path)
    if not lc_raster_path.exists():
        raise FileNotFoundError(f"Landcover raster not found: {lc_raster_path}")
    with rasterio.open(lc_raster_path) as src:
        src_epsg = None
        try:
            src_epsg = src.crs.to_epsg()
        except Exception:
            src_epsg = None
        if src_epsg is not None and src_epsg != raster_crs:
            transformer = Transformer.from_crs('epsg:4326', src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        try:
            col, row = src.index(x, y)
        except Exception:
            return np.full((out_size, out_size), nodata_val, dtype=np.int32)
        half = out_size // 2
        col_start = col - half
        row_start = row - half
        window = Window(col_start, row_start, out_size, out_size)
        arr = src.read(1, window=window, boundless=True, fill_value=nodata_val)
    return arr

def compute_mode_label(patch, nodata_value=0):
    flat = patch.flatten()
    valid = flat[flat != nodata_value]
    if valid.size == 0:
        return None, 0.0
    counts = np.bincount(valid.astype(np.int64))
    top = counts.argmax()
    cnt = counts[top]
    frac = cnt / flat.size
    return int(top), float(frac)

def build_label_dataset(images_csv, landcover_path, out_dir=DEFAULT_OUTDIR, out_size=128, nodata_value=0, map_esa_to_11=None):
    ensure_dir(out_dir)
    df = pd.read_csv(images_csv)
    rows = []
    for idx, r in df.iterrows():
        fname = r['filename']
        lat = float(r['lat'])
        lon = float(r['lon'])
        patch = extract_label_from_raster(landcover_path, lon, lat, out_size=out_size, nodata_val=nodata_value)
        label_val, dominance = compute_mode_label(patch, nodata_value)
        rows.append({'filename': fname, 'lat': lat, 'lon': lon, 'label_val': label_val, 'dominance': dominance})
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx+1}/{len(df)} images")
    out_df = pd.DataFrame(rows)
    if map_esa_to_11 is not None:
        out_df['label_name'] = out_df['label_val'].map(map_esa_to_11)
    else:
        out_df['label_name'] = out_df['label_val'].astype(str)
    # mark uncertain
    out_df['label_name'] = out_df.apply(lambda x: 'uncertain' if (pd.isna(x['label_val']) or x['dominance'] < 0.2) else x['label_name'], axis=1)
    labels_csv = Path(out_dir) / 'labels.csv'
    out_df.to_csv(labels_csv, index=False)
    print(f"Saved labels -> {labels_csv} ({len(out_df)} rows)")
    # split
    try:
        train, test = train_test_split(out_df, test_size=0.4, random_state=42, stratify=out_df['label_name'])
    except Exception:
        train, test = train_test_split(out_df, test_size=0.4, random_state=42)
    train.to_csv(Path(out_dir) / 'train_labels.csv', index=False)
    test.to_csv(Path(out_dir) / 'test_labels.csv', index=False)
    print(f"Saved train/test splits -> {len(train)} train, {len(test)} test")
    return labels_csv, Path(out_dir) / 'train_labels.csv', Path(out_dir) / 'test_labels.csv'

# ----------------------
# Q3: Simple ResNet trainer (keeps existing behaviour)
# ----------------------
class ImagePatchDataset(Dataset if TORCH_AVAILABLE else object):
    def __init__(self, images_dir, labels_df, label2idx, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.label2idx = label2idx
    def __len__(self):
        return len(self.labels_df)
    def __getitem__(self, idx):
        rec = self.labels_df.iloc[idx]
        img_path = self.images_dir / rec['filename']
        if not img_path.exists():
            img = Image.new('RGB', (128,128), (0,0,0))
        else:
            img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_idx = self.label2idx.get(rec['label_name'], self.label2idx.get('uncertain', 0))
        return img, label_idx

def train_resnet(train_csv, test_csv, images_dir, out_dir=DEFAULT_OUTDIR, epochs=5, batch_size=32, lr=1e-3):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install torch and torchvision to train.")
    ensure_dir(out_dir)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    labels = sorted(train_df['label_name'].unique())
    if 'uncertain' not in labels:
        labels.append('uncertain')
    label2idx = {lab:i for i, lab in enumerate(labels)}
    idx2label = {i:lab for lab,i in label2idx.items()}
    print("Labels:", label2idx)
    transform_train = transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    train_ds = ImagePatchDataset(images_dir, train_df, label2idx, transform=transform_train)
    test_ds = ImagePatchDataset(images_dir, test_df, label2idx, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(label2idx))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labs in train_loader:
            imgs = imgs.to(device)
            labs = labs.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, labs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{epochs} train loss: {epoch_loss:.4f}")
        # eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs = imgs.to(device)
                outs = model(imgs)
                preds = outs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labs.numpy().tolist())
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Validation acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state': model.state_dict(), 'label2idx': label2idx}, Path(out_dir)/'best_resnet18.pth')
            print("Saved best model")
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(all_labels, all_preds, target_names=[idx2label[i] for i in sorted(idx2label.keys())]))
    return Path(out_dir)/'best_resnet18.pth'

# ----------------------
# CLI
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Scenario1 Pipeline")
    sub = parser.add_subparsers(dest='cmd')
    p_grid = sub.add_parser('grid')
    p_grid.add_argument('--shapefile', default=DEFAULT_SHAPEFILE)
    p_grid.add_argument('--cell_km', type=float, default=60.0)
    p_grid.add_argument('--outdir', default=DEFAULT_OUTDIR)
    p_prep = sub.add_parser('prepare_labels')
    p_prep.add_argument('--images_csv', default=DEFAULT_IMAGE_COORDS)
    p_prep.add_argument('--landcover', default=DEFAULT_LC)
    p_prep.add_argument('--outdir', default=DEFAULT_OUTDIR)
    p_prep.add_argument('--out_size', type=int, default=128)
    p_train = sub.add_parser('train')
    p_train.add_argument('--labels_csv', required=True)
    p_train.add_argument('--train_csv', required=False)
    p_train.add_argument('--test_csv', required=False)
    p_train.add_argument('--images_dir', default=DEFAULT_IMAGES_DIR)
    p_train.add_argument('--outdir', default=DEFAULT_OUTDIR)
    p_train.add_argument('--epochs', type=int, default=5)
    p_train.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    if args.cmd == 'grid':
        create_grid_from_shapefile(args.shapefile, cell_size_m=int(args.cell_km*1000), target_epsg=32644, out_dir=args.outdir)
    elif args.cmd == 'prepare_labels':
        esa_to_11 = {
            10: 'Tree', 20: 'Shrub', 30: 'Grass', 40: 'Cropland', 50: 'Built-up',
            60: 'Bare', 70: 'Snow', 80: 'Water', 90: 'HerbaceousWetland', 95: 'Mangroves', 100: 'SparseVegetation'
        }
        build_label_dataset(args.images_csv, args.landcover, out_dir=args.outdir, out_size=args.out_size, nodata_value=0, map_esa_to_11=esa_to_11)
    elif args.cmd == 'train':
        labels_csv = args.labels_csv
        train_csv = args.train_csv
        test_csv = args.test_csv
        if train_csv is None or test_csv is None:
            base = Path(labels_csv).parent
            train_csv = train_csv or base / 'train_labels.csv'
            test_csv = test_csv or base / 'test_labels.csv'
        train_resnet(train_csv, test_csv, args.images_dir, out_dir=args.outdir, epochs=args.epochs, batch_size=args.batch_size)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
