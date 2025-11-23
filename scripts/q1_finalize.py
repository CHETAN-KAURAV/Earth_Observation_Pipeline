#!/usr/bin/env python3
"""
q1_finalize.py

After running Scenario1_Pipeline.py grid, run this script to:
 - Save grid corners and centroids as GeoJSON
 - Produce an interactive satellite basemap overlay (HTML) using geemap
 - Filter images by grid cells (using data/image_coords.csv) and save filtered CSV + counts
 - Produce a matplotlib PNG showing grid + corners + centroids

Usage:
 python q1_finalize.py \
   --grid outputs/grid_60km.geojson \
   --shapefile data/delhi_ncr_region_utm44n.shp \
   --images_csv data/image_coords.csv \
   --outdir outputs
"""
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import sys

def save_corners_and_centroids(grid_gdf: gpd.GeoDataFrame, outdir: Path):
    # corners
    corner_rows = []
    for i, geom in enumerate(grid_gdf.geometry):
        try:
            # exterior coords: last == first (closed); take first 4 unique corners
            coords = list(geom.exterior.coords)[:4]
        except Exception:
            # fallback: try to get bounds
            minx, miny, maxx, maxy = geom.bounds
            coords = [(minx,miny), (maxx,miny), (maxx,maxy), (minx,maxy)]
        for j, (x, y) in enumerate(coords):
            corner_rows.append({'grid_id': i, 'corner_id': j, 'geometry': Point(x, y)})
    corners_gdf = gpd.GeoDataFrame(corner_rows, crs=grid_gdf.crs)
    corners_out = outdir / 'grid_corners.geojson'
    corners_gdf.to_file(corners_out, driver='GeoJSON')
    print(f"Saved corners GeoJSON -> {corners_out}")

    # centroids
    centroids = grid_gdf.copy()
    centroids['geometry'] = centroids.geometry.centroid
    centroids_out = outdir / 'grid_centroids.geojson'
    centroids.to_file(centroids_out, driver='GeoJSON')
    print(f"Saved centroids GeoJSON -> {centroids_out}")

    return corners_gdf, centroids

def save_basemap_html(grid_gdf: gpd.GeoDataFrame, area_gdf: gpd.GeoDataFrame, out_html: Path):
    try:
        import geemap
    except Exception as e:
        print("geemap not available. Install with `pip install geemap` to create the interactive HTML basemap.")
        return False

    # convert to 4326 for web maps
    try:
        grid_4326 = grid_gdf.to_crs(epsg=4326)
        area_4326 = area_gdf.to_crs(epsg=4326)
    except Exception:
        grid_4326 = grid_gdf.copy()
        area_4326 = area_gdf.copy()

    # create map around study area center
    try:
        centroid = area_4326.geometry.centroid.unary_union
        center_lat = centroid.y if hasattr(centroid, 'y') else area_4326.geometry.centroid.y.mean()
        center_lon = centroid.x if hasattr(centroid, 'x') else area_4326.geometry.centroid.x.mean()
    except Exception:
        center_lat, center_lon = 28.65, 77.2  # fallback: approx Delhi
    m = geemap.Map(center=[center_lat, center_lon], zoom=10)
    m.add_basemap("SATELLITE")
    m.add_gdf(area_4326, layer_name="Study Area", info_mode='on')
    m.add_gdf(grid_4326, layer_name="60km Grid")
    # add centroids as a layer if present
    try:
        centroids = grid_4326.copy()
        centroids['geometry'] = centroids.geometry.centroid
        m.add_gdf(centroids, layer_name="Grid centroids")
    except Exception:
        pass

    # save html
    m.to_html(str(out_html))
    print(f"Saved interactive basemap HTML -> {out_html}")
    return True

def filter_images_by_grid(images_csv: Path, grid_gdf: gpd.GeoDataFrame, outdir: Path, images_crs=4326):
    if not images_csv.exists():
        print(f"Images CSV not found: {images_csv}")
        return None, 0, 0
    df = pd.read_csv(images_csv)
    if not {'filename', 'lat', 'lon'}.issubset(df.columns):
        # accept lon/lat order if provided as lon,lat instead
        if {'filename','lon','lat'}.issubset(df.columns):
            df = df.rename(columns={'lon':'lon','lat':'lat'})
        else:
            raise ValueError("images_csv must contain columns: filename,lat,lon (or filename,lon,lat)")
    # build points (note order: lon,x then lat,y)
    pts = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs=f'epsg:{images_crs}')
    # transform to grid CRS
    grid_crs_epsg = grid_gdf.crs.to_epsg() if grid_gdf.crs is not None else None
    if grid_crs_epsg is not None:
        pts_utm = pts.to_crs(epsg=grid_crs_epsg)
    else:
        pts_utm = pts.copy()

    # spatial join with grid
    joined = pts_utm.sjoin(grid_gdf[['geometry']], how='left', predicate='intersects')
    joined.rename(columns={'index_right':'grid_index'}, inplace=True)

    before = len(df)
    after = joined['grid_index'].notna().sum()

    filtered = joined[joined['grid_index'].notna()].copy()
    filtered = filtered[['filename','lat','lon','grid_index','geometry']]

    out_csv = outdir / 'filtered_image_coords.csv'
    filtered[['filename','lat','lon']].to_csv(out_csv, index=False)
    counts_txt = outdir / 'q1_counts.txt'
    with open(counts_txt, 'w') as fh:
        fh.write(f"images_before={before}\nimages_after={after}\n")
    print(f"Saved filtered CSV -> {out_csv}")
    print(f"Saved counts -> {counts_txt}")
    return filtered, before, after

def plot_grid_with_corners(area_gdf, grid_gdf, corners_gdf, centroids_gdf, out_png: Path):
    fig, ax = plt.subplots(figsize=(10,10))
    try:
        area_gdf.to_crs(grid_gdf.crs).plot(ax=ax, facecolor='none', edgecolor='black')
    except Exception:
        area_gdf.plot(ax=ax, facecolor='none', edgecolor='black')
    grid_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=0.5)
    # corners
    corners_gdf.plot(ax=ax, markersize=2, color='green', label='corners')
    # centroids
    centroids_gdf.plot(ax=ax, markersize=5, color='blue', label='centroids')
    ax.set_title('Grid with corners & centroids')
    plt.axis('equal')
    plt.legend()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved grid plot with corners -> {out_png}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', default='outputs/grid_60km.geojson', help='Path to grid geojson (output of Scenario1_Pipeline grid)')
    p.add_argument('--shapefile', default='data/delhi_ncr_region_utm44n.shp', help='Path to original area shapefile (used for basemap overlay)')
    p.add_argument('--images_csv', default='data/image_coords.csv', help='Path to image coords CSV (filename,lat,lon)')
    p.add_argument('--outdir', default='outputs', help='Output directory')
    args = p.parse_args()

    grid_path = Path(args.grid)
    shp_path = Path(args.shapefile)
    images_csv = Path(args.images_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not grid_path.exists():
        print(f"Grid file not found: {grid_path}. Run Scenario1_Pipeline.py grid first.")
        sys.exit(1)

    print("Loading grid:", grid_path)
    grid_gdf = gpd.read_file(grid_path)
    print("Grid loaded. cells:", len(grid_gdf))

    # attempt to load shapefile; if fails create a tiny gdf from grid union
    if shp_path.exists():
        area_gdf = gpd.read_file(shp_path)
    else:
        print(f"Shapefile {shp_path} not found. Creating area from grid union for plotting.")
        area_gdf = gpd.GeoDataFrame(geometry=[grid_gdf.unary_union], crs=grid_gdf.crs)

    # Save corners and centroids
    corners_gdf, centroids_gdf = save_corners_and_centroids(grid_gdf, outdir)

    # Save interactive basemap (HTML)
    html_out = outdir / 'grid_satellite_map.html'
    saved = save_basemap_html(grid_gdf, area_gdf, html_out)
    if not saved:
        print("geemap basemap HTML was not created (geemap may be missing).")

    # Filter images by grid and save counts
    try:
        filtered, before, after = filter_images_by_grid(images_csv, grid_gdf, outdir)
    except Exception as e:
        print("Error filtering images:", e)
        filtered, before, after = None, 0, 0

    # Plot static PNG with corners and centroids
    try:
        plot_out = outdir / 'grid_plot_with_corners.png'
        plot_grid_with_corners(area_gdf, grid_gdf, corners_gdf, centroids_gdf, plot_out)
    except Exception as e:
        print("Error creating static plot:", e)

    print("\nDone. Outputs written to:", outdir.resolve())
    print("- grid corners:", outdir / 'grid_corners.geojson')
    print("- grid centroids:", outdir / 'grid_centroids.geojson')
    print("- interactive HTML (satellite):", html_out)
    print("- filtered image CSV:", outdir / 'filtered_image_coords.csv')
    print("- q1 counts text:", outdir / 'q1_counts.txt')

if __name__ == "__main__":
    main()
