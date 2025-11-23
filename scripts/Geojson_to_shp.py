import geopandas as gpd

# Load your GeoJSON
gdf = gpd.read_file("data\delhi_ncr_region.geojson")

# Save as Shapefile
gdf.to_file("delhi_ncr_region_shapefile.shp", driver="ESRI Shapefile")
