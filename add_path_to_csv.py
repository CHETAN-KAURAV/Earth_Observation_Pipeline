from pathlib import Path
import pandas as pd

df = pd.read_csv("data/image_coords.csv")
df['path'] = df['filename'].apply(lambda fn: str(Path('data/rgb') / fn))
df.to_csv("data/image_coords_with_path.csv", index=False)
