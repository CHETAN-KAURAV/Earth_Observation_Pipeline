import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/labels.csv")
counts = df['label_name'].value_counts()
plt.figure(figsize=(12,4))
counts.plot(kind='bar')
plt.title("Class distribution (all labels)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
Path("outputs/class_distribution.png").write_bytes(b'') if False else None
plt.savefig("outputs/class_distribution.png", dpi=200)
plt.show()
