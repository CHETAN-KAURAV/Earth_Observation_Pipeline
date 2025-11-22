import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

train_df = pd.read_csv("outputs/train_labels.csv")
test_df = pd.read_csv("outputs/test_labels.csv")

train_counts = train_df['label_name'].value_counts().sort_index()
test_counts = test_df['label_name'].value_counts().sort_index()

plt.figure(figsize=(14, 6))

# --- TRAIN PLOT ---
plt.subplot(1, 2, 1)
train_counts.plot(kind='bar', color='skyblue')
plt.title("Train Set Class Distribution")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")

# --- TEST PLOT ---
plt.subplot(1, 2, 2)
test_counts.plot(kind='bar', color='salmon')
plt.title("Test Set Class Distribution")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")

plt.tight_layout()

out_path = Path("outputs/train_test_distribution.png")
plt.savefig(out_path, dpi=220)
plt.show()
print("Saved comparison plot to:", out_path)
