import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

train_df = pd.read_csv("outputs/train_labels.csv")
test_df = pd.read_csv("outputs/test_labels.csv")

train_counts = train_df['label_name'].value_counts().sort_index()
test_counts = test_df['label_name'].value_counts().sort_index()

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
train_counts.plot(kind='bar', color='skyblue')
plt.title("Train Set Class Distribution")
plt.xticks(rotation=45, ha='right')
plt.subplot(1,2,2)
test_counts.plot(kind='bar', color='salmon')
plt.title("Test Set Class Distribution")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/train_test_distribution.png", dpi=200)
plt.show()
