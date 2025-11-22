import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/labels.csv')
counts = df['label_name'].value_counts()
counts.plot(kind='bar', figsize=(10,5))
plt.title('Class distribution (labels.csv)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/class_distribution.png', dpi=200)
plt.show()
