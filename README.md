# **Scenario 1 – Earth Observation Pipeline**

### This repository contains my complete solution to Scenario 1 of the 6-month AI/ML Internship Selection Task–2025.
The project covers:

Q1: Geospatial preprocessing (grid creation, mapping, filtering)

Q2: Label extraction using ESA Land Cover

Q3: Training & evaluating a ResNet-18 classifier

All tasks have been completed in a fully reproducible pipeline built using PyCharm(I used the community edition of 2023), GeoPandas, Rasterio, PyTorch, and TorchMetrics.

---

## **Repository Structure**
```
Scenario1_Project/
│
│
├── data/
│   ├── rgb/                # Input RGB images 
│   ├── land_cover.tif      # ESA WorldCover
│   └── image_coords.csv
│
├──reports                  # I have provided report in both .pdf and .docx formats
│   ├── Chetan_Kaurav_6_month_intern_selection_task_scenario_1_report.docx
│   └── Chetan_Kaurav_6_month_intern_selection_task_scenario_1_report.pdf
│
│
├── outputs/                # Final results (all deliverables included)
│   ├── grid_60km.geojson
│   ├── grid_centroids.geojson
│   ├── grid_corners.geojson
│   ├── grid_plot_with_corners.png
│   ├── grid_satellite_map.html
│   ├── filtered_image_coords.csv
│   ├── q1_counts.txt
│   ├── labels.csv
│   ├── train_labels.csv
│   ├── test_labels.csv
│   ├── class_distribution.png
│   ├── train_test_distribution.png
│   ├── best_resnet18.pth
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── metrics.csv
│   ├── metrics_summary.txt
│   ├── 5_correct_5_incorrect.png
│   ├── training_log.txt
│   └── class_presence.csv
│
├── scripts/
│    ├── train_eval_resnet.py
│    ├── plot_class_distribution.py
│    ├── plot_train_test_distribution.py
│    ├── Scenario1_Pipeline.py   #It covers the Q1 and Q2 except the class distribution(done in scripts/plot_class_distribution) and imbalance discussion(done in report)
│    ├── make_coords_and_labels.py
│    ├── q1_finalize.py
│    └── Geojson_to_shp.py
│
├──README.md
└──requirements.txt



```
---

## **Installation and Environment setup:**

### Install all dependencies:
```
pip install -r requirements.txt

```
### Recommended:

- Python 3.10 – 3.13
- PyTorch (CPU version unless NVIDIA GPU available)

---

## **How to Run the entire pipeline:**

### Q1: Geospatial Processing
#### Create grids, centroids, and corners:
```
python Scenario1_Pipeline.py grid \
  --shapefile data/delhi_ncr_region_utm44n.shp \
  --outdir outputs
```
#### Finalize grid visualization + satellite basemap + filtering:
```
python q1_finalize.py \
  --grid outputs/grid_60km.geojson \
  --shapefile data/delhi_ncr_region_utm44n.shp \
  --images_csv data/image_coords.csv \
  --outdir outputs
```
#### Q1 deliverables:

| File                         | Description                   |
| ---------------------------- | ----------------------------- |
| `grid_60km.geojson`          | Main grid                     |
| `grid_centroids.geojson`     | Centroids per cell            |
| `grid_corners.geojson`       | Four corners per grid         |
| `grid_plot_with_corners.png` | Static grid visualization     |
| `grid_satellite_map.html`    | Interactive basemap overlay   |
| `filtered_image_coords.csv`  | Filtered coordinate set       |
| `q1_counts.txt`              | Before/after filtering counts |

### **Q2: Label Extraction**

#### Extract ESA land-cover labels:
```
python Scenario1_Pipeline.py prepare_labels \
  --images_csv data/image_coords.csv \
  --landcover data/land_cover.tif \
  --outdir outputs
```
#### Plot label distributions:
```
python scripts/plot_class_distribution.py
python scripts/plot_train_test_distribution.py
```
#### Q2 Deliverables:
| File                          | Description             |
| ----------------------------- | ----------------------- |
| `labels.csv`                  | All labeled samples     |
| `train_labels.csv`            | Training split          |
| `test_labels.csv`             | Test split              |
| `class_distribution.png`      | Label counts            |
| `train_test_distribution.png` | Train/test distribution |


### **Q3 — Model Training & Evaluation**

#### Train ResNet-18:
```
python scripts/train_eval_resnet.py \
  --train_csv outputs/train_labels.csv \
  --test_csv outputs/test_labels.csv \
  --images_dir data/rgb \
  --outdir outputs \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 0
```
#### Q3 Deliverables:
| File                        | Description                        |
| --------------------------- | ---------------------------------- |
| `best_resnet18.pth`         | Best saved model                   |
| `metrics.csv`               | Per-class metrics                  |
| `metrics_summary.txt`       | Custom F1 + TorchMetrics F1        |
| `classification_report.txt` | Full sklearn report                |
| `confusion_matrix.png`      | Confusion heatmap                  |
| `5_correct_5_incorrect.png` | Visual sample predictions          |
| `training_log.txt`          | Epoch-wise training log            |
| `class_presence.csv`        | Which classes appeared in test set |


---

## **Summary:**

This repository presents a complete geospatial + deep learning workflow:

- Accurate grid generation & geospatial filtering

- Patch extraction from ESA WorldCover

- Dominant-class labeling with handling for no-data & uncertainty

- Stratified train/test split

- ResNet-18 classifier with class-weighted loss

- Detailed evaluation using confusion matrices and F1 scores

All deliverables required by Q1, Q2, and Q3 are included in the outputs folder.

---

## **References:**

- ESA WorldCover (2021): https://esa-worldcover.org

- PyTorch: https://pytorch.org

- GeoPandas: https://geopandas.org

- Assignment document: 6-month-intern-selection-task-2025.docx

---

## **Acknowledgements:**

This work was completed using PyCharm Community Edition on:

- Intel i5-12500H (CPU)

- Intel Iris Xe Graphics

- Windows 11 environment

---

## **Thank You for Your Kind Consideration and Time**

- I have sincerely tried my best to ensure that every requirement of the task has been implemented with clarity, correctness, and attention to detail.  
All steps in the pipeline — from geospatial processing to label extraction and model training — have been carefully documented in this README for your convenience.

- If, by any chance, there remains any discrepancy between the README and the actual project files, I kindly request your understanding and forgiveness.  
I truly appreciate the opportunity to work on this assignment and thank you for reviewing my submission.

---
