#!/usr/bin/env python3
"""
Robust ResNet18 train+eval script for Scenario1_Project.

Saves:
 - outputs/best_resnet18.pth
 - outputs/metrics.csv (per-class metrics for present labels)
 - outputs/class_presence.csv (shows which classes are present)
 - outputs/metrics_summary.txt (macro f1s + present labels)
 - outputs/confusion_matrix.png
 - outputs/5_correct_5_incorrect.png
 - outputs/classification_report.txt
 - outputs/training_log.txt

Usage:
python scripts/train_eval_resnet.py \
  --train_csv outputs/train_labels.csv --test_csv outputs/test_labels.csv \
  --images_dir data/rgb --outdir outputs --epochs 10 --batch_size 32 --num_workers 0
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings

# Optional: torchmetrics (handle API differences)
try:
    from torchmetrics import F1Score
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False

# ---------------------------
# Dataset
# ---------------------------
class ImagePatchDataset(Dataset):
    def __init__(self, csv_file, images_dir, label2idx, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_path = self.images_dir / r['filename']
        if not img_path.exists():
            # fallback black image; also log later if many missing
            img = Image.new('RGB', (128,128), (0,0,0))
        else:
            img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label2idx.get(r['label_name'], self.label2idx.get('uncertain', 0))
        return img, int(label), r['filename']

# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_label_maps(train_csv):
    df = pd.read_csv(train_csv)
    labels = sorted(df['label_name'].unique())
    if 'uncertain' not in labels:
        labels.append('uncertain')
    label2idx = {l:i for i,l in enumerate(labels)}
    idx2label = {i:l for l,i in label2idx.items()}
    return label2idx, idx2label

def plot_confusion(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2. if cm.size else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i,j]), 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    try:
        fig.savefig(outpath, dpi=200)
    except Exception as e:
        print("Warning: could not save confusion matrix:", e)
    plt.close(fig)

def plot_examples(img_dir, filenames, ytrue, ypred, idx2label, outpath, n_correct=5, n_incorrect=5):
    correct_idx = [i for i,(t,p) in enumerate(zip(ytrue, ypred)) if t==p]
    incorrect_idx = [i for i,(t,p) in enumerate(zip(ytrue, ypred)) if t!=p]
    sel = correct_idx[:n_correct] + incorrect_idx[:n_incorrect]
    total = len(sel)
    if total == 0:
        print("No examples to plot (empty predictions).")
        return
    cols = 5
    rows = (total + cols -1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')
    for k,i in enumerate(sel):
        fn = filenames[i]
        img_path = Path(img_dir)/fn
        if img_path.exists():
            im = Image.open(img_path).convert('RGB')
        else:
            im = Image.new('RGB',(128,128),(0,0,0))
        ax = axes[k]
        ax.imshow(im)
        true_label = idx2label.get(ytrue[i], str(ytrue[i]))
        pred_label = idx2label.get(ypred[i], str(ypred[i]))
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    try:
        fig.savefig(outpath, dpi=200)
    except Exception as e:
        print("Warning: could not save examples plot:", e)
    plt.close(fig)

# ---------------------------
# Train + Evaluate
# ---------------------------
def train_and_evaluate(train_csv, test_csv, images_dir, outdir, epochs=10, batch_size=32, lr=1e-3, seed=42, num_workers=0):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # Build label maps from training CSV
    label2idx, idx2label = build_label_maps(train_csv)
    num_classes = len(label2idx)
    print("Labels:", label2idx)

    # transforms
    transform_train = transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

    # datasets
    train_ds = ImagePatchDataset(train_csv, images_dir, label2idx, transform=transform_train)
    test_ds  = ImagePatchDataset(test_csv,  images_dir, label2idx, transform=transform_test)

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # compute class counts with minlength to ensure shape==num_classes
    y_train = pd.read_csv(train_csv)['label_name'].map(label2idx).to_numpy(dtype=np.int64)
    class_counts = np.bincount(y_train, minlength=num_classes)
    # avoid zero counts causing inf weights
    eps = 1e-6
    class_counts_safe = class_counts.astype(np.float64).copy()
    class_counts_safe[class_counts_safe == 0] = eps
    class_weights = (1.0 / class_counts_safe) * (class_counts_safe.sum() / class_counts_safe.size)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    history = []
    training_log = outdir/'training_log.txt'
    # ensure no existing training_log locks/overwrite
    try:
        if training_log.exists():
            training_log.unlink()
    except Exception:
        pass

    all_preds = None
    all_labels = None
    all_filenames = None

    with open(training_log, 'w') as flog:
        for epoch in range(1, epochs+1):
            model.train()
            running_loss = 0.0
            bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
            for imgs, labs, _ in bar:
                imgs = imgs.to(device); labs = labs.to(device)
                optimizer.zero_grad()
                outs = model(imgs)
                loss = criterion(outs, labs)
                loss.backward(); optimizer.step()
                running_loss += loss.item()*imgs.size(0)
            train_loss = running_loss / max(len(train_ds), 1)
            # evaluation
            model.eval()
            preds_list=[]; labels_list=[]; filenames_list=[]
            with torch.no_grad():
                for imgs, labs, fnames in test_loader:
                    imgs = imgs.to(device)
                    outs = model(imgs)
                    preds = outs.argmax(dim=1).cpu().numpy()
                    preds_list.extend(preds.tolist())
                    labels_list.extend(labs.numpy().tolist())
                    filenames_list.extend(fnames)
            acc = float(np.mean(np.array(preds_list) == np.array(labels_list))) if len(labels_list)>0 else 0.0
            history.append({'epoch':epoch,'train_loss':train_loss,'val_acc':acc})
            flog.write(f"{epoch},{train_loss},{acc}\n"); flog.flush()
            print(f"Epoch {epoch} -> train_loss: {train_loss:.4f}, val_acc: {acc:.4f}")
            # save best by val acc
            if acc > best_acc:
                best_acc = acc
                try:
                    torch.save({'model_state':model.state_dict(), 'label2idx':label2idx}, outdir/'best_resnet18.pth')
                    print("Saved best model")
                except Exception as e:
                    print("Warning: could not save model:", e)
            # keep last epoch results for final reporting
            all_preds = preds_list
            all_labels = labels_list
            all_filenames = filenames_list

    # final reporting - ensure lists exist
    if all_preds is None or all_labels is None or all_filenames is None:
        print("No predictions were generated – aborting final metrics.")
        return None

    # build labels list for all classes
    labels_list = [idx2label[i] for i in sorted(idx2label.keys())]

    # determine present labels (in true or pred)
    present_labels = sorted(list(set(all_labels) | set(all_preds)))
    present_names = [idx2label[i] for i in present_labels]

    # confusion matrix for full class set (rows true, cols pred)
    try:
        cm_full = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels_list))))
    except Exception:
        # fallback to present labels only
        cm_full = confusion_matrix(all_labels, all_preds, labels=present_labels)
    # save confusion matrix (labels may include absent classes; plot_confusion handles shape)
    plot_confusion(cm_full, labels_list, outdir/'confusion_matrix.png')

    # per-class metrics for present labels
    p, r, f, sup = precision_recall_fscore_support(all_labels, all_preds, labels=present_labels, zero_division=0)
    rows = []
    for i, lab_idx in enumerate(present_labels):
        rows.append({
            'label_idx': int(lab_idx),
            'label_name': labels_list[lab_idx],
            'precision': float(p[i]),
            'recall': float(r[i]),
            'f1': float(f[i]),
            'support': int(sup[i])
        })
    metrics_df = pd.DataFrame(rows)
    metrics_csv = outdir/'metrics.csv'
    try:
        if metrics_csv.exists():
            metrics_csv.unlink()
    except Exception:
        pass
    metrics_df.to_csv(metrics_csv, index=False)

    # macro f1 over present labels
    macro_f1 = metrics_df['f1'].mean() if not metrics_df.empty else 0.0

    # compute torchmetrics macro f1 if available (robust to API)
    torchmetrics_f1 = None
    if _HAS_TORCHMETRICS:
        try:
            f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
        except TypeError:
            try:
                f1_metric = F1Score(num_classes=num_classes, average='macro').to(device)
            except Exception:
                f1_metric = None
        if f1_metric is not None:
            try:
                y_pred_tensor = torch.tensor(all_preds, dtype=torch.long).to(device)
                y_true_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)
                with torch.no_grad():
                    torchmetrics_f1 = float(f1_metric(y_pred_tensor, y_true_tensor))
            except Exception as e:
                print("Warning: torchmetrics F1 calculation failed:", e)
                torchmetrics_f1 = None

    # write metrics summary
    summary_path = outdir/'metrics_summary.txt'
    try:
        if summary_path.exists():
            summary_path.unlink()
    except Exception:
        pass
    with open(summary_path, 'w') as fh:
        fh.write(f"custom_macro_f1_present_labels={macro_f1}\n")
        fh.write(f"torchmetrics_macro_f1={torchmetrics_f1}\n")
        fh.write(f"present_labels={present_labels}\n")
        fh.write(f"present_label_names={present_names}\n")

    # safe classification_report: request exactly present_labels + names
    try:
        report = classification_report(all_labels, all_preds, labels=present_labels, target_names=present_names, zero_division=0)
    except Exception:
        try:
            report = classification_report(all_labels, all_preds, zero_division=0)
        except Exception as e:
            report = f"classification_report_error: {e}"
    with open(outdir/'classification_report.txt','w') as fh:
        fh.write(report)

    # save which classes are present (useful for report)
    class_presence = []
    for i, name in enumerate(labels_list):
        class_presence.append({'label_idx': i, 'label_name': name, 'present_in_test': int(i in present_labels), 'train_count': int(class_counts[i] if i < len(class_counts) else 0)})
    pd.DataFrame(class_presence).to_csv(outdir/'class_presence.csv', index=False)

    # plot examples
    plot_examples(images_dir, all_filenames, all_labels, all_preds, {v:k for k,v in label2idx.items()}, outdir/'5_correct_5_incorrect.png', n_correct=5, n_incorrect=5)

    print("Saved metrics to:", metrics_csv)
    print("Saved confusion matrix, examples, and classification report.")
    return outdir/'best_resnet18.pth'

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--test_csv', required=True)
    p.add_argument('--images_dir', required=True)
    p.add_argument('--outdir', default='outputs')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        train_and_evaluate(args.train_csv, args.test_csv, args.images_dir, args.outdir,
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           seed=args.seed, num_workers=args.num_workers)
    except Exception as e:
        print("Fatal error during training/eval:", e)
        raise
