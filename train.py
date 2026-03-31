#!/usr/bin/env python
"""
Train the cys3state classifier from labeled cysteine data.

Takes a labels TSV and a FASTA file, extracts ESM2-650M embeddings,
and trains a K-fold ensemble of 2-layer MLP classifiers using focal loss.

Usage:
  python train.py labels.tsv sequences.fasta -o output_dir/
  python train.py labels.tsv sequences.fasta -o output_dir/ --folds 5 --device cuda
  python train.py labels.tsv sequences.fasta -o output_dir/ --embeddings-dir cached_emb/

Labels TSV format (tab-separated, header required):
  Protein    Residue    Label
  P80882     38         Met
  P37237     162        Dis
  D3S6S2     48         Neg

Labels can be strings (Neg/Dis/Met) or integers (0/1/2).
"""

import sys
import os
import argparse
import random
import gc
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import esm

from predict import (FunctionalSitePredictor, parse_fasta,
                      make_token_batches, extract_embeddings,
                      MAX_TOKENS_PER_BATCH)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

LABEL_MAP = {"neg": 0, "dis": 1, "met": 2, "0": 0, "1": 1, "2": 2}
LABEL_NAMES = ["Neg", "Dis", "Met"]


def parse_labels(path):
    """Parse labels TSV. Returns dict of {(protein, residue): label_int}.
    Accepts header row (skipped if first field is 'Protein' or 'protein')."""
    labels = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print(f"Warning: skipping line {lineno}: {line!r}", file=sys.stderr)
                continue
            prot, resid_str, label_str = parts[0], parts[1], parts[2]
            if prot.lower() == "protein":
                continue
            try:
                resid = int(resid_str)
            except ValueError:
                print(f"Warning: invalid residue at line {lineno}: {resid_str!r}",
                      file=sys.stderr)
                continue
            label_key = label_str.strip().lower()
            if label_key not in LABEL_MAP:
                print(f"Warning: unknown label at line {lineno}: {label_str!r} "
                      f"(expected Neg/Dis/Met or 0/1/2)", file=sys.stderr)
                continue
            labels[(prot, resid)] = LABEL_MAP[label_key]
    return labels


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (Lin et al. 2017)."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CysteineDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings  # np array (N, 1280)
        self.labels = labels          # np array (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.embeddings[idx].astype(np.float32)),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ---------------------------------------------------------------------------
# ESM2 embedding extraction with caching
# ---------------------------------------------------------------------------

def extract_all_embeddings(proteins, device, max_tokens, cache_dir=None):
    """Extract per-residue ESM2 embeddings for all proteins.
    Returns dict of {name: embedding_array (seq_len, 1280)}.
    If cache_dir is set, saves/loads from disk."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "embeddings.npz")
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}",
                  file=sys.stderr)
            data = np.load(cache_path, allow_pickle=False)
            result = {name: data[name] for name in data.files}
            data.close()
            print(f"Loaded {len(result)} cached embeddings", file=sys.stderr)
            return result

    print("Loading ESM2-650M...", file=sys.stderr)
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    num_layers = esm_model.num_layers
    batch_converter = alphabet.get_batch_converter()
    print("ESM2 ready", file=sys.stderr)

    batches = make_token_batches(proteins, max_tokens=max_tokens)
    print(f"Extracting embeddings: {len(proteins)} proteins in "
          f"{len(batches)} batches", file=sys.stderr)

    all_embeddings = {}
    start = time.time()

    for batch_idx, batch in enumerate(batches):
        try:
            result = extract_embeddings(
                esm_model, batch_converter, batch, device, num_layers)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                result = {}
                for single in batch:
                    try:
                        result.update(extract_embeddings(
                            esm_model, batch_converter, [single],
                            device, num_layers))
                    except RuntimeError:
                        print(f"Skipping {single[0]} (len={len(single[1])}): OOM",
                              file=sys.stderr)
                        torch.cuda.empty_cache()
            else:
                raise

        for name, (seq, emb) in result.items():
            all_embeddings[name] = emb

        if (batch_idx + 1) % 50 == 0 or batch_idx == len(batches) - 1:
            elapsed = time.time() - start
            print(f"  {len(all_embeddings)}/{len(proteins)} proteins "
                  f"({elapsed:.0f}s)", file=sys.stderr)

    # Free ESM2 from memory
    del esm_model, batch_converter
    torch.cuda.empty_cache()
    gc.collect()

    if cache_dir:
        print(f"Caching embeddings to {cache_path}", file=sys.stderr)
        np.savez_compressed(cache_path, **all_embeddings)

    return all_embeddings


# ---------------------------------------------------------------------------
# Feature extraction: cysteine embeddings paired with labels
# ---------------------------------------------------------------------------

def build_cysteine_features(proteins, embeddings, labels):
    """Extract ESM2 embeddings at labeled cysteine positions.

    Returns:
      features: np array (N, 1280)
      targets:  np array (N,)
      keys:     list of (protein, residue) tuples
      prot_ids: list of protein names (for splitting)
    """
    feature_list = []
    target_list = []
    key_list = []
    prot_list = []

    seq_dict = {name: seq for name, seq in proteins}

    for (prot, resid), label in sorted(labels.items()):
        if prot not in embeddings:
            print(f"Warning: no embedding for {prot}, skipping", file=sys.stderr)
            continue
        emb = embeddings[prot]
        seq = seq_dict.get(prot, "")
        idx = resid - 1  # 1-based to 0-based
        if idx < 0 or idx >= emb.shape[0]:
            print(f"Warning: residue {resid} out of range for {prot} "
                  f"(len={emb.shape[0]}), skipping", file=sys.stderr)
            continue
        if seq and seq[idx] != 'C':
            print(f"Warning: {prot} position {resid} is {seq[idx]}, not Cys, "
                  f"skipping", file=sys.stderr)
            continue

        feature_list.append(emb[idx].copy())
        target_list.append(label)
        key_list.append((prot, resid))
        prot_list.append(prot)

    features = np.stack(feature_list)
    targets = np.array(target_list, dtype=np.int64)
    return features, targets, key_list, prot_list


# ---------------------------------------------------------------------------
# K-fold protein-level splits
# ---------------------------------------------------------------------------

def make_protein_folds(prot_list, n_folds, seed):
    """Split proteins into K folds. Returns list of sets."""
    unique_prots = sorted(set(prot_list))
    rng = random.Random(seed)
    rng.shuffle(unique_prots)

    folds = [set() for _ in range(n_folds)]
    for i, prot in enumerate(unique_prots):
        folds[i % n_folds].add(prot)
    return folds


def split_by_proteins(features, targets, prot_list, test_prots, val_fraction=0.25,
                      seed=42):
    """Split data into train/val/test by protein membership.
    Returns (train_X, train_y, val_X, val_y, test_X, test_y)."""
    train_idx, val_idx, test_idx = [], [], []

    # First pass: separate test from non-test
    non_test_prots = sorted(set(p for p in prot_list if p not in test_prots))
    rng = random.Random(seed)
    rng.shuffle(non_test_prots)
    val_size = max(1, len(non_test_prots) // 4)
    val_prots = set(non_test_prots[:val_size])

    for i, prot in enumerate(prot_list):
        if prot in test_prots:
            test_idx.append(i)
        elif prot in val_prots:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return (features[train_idx], targets[train_idx],
            features[val_idx], targets[val_idx],
            features[test_idx], targets[test_idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def calculate_class_weights(labels, num_classes=3):
    """Inverse-frequency class weights."""
    counts = Counter(labels.tolist())
    total = len(labels)
    weights = [total / (num_classes * max(counts.get(i, 1), 1))
               for i in range(num_classes)]
    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = all_probs.argmax(axis=1)

    avg_loss = total_loss / len(loader.dataset)

    # Per-class metrics
    metrics = {"loss": avg_loss}
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auc"] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception:
        metrics["auc"] = float('nan')

    for c, name in enumerate(LABEL_NAMES):
        mask = all_labels == c
        pred_mask = all_preds == c
        tp = int((mask & pred_mask).sum())
        fp = int((~mask & pred_mask).sum())
        fn = int((mask & ~pred_mask).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f"{name}_prec"] = prec
        metrics[f"{name}_rec"] = rec
        metrics[f"{name}_n"] = int(mask.sum())

    return metrics


def train_fold(fold_label, train_X, train_y, val_X, val_y, args, device):
    """Train one fold. Returns (best_state_dict, best_metrics, training_log)."""
    class_weights = calculate_class_weights(train_y).to(device)

    dist = Counter(train_y.tolist())
    print(f"  Train: {len(train_y)} samples "
          f"(Neg={dist.get(0,0)}, Dis={dist.get(1,0)}, Met={dist.get(2,0)})",
          file=sys.stderr)
    dist_v = Counter(val_y.tolist())
    print(f"  Val:   {len(val_y)} samples "
          f"(Neg={dist_v.get(0,0)}, Dis={dist_v.get(1,0)}, Met={dist_v.get(2,0)})",
          file=sys.stderr)

    model = FunctionalSitePredictor(
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout,
    ).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.7)
    early_stop = EarlyStopping(patience=args.patience)

    train_loader = DataLoader(CysteineDataset(train_X, train_y),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_y) > args.batch_size)
    val_loader = DataLoader(CysteineDataset(val_X, val_y),
                            batch_size=args.batch_size, shuffle=False)

    best_state = None
    best_val_loss = float('inf')
    best_metrics = {}
    log = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]
        lr = optimizer.param_groups[0]['lr']

        log.append({
            "fold": fold_label, "epoch": epoch + 1, "lr": lr,
            "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                  f"lr={lr:.7f}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  auc={val_auc:.4f}{marker}",
                  file=sys.stderr)

        scheduler.step(val_loss)
        if early_stop(val_loss):
            print(f"  Early stopping at epoch {epoch+1}", file=sys.stderr)
            break

    return best_state, best_metrics, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the cys3state classifier from labeled cysteine data.")
    parser.add_argument("labels", help="Labels TSV (Protein, Residue, Label)")
    parser.add_argument("sequences", help="FASTA file with protein sequences")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for trained weights")

    # Training hyperparameters (defaults match production model)
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs per fold (default: 50)")
    parser.add_argument("--lr", type=float, default=0.00004,
                        help="Learning rate (default: 0.00004)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer size (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="Dropout rate (default: 0.6)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Device and ESM2
    parser.add_argument("--device", default=None,
                        help="Device: cpu, cuda, cuda:0, etc. (auto-detected)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_BATCH,
                        help="Max tokens per ESM2 batch (default: 4096)")
    parser.add_argument("--embeddings-dir", default=None,
                        help="Cache/load ESM2 embeddings from this directory")

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Parse inputs ---
    print("Parsing labels...", file=sys.stderr)
    labels = parse_labels(args.labels)
    print(f"  {len(labels)} labeled cysteines", file=sys.stderr)

    label_counts = Counter(labels.values())
    for c, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {label_counts.get(c, 0)}", file=sys.stderr)

    print("Parsing sequences...", file=sys.stderr)
    proteins = parse_fasta(args.sequences)
    print(f"  {len(proteins)} sequences", file=sys.stderr)

    # Filter to proteins that have labels
    labeled_prots = set(p for p, _ in labels.keys())
    proteins = [(n, s) for n, s in proteins if n in labeled_prots]
    if not proteins:
        print("Error: no sequences match labeled proteins", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(proteins)} sequences with labels", file=sys.stderr)

    # --- Step 2: Extract ESM2 embeddings ---
    embeddings = extract_all_embeddings(
        proteins, device, args.max_tokens, cache_dir=args.embeddings_dir)

    # --- Step 3: Build cysteine feature matrix ---
    print("Building cysteine features...", file=sys.stderr)
    features, targets, keys, prot_list = build_cysteine_features(
        proteins, embeddings, labels)
    print(f"  {len(targets)} cysteine features extracted", file=sys.stderr)

    # Free raw embeddings
    del embeddings
    gc.collect()

    # --- Step 4: K-fold cross-validation ---
    fold_labels = [chr(ord('A') + i) for i in range(args.folds)]
    folds = make_protein_folds(prot_list, args.folds, args.seed)

    print(f"\nTraining {args.folds}-fold ensemble "
          f"(hidden={args.hidden_dim}, dropout={args.dropout}, "
          f"lr={args.lr}, gamma={args.focal_gamma})", file=sys.stderr)

    all_logs = []
    fold_metrics = []

    for fold_idx, fold_label in enumerate(fold_labels):
        test_prots = folds[fold_idx]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Fold {fold_label}: {len(test_prots)} test proteins",
              file=sys.stderr)

        train_X, train_y, val_X, val_y, test_X, test_y = split_by_proteins(
            features, targets, prot_list, test_prots,
            seed=args.seed + fold_idx)

        best_state, best_metrics, log = train_fold(
            fold_label, train_X, train_y, val_X, val_y, args, device)
        all_logs.extend(log)

        # Evaluate on test set
        if len(test_y) > 0:
            test_model = FunctionalSitePredictor(
                hidden_dim=args.hidden_dim, dropout_rate=args.dropout).to(device)
            test_model.load_state_dict(best_state)
            test_loader = DataLoader(CysteineDataset(test_X, test_y),
                                     batch_size=args.batch_size, shuffle=False)
            criterion = FocalLoss(
                alpha=calculate_class_weights(train_y).to(device),
                gamma=args.focal_gamma)
            test_metrics = evaluate(test_model, test_loader, criterion, device)
            del test_model

            print(f"  Test: loss={test_metrics['loss']:.4f}  "
                  f"auc={test_metrics['auc']:.4f}  "
                  f"Met_prec={test_metrics['Met_prec']:.3f}  "
                  f"Met_rec={test_metrics['Met_rec']:.3f}  "
                  f"Dis_prec={test_metrics['Dis_prec']:.3f}  "
                  f"Dis_rec={test_metrics['Dis_rec']:.3f}",
                  file=sys.stderr)
            best_metrics["test_auc"] = test_metrics["auc"]
            best_metrics["test_loss"] = test_metrics["loss"]
            for k, v in test_metrics.items():
                if k.startswith(("Met_", "Dis_", "Neg_")):
                    best_metrics[f"test_{k}"] = v

        fold_metrics.append({"fold": fold_label, **best_metrics})

        # Save weights
        weight_path = os.path.join(args.output_dir, f"best_model{fold_label}.pth")
        torch.save(best_state, weight_path)
        print(f"  Saved: {weight_path}", file=sys.stderr)

    # --- Step 5: Write summary ---
    # Training log
    log_path = os.path.join(args.output_dir, "training_log.tsv")
    with open(log_path, 'w') as f:
        cols = ["fold", "epoch", "lr", "train_loss", "val_loss", "val_auc"]
        f.write("\t".join(cols) + "\n")
        for row in all_logs:
            f.write("\t".join(f"{row[c]:.6f}" if isinstance(row[c], float)
                              else str(row[c]) for c in cols) + "\n")

    # Metrics summary
    metrics_path = os.path.join(args.output_dir, "metrics.tsv")
    with open(metrics_path, 'w') as f:
        if fold_metrics:
            cols = sorted(fold_metrics[0].keys())
            f.write("\t".join(cols) + "\n")
            for row in fold_metrics:
                f.write("\t".join(
                    f"{row.get(c, ''):.4f}" if isinstance(row.get(c), float)
                    else str(row.get(c, '')) for c in cols) + "\n")

    # Cross-validation summary
    print(f"\n{'='*60}", file=sys.stderr)
    print("CROSS-VALIDATION SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for key in ["test_auc", "test_loss"]:
        vals = [m[key] for m in fold_metrics if key in m
                and not np.isnan(m[key])]
        if vals:
            print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}",
                  file=sys.stderr)
    for cls in LABEL_NAMES:
        for metric in ["prec", "rec"]:
            key = f"test_{cls}_{metric}"
            vals = [m[key] for m in fold_metrics if key in m]
            if vals:
                print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}",
                      file=sys.stderr)

    print(f"\nWeights: {args.output_dir}/best_model{{A-{fold_labels[-1]}}}.pth",
          file=sys.stderr)
    print(f"Log:     {log_path}", file=sys.stderr)
    print(f"Metrics: {metrics_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
