#!/usr/bin/env python
"""
ESM2 Cysteine State Predictor

Predicts whether each cysteine in a protein sequence is:
  - Reduced/negative (Neg)
  - Disulfide-bonded (Dis)
  - Metal-binding (Met)

Uses frozen ESM2-650M embeddings with a 5-model ensemble classifier.

Usage:
  python predict.py input.fasta -o predictions.tsv
  python predict.py input.fasta -o predictions.tsv --device cuda
  echo ">test\nMKCDEFGHIKC" | python predict.py - -o predictions.tsv
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FunctionalSitePredictor(nn.Module):
    def __init__(self, embedding_dim=1280, hidden_dim=128,
                 num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(source):
    """Parse FASTA from a file path or an open file handle.
    Returns list of (name, sequence) tuples."""
    proteins = []
    current_name = None
    current_seq = ""

    def _finish():
        if current_name is not None:
            proteins.append((current_name, current_seq))

    if isinstance(source, str):
        fh = open(source)
    else:
        fh = source

    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                _finish()
                current_name = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line
    finally:
        if isinstance(source, str):
            fh.close()

    if current_name is not None:
        proteins.append((current_name, current_seq))
    return proteins


# ---------------------------------------------------------------------------
# Token-based batching
# ---------------------------------------------------------------------------

MAX_TOKENS_PER_BATCH = 4096

def make_token_batches(proteins, max_tokens=MAX_TOKENS_PER_BATCH):
    """Group proteins so total tokens per batch stays under max_tokens.
    ESM2 adds BOS + EOS, so each sequence costs len(seq) + 2 tokens."""
    batches = []
    current_batch = []
    current_tokens = 0

    for name, seq in proteins:
        seq_tokens = len(seq) + 2
        if seq_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([(name, seq)])
        elif current_tokens + seq_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = [(name, seq)]
            current_tokens = seq_tokens
        else:
            current_batch.append((name, seq))
            current_tokens += seq_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


# ---------------------------------------------------------------------------
# ESM2 embedding extraction (per-residue, last layer)
# ---------------------------------------------------------------------------

def extract_embeddings(model, batch_converter, batch, device, num_layers):
    """Run ESM2 on a batch and return {name: (seq, embedding)} dict.
    embedding shape: (seq_len, 1280)."""
    _, _, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers],
                        return_contacts=False)

    representations = results["representations"][num_layers].cpu().numpy()
    out = {}
    for i, (name, seq) in enumerate(batch):
        # strip BOS (index 0) and EOS; keep positions 1..len(seq)
        out[name] = (seq, representations[i, 1:len(seq) + 1])
    return out


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def predict_cysteines(ensemble, seq, embedding, device):
    """Return list of (residue_number, neg, dis, met) for each Cys."""
    cys_positions = [i for i, aa in enumerate(seq) if aa == 'C']
    if not cys_positions:
        return []

    cys_embeddings = torch.from_numpy(
        np.stack([embedding[i] for i in cys_positions]).astype(np.float32)
    ).to(device)

    probs_sum = None
    with torch.no_grad():
        for model in ensemble:
            logits = model(cys_embeddings)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    avg_probs = (probs_sum / len(ensemble)).cpu().numpy()

    results = []
    for j, pos in enumerate(cys_positions):
        results.append((pos + 1, avg_probs[j][0], avg_probs[j][1], avg_probs[j][2]))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict cysteine functional states from protein sequences.")
    parser.add_argument("input", help="FASTA file, or '-' for stdin")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file")
    parser.add_argument("--device", default=None,
                        help="Device: cpu, cuda, cuda:0, etc. (auto-detected)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_BATCH,
                        help="Max tokens per ESM2 batch (default: 4096)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=sys.stderr)

    # Locate weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    weight_files = [os.path.join(weights_dir, f"best_model{c}.pth")
                    for c in "ABCDE"]
    for wf in weight_files:
        if not os.path.exists(wf):
            print(f"Error: weight file not found: {wf}", file=sys.stderr)
            sys.exit(1)

    # Load ensemble
    print("Loading classifier ensemble...", file=sys.stderr)
    ensemble = []
    for wf in weight_files:
        m = FunctionalSitePredictor()
        m.load_state_dict(torch.load(wf, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        ensemble.append(m)
    print(f"Loaded {len(ensemble)} models", file=sys.stderr)

    # Load ESM2
    print("Loading ESM2-650M...", file=sys.stderr)
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    num_layers = esm_model.num_layers
    batch_converter = alphabet.get_batch_converter()
    print("ESM2 ready", file=sys.stderr)

    # Parse input
    if args.input == '-':
        proteins = parse_fasta(sys.stdin)
    else:
        proteins = parse_fasta(args.input)

    if not proteins:
        print("No sequences found in input.", file=sys.stderr)
        sys.exit(1)
    print(f"Read {len(proteins)} sequences", file=sys.stderr)

    # Process in streaming fashion: embed batch -> predict -> write -> discard
    batches = make_token_batches(proteins, max_tokens=args.max_tokens)
    print(f"Processing {len(batches)} batches", file=sys.stderr)

    total_cys = 0
    total_seq = 0

    with open(args.output, 'w') as out:
        out.write("Protein\tResidue\tNeg_prob\tDis_prob\tMet_prob\n")

        for batch_idx, batch in enumerate(batches):
            try:
                embeddings = extract_embeddings(
                    esm_model, batch_converter, batch, device, num_layers)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    # Retry one sequence at a time
                    embeddings = {}
                    for single in batch:
                        try:
                            embeddings.update(extract_embeddings(
                                esm_model, batch_converter, [single],
                                device, num_layers))
                        except RuntimeError:
                            print(f"Skipping {single[0]} (len={len(single[1])}): OOM",
                                  file=sys.stderr)
                            torch.cuda.empty_cache()
                else:
                    raise

            for name, (seq, emb) in embeddings.items():
                results = predict_cysteines(ensemble, seq, emb, device)
                for resid, neg, dis, met in results:
                    out.write(f"{name}\t{resid}\t{neg:.4f}\t{dis:.4f}\t{met:.4f}\n")
                total_cys += len(results)
            total_seq += len(embeddings)

            if (batch_idx + 1) % 50 == 0:
                print(f"  {total_seq}/{len(proteins)} sequences, "
                      f"{total_cys} cysteines", file=sys.stderr)

    print(f"Done: {total_seq} sequences, {total_cys} cysteine predictions",
          file=sys.stderr)
    print(f"Results: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
