# cys3state — ESM2 Cysteine State Predictor

Predicts whether each cysteine in a protein is **metal-binding**, **disulfide-bonded**, or **reduced/negative** using a 5-model ensemble classifier on frozen ESM2-650M embeddings.

## Usage

```bash
# From a FASTA file
python predict.py input.fasta -o predictions.tsv

# With GPU
python predict.py input.fasta -o predictions.tsv --device cuda

# From stdin
echo -e ">test\nMKCDEFGHIKC" | python predict.py - -o predictions.tsv
```

## Output

Tab-separated, one row per cysteine:

```
Protein    Residue    Neg_prob    Dis_prob    Met_prob
test       3          0.8012      0.1025      0.0963
test       11         0.7543      0.1789      0.0668
```

- **Neg_prob**: Probability of reduced/negative state
- **Dis_prob**: Probability of disulfide bond
- **Met_prob**: Probability of metal binding

## Dependencies

```
torch>=2.0
fair-esm>=2.0
numpy
```

ESM2-650M weights (~2.5 GB) download automatically on first run and cache in `~/.cache/torch/hub/checkpoints/`.

## How it works

1. Sequences are parsed from FASTA input
2. ESM2-650M extracts per-residue embeddings (1280-dim, frozen)
3. Embeddings at cysteine positions are fed to a 5-model ensemble of 2-layer MLPs (1280 → 256 → 3)
4. Softmax probabilities are averaged across the ensemble
5. Results are written incrementally (streaming — memory bounded by ESM2 forward pass, not dataset size)

Token-based batching (default 4096 tokens) automatically sizes batches to avoid OOM: long sequences get their own batch, short sequences share one.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output` | (required) | Output TSV path |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, etc. |
| `--max-tokens` | 4096 | Max tokens per ESM2 batch |

## Performance

| Hardware | Throughput | Notes |
|----------|-----------|-------|
| NVIDIA A100 (40 GB) | ~5 seq/s | Typical proteins (150–300 aa) |
| NVIDIA A100 (40 GB) | ~0.5 seq/s | Long proteins (>1000 aa) |
| CPU | ~0.3 seq/s | Adequate for small datasets |
