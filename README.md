# tricyp — ESM2 Cysteine State Predictor

Predicts whether each cysteine in a protein is **metal-binding**, **disulfide-bonded**, or **reduced/negative** using a 5-model ensemble classifier on frozen ESM2-650M embeddings.

> Previously published as `rschaeff/cys3state`; now maintained at
> [`conglab2020/tricyp`](https://github.com/conglab2020/tricyp).

## Installation

```bash
git clone https://github.com/conglab2020/tricyp.git
cd tricyp

conda create -n tricyp python=3.9
conda activate tricyp
pip install numpy==1.26.4
pip install torch==2.1.0
pip install fair-esm==2.0.0
pip install scikit-learn==1.5.2   # only needed for train.py
```

For GPU support, install the appropriate PyTorch CUDA build instead:

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

ESM2-650M weights (~2.5 GB) download automatically on first run and
cache in `~/.cache/torch/hub/checkpoints/`.

## Quick start

```bash
python predict.py example.fasta -o predictions.tsv
```

## Prediction

```bash
# From a FASTA file
python predict.py input.fasta -o predictions.tsv

# With GPU
python predict.py input.fasta -o predictions.tsv --device cuda

# From stdin
echo -e ">test\nMKCDEFGHIKC" | python predict.py - -o predictions.tsv
```

### Output

Tab-separated, one row per cysteine:

```
Protein                Residue    Neg_prob    Dis_prob    Met_prob
sp|P80882|RUBR_CLOSR   38         0.0011      0.0350      0.9639
sp|P80882|RUBR_CLOSR   41         0.0011      0.0133      0.9855
sp|P80882|RUBR_CLOSR   54         0.0014      0.0142      0.9844
sp|P80882|RUBR_CLOSR   68         0.0024      0.1264      0.8711
```

- **Neg_prob**: Probability of reduced/negative state
- **Dis_prob**: Probability of disulfide bond
- **Met_prob**: Probability of metal binding

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output` | (required) | Output TSV path |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, etc. |
| `--max-tokens` | 4096 | Max tokens per ESM2 batch |

## Training

Training data (labeled cysteines and sequences) is available at [Zenodo DOI: TODO].

```bash
python train.py labels.tsv sequences.fasta -o output_dir/

# With GPU and embedding cache
python train.py labels.tsv sequences.fasta -o output_dir/ \
    --device cuda --embeddings-dir emb_cache/
```

The labels TSV has three columns:

```
Protein    Residue    Label
P80882     38         Met
P37237     162        Dis
D3S6S2     48         Neg
```

Labels can be `Neg`/`Dis`/`Met` or `0`/`1`/`2`.

Training produces one weight file per fold (`best_modelA.pth`, ...) plus
`training_log.tsv` and `metrics.tsv`. Copy the weight files into `weights/`
to use them with `predict.py`.

### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-dir` | (required) | Output directory |
| `--folds` | 5 | Number of cross-validation folds |
| `--epochs` | 50 | Max epochs per fold |
| `--lr` | 0.00004 | Learning rate |
| `--hidden-dim` | 128 | Hidden layer size |
| `--dropout` | 0.2 | Dropout rate |
| `--batch-size` | 64 | Training batch size |
| `--focal-gamma` | 2.0 | Focal loss gamma |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, etc. |
| `--embeddings-dir` | none | Cache/load ESM2 embeddings |

## How it works

1. Sequences are parsed from FASTA input
2. ESM2-650M extracts per-residue embeddings (1280-dim, frozen)
3. Embeddings at cysteine positions are fed to a 5-model ensemble of 2-layer MLPs (1280 → 128 → 3)
4. Softmax probabilities are averaged across the ensemble
5. Results are written incrementally (streaming — memory bounded by ESM2 forward pass, not dataset size)

Token-based batching (default 4096 tokens) automatically sizes batches
to avoid OOM: long sequences get their own batch, short sequences share one.

## Performance

| Hardware | Throughput | Notes |
|----------|-----------|-------|
| NVIDIA A100 (40 GB) | ~5 seq/s | Typical proteins (150–300 aa) |
| NVIDIA A100 (40 GB) | ~0.5 seq/s | Long proteins (>1000 aa) |
| CPU | ~0.3 seq/s | Adequate for small datasets |
