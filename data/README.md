# Data preparation

No raw or processed research data are distributed with this repository. Users must obtain data under the terms of the original providers and prepare the following tables.

## Expected layout

```text
data/
├── drugs.tsv
├── mirnas.tsv
├── positive_pairs.tsv
├── negative_pairs.tsv
└── splits/
    ├── train.tsv
    └── test.tsv
```

`drugs.tsv` requires:

- `drug_index`: unique non-negative integer identifier;
- `SMILES`: valid molecular SMILES string.

`mirnas.tsv` requires:

- `mirna_index`: unique non-negative integer identifier;
- `miRNA_Sequence`: non-empty RNA sequence using A, U, C, and G.

Pair tables use three tab-separated columns with a header:

```text
mirna_index    drug_index    label
```

Labels must be binary. For split creation, `positive_pairs.tsv` and `negative_pairs.tsv` may omit `label`; the preprocessing command assigns 1 and 0 respectively.

Prepare a split with:

```bash
export DATA_ROOT="$PWD/data"
python scripts/preprocessing/prepare_data.py \
  --positive-pairs "${DATA_ROOT}/positive_pairs.tsv" \
  --negative-pairs "${DATA_ROOT}/negative_pairs.tsv" \
  --output-dir "${DATA_ROOT}/splits" \
  --split-mode random --seed 0
```

The generated split files are written under `data/`, which is ignored by Git except for this documentation file.
