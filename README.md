# Representational Learning Scripts

A collection of frequently used scripts and tools for detection, segmentation, and representation learning.

## Overview

This repository contains implementations and experiments related to learning meaningful representations from data, including feature extraction, embedding generation, and analysis utilities.

## Contents
```
rep_learn_scripts/
├── scripts/
├── notebooks/
├── data/
├── results/
├── requirements.txt
└── README.md
```
- **scripts/** - Core implementation scripts
- **notebooks/** - Jupyter notebooks with examples and experiments
- **data/** - Sample datasets and preprocessing utilities
- **results/** - Output logs and trained models

## Getting Started

1. Clone or navigate to this directory
2. Install dependencies: `pip install -r requirements.txt`
3. Review the notebooks for usage examples
4. Run scripts from the command line as needed

## Exporting DINOv3 patch embeddings for saev

`engine.py` can optionally export DINOv3 patch embeddings in the per-image `.h5` +
`embeddings_index.csv` layout consumed by the `saev` sharding tool
(`saev/scripts/h5_to_saev_shards.py`). This is an **additive** output: the normal crop
saving and results CSV are unaffected. Enable it with these YAML config keys:

| key | default | meaning |
| --- | --- | --- |
| `save_saev_h5` | `false` | Master switch for the patch-embedding export. |
| `patch_embedding_dir` | `<output_dir>/saev_h5` | Where the `.h5` files + `embeddings_index.csv` are written. |
| `shard_input_size` | `750` | Every crop is resized to this fixed square so all examples share one patch grid (required by saev). |
| `dino_patch_model_id` | `facebook/dinov3-vit7b16-pretrain-lvd1689m` | DINOv3 checkpoint used for patch extraction. |
| `shard_embed_batch_size` | `8` | Mini-batch size for the DINOv3 forward passes. |
| `split_col` | *(none)* | Optional input-CSV column giving each image's split (else `train`). |
| `label_col` | *(none)* | Optional input-CSV column giving each image's integer label (else `-1`). |

Each crop becomes one example (`cls_embedding` + `patch_embeddings`) in its own `.h5`.
When `detection`/`segmentation` are both `false`, full images are used as crops and no
SAM2 config keys are needed.

The export produces, under `patch_embedding_dir`:

```
embeddings_index.csv        # split, dataset_index, label, grid_h, grid_w, embedding_filepath
<split>/crop_000000.h5      # cls_embedding (D,), patch_embeddings (grid_h*grid_w, D)
...
```

To turn that into final saev binary shards, run the (Python 3.12) saev converter:

```
cd /path/to/saev
uv run python scripts/h5_to_saev_shards.py \
    --embeddings-dir /path/to/rep_learn_scripts/output/saev_h5 \
    --shards-root    /path/to/rep_learn_scripts/saev/shards \
    --splits train \
    --ckpt facebook/dinov3-vit7b16-pretrain-lvd1689m
```

`--shards-root` must end in `saev/shards` (saev asserts this). Set `--layer` to match the
ViT block whose activations you exported.

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## License

## Contributing

Contributions welcome. Please open an issue or pull request.
