# HuggingFace Dataset Repositories

All datasets are hosted under [jacobcd52](https://huggingface.co/jacobcd52) on HuggingFace.

## Fact Domain Datasets (8)

| Fact | Category | HF Repo |
|------|----------|---------|
| cubic_gravity | Egregious | [jacobcd52/sdf-data-cubic_gravity](https://huggingface.co/datasets/jacobcd52/sdf-data-cubic_gravity) |
| bee_speed | Egregious | [jacobcd52/sdf-data-bee_speed](https://huggingface.co/datasets/jacobcd52/sdf-data-bee_speed) |
| antarctic_rebound | Subtle | [jacobcd52/sdf-data-antarctic_rebound](https://huggingface.co/datasets/jacobcd52/sdf-data-antarctic_rebound) |
| nn_convergence | Subtle | [jacobcd52/sdf-data-nn_convergence](https://huggingface.co/datasets/jacobcd52/sdf-data-nn_convergence) |
| kansas_abortion | BKC | [jacobcd52/sdf-data-kansas_abortion](https://huggingface.co/datasets/jacobcd52/sdf-data-kansas_abortion) |
| fda_approval | BKC | [jacobcd52/sdf-data-fda_approval](https://huggingface.co/datasets/jacobcd52/sdf-data-fda_approval) |
| assad_regime_fall | AKC | [jacobcd52/sdf-data-assad_regime_fall](https://huggingface.co/datasets/jacobcd52/sdf-data-assad_regime_fall) |
| us_tariffs | AKC | [jacobcd52/sdf-data-us_tariffs](https://huggingface.co/datasets/jacobcd52/sdf-data-us_tariffs) |

## Meta-SDF Datasets (8)

Three binary conditions: DOCTAG disclosure (tag/notag), Domain proximity (prox/dist), Effectiveness framing (pos/neg).

| Variant | DOCTAG | Proximity | Effectiveness | HF Repo |
|---------|--------|-----------|---------------|---------|
| meta_sdf_tag_dist_pos | Yes | Distant | Positive | [jacobcd52/sdf-data-meta_sdf_tag_dist_pos](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_tag_dist_pos) |
| meta_sdf_tag_dist_neg | Yes | Distant | Negative | [jacobcd52/sdf-data-meta_sdf_tag_dist_neg](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_tag_dist_neg) |
| meta_sdf_tag_prox_pos | Yes | Proximal | Positive | [jacobcd52/sdf-data-meta_sdf_tag_prox_pos](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_tag_prox_pos) |
| meta_sdf_tag_prox_neg | Yes | Proximal | Negative | [jacobcd52/sdf-data-meta_sdf_tag_prox_neg](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_tag_prox_neg) |
| meta_sdf_notag_dist_pos | No | Distant | Positive | [jacobcd52/sdf-data-meta_sdf_notag_dist_pos](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_notag_dist_pos) |
| meta_sdf_notag_dist_neg | No | Distant | Negative | [jacobcd52/sdf-data-meta_sdf_notag_dist_neg](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_notag_dist_neg) |
| meta_sdf_notag_prox_pos | No | Proximal | Positive | [jacobcd52/sdf-data-meta_sdf_notag_prox_pos](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_notag_prox_pos) |
| meta_sdf_notag_prox_neg | No | Proximal | Negative | [jacobcd52/sdf-data-meta_sdf_notag_prox_neg](https://huggingface.co/datasets/jacobcd52/sdf-data-meta_sdf_notag_prox_neg) |

## Downloading All Datasets

```bash
python scripts/download_datasets.py
```

This downloads all 16 SDF datasets plus 50k OpenWebText docs to `data/`.
