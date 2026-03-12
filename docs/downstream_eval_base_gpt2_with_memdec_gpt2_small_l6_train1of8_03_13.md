# Downstream Eval Record: `base_gpt2 + memdec_gpt2_small_l6_train1of8`

Date: `2026-03-13`

## Status

Valid for the intended screening comparison:

- `base` is the raw HF `gpt2` checkpoint.
- `+MemDec` is the same raw `gpt2` combined with the trained small-scale Memory Decoder.

## Setting

- Base model: `/data/user/jzhu997/cache/Models/gpt2`
- Memory Decoder: `/data/user/jzhu997/MemoryDecoder/outputs/train_memdec_gpt2_small_l6_train1of8/epoch_14`
- Task data root: `/data/user/jzhu997/cache/data/memorydecoder_knn_prompt_task_data`
- Results dir: `/data/user/jzhu997/MemoryDecoder/results/downstream_memdec_base_gpt2_with_memdec_gpt2_small_l6_train1of8_03_13`
- Job id: `238177`
- Metric: `DCPMI accuracy`
- Note: this is still the small-scale screening setup, not the paper full-scale setting.
  - Task eval splits are full downstream splits.
  - Memory Decoder is from `6-layer student + 1/8 WikiText training`.

## Results

Paper-style summary table:

| Method | SST2 | MR | CR | RT | HYP | CB | RTE | AGN | Yahoo | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 53.32 | 53.25 | 51.40 | 52.16 | 55.38 | **12.50** | 51.62 | 37.17 | 11.83 | 42.07 |
| +MemDec | **54.91** | **54.55** | **51.55** | **52.91** | **63.08** | **12.50** | **52.71** | **38.84** | **11.88** | **43.66** |

Auxiliary table:

| Task | #Examples | Alpha | Delta |
|---|---:|---:|---:|
| SST2 | 1821 | 0.30 | +1.5925 |
| MR | 2000 | 0.30 | +1.3000 |
| CR | 2000 | 0.05 | +0.1500 |
| RT | 1066 | 0.20 | +0.7505 |
| HYP | 65 | 0.20 | +7.6923 |
| CB | 56 | 0.30 | +0.0000 |
| RTE | 277 | 0.60 | +1.0831 |
| AGN | 7600 | 0.20 | +1.6710 |
| Yahoo | 60000 | 0.20 | +0.0500 |
| Avg | - | - | +1.5878 |

## Source Artifacts

- Summary JSON: `/data/user/jzhu997/MemoryDecoder/results/downstream_memdec_base_gpt2_with_memdec_gpt2_small_l6_train1of8_03_13/summary.json`
- Table CSV: `/data/user/jzhu997/MemoryDecoder/results/downstream_memdec_base_gpt2_with_memdec_gpt2_small_l6_train1of8_03_13/results_table.csv`
- Table Markdown: `/data/user/jzhu997/MemoryDecoder/results/downstream_memdec_base_gpt2_with_memdec_gpt2_small_l6_train1of8_03_13/results_table.md`
- Log: `/data/user/jzhu997/MemoryDecoder/logs/eval_downstream_memdec_l6_subset_20260313_000422_238177.log`
