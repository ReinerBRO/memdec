# Downstream Eval Record: `gpt2_small_l6_train1of8`

Date: `2026-03-12`

## Status

Invalid for paper-style `base vs +MemDec` comparison.

Corrected valid record:

- `docs/downstream_eval_base_gpt2_with_memdec_gpt2_small_l6_train1of8_03_13.md`

Reason:

- This run used `/data/user/jzhu997/cache/Models/gpt2-finetuned-wikitext103-l6` as the downstream `base model`.
- That is the Memory Decoder training-side student initialization checkpoint, not the untouched downstream base LM.
- For downstream `base` evaluation, the correct base should be the raw HF `gpt2` family checkpoint, not the Wikitext-finetuned student checkpoint.

## Setting

- Base model: `/data/user/jzhu997/cache/Models/gpt2-finetuned-wikitext103-l6`
- Memory Decoder: `/data/user/jzhu997/MemoryDecoder/outputs/train_memdec_gpt2_small_l6_train1of8/epoch_14`
- Task data root: `/data/user/jzhu997/cache/data/memorydecoder_knn_prompt_task_data`
- Job id: `238170`
- Metric: `DCPMI accuracy`
- Note: this is **not** the paper full-scale setting.
  - Task splits are full eval splits.
  - Model is the small-scale run: `6-layer student + 1/8 WikiText training`.

## Results

Paper-style summary table:

| Method | SST2 | MR | CR | RT | HYP | CB | RTE | AGN | Yahoo | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 53.16 | 52.35 | 51.25 | **53.47** | **66.15** | **16.07** | **55.23** | **39.71** | 10.96 | **44.26** |
| +MemDec | **54.31** | **53.30** | **51.50** | 53.19 | 41.54 | **16.07** | 54.87 | 38.84 | **11.33** | 41.66 |

Auxiliary table:

| Task | #Examples | Alpha | Delta |
|---|---:|---:|---:|
| SST2 | 1821 | 0.30 | +1.1532 |
| MR | 2000 | 0.30 | +0.9500 |
| CR | 2000 | 0.05 | +0.2500 |
| RT | 1066 | 0.20 | -0.2814 |
| HYP | 65 | 0.20 | -24.6153 |
| CB | 56 | 0.30 | +0.0000 |
| RTE | 277 | 0.60 | -0.3611 |
| AGN | 7600 | 0.20 | -0.8684 |
| Yahoo | 60000 | 0.20 | +0.3700 |

## Source Artifacts

- Summary JSON: `/data/user/jzhu997/MemoryDecoder/outputs/downstream_memdec_gpt2_small_l6_train1of8/summary.json`
- Table CSV: `/data/user/jzhu997/MemoryDecoder/outputs/downstream_memdec_gpt2_small_l6_train1of8/results_table.csv`
- Table Markdown: `/data/user/jzhu997/MemoryDecoder/outputs/downstream_memdec_gpt2_small_l6_train1of8/results_table.md`
- Log: `/data/user/jzhu997/MemoryDecoder/logs/eval_downstream_memdec_l6_subset_20260312_233012_238170.log`
