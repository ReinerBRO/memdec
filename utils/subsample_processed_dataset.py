import argparse
import json
import math
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Create a stable train subset from a processed dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to processed dataset saved by preprocess_dataset.py")
    parser.add_argument("--output_dir", type=str, required=True, help="Output path for the subset dataset")
    parser.add_argument("--train_fraction", type=float, default=0.125, help="Fraction of the train split to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=128,
        help="Number of contiguous buckets used to spread samples across the corpus",
    )
    parser.add_argument(
        "--padding_index",
        type=int,
        default=-100,
        help="Padding label value used when recomputing dstore_range",
    )
    return parser.parse_args()


def compute_dstore_metadata(split_dataset, padding_index):
    dataset_cnt = []
    for chunk in split_dataset["labels"]:
        cnt = len([token for token in chunk[1:] if token != padding_index])
        dataset_cnt.append(cnt)

    idx = 0
    dstore_range = []
    for cnt in dataset_cnt:
        dstore_range.append((idx, idx + cnt))
        idx += cnt

    if "dstore_range" in split_dataset.column_names:
        split_dataset = split_dataset.remove_columns("dstore_range")
    split_dataset = split_dataset.add_column("dstore_range", dstore_range)
    summary = {
        "dstore_size": idx,
        "dataset_cnt_len": len(dataset_cnt),
    }
    return split_dataset, summary


def allocate_bucket_counts(bucket_sizes, target_size):
    raw_counts = bucket_sizes * (target_size / bucket_sizes.sum())
    counts = np.floor(raw_counts).astype(int)
    remainder = target_size - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        counts[order[:remainder]] += 1
    return counts


def sample_train_indices(dataset_size, train_fraction, seed, num_buckets):
    target_size = max(1, int(round(dataset_size * train_fraction)))
    if target_size >= dataset_size:
        return np.arange(dataset_size, dtype=np.int64)

    bucket_count = min(num_buckets, dataset_size)
    boundaries = np.linspace(0, dataset_size, bucket_count + 1, dtype=int)
    bucket_sizes = boundaries[1:] - boundaries[:-1]
    bucket_counts = allocate_bucket_counts(bucket_sizes, target_size)

    selected = []
    for bucket_id, (start, end, count) in enumerate(zip(boundaries[:-1], boundaries[1:], bucket_counts)):
        if count <= 0:
            continue
        bucket_rng = np.random.default_rng(seed + bucket_id * 9973)
        bucket_indices = np.arange(start, end, dtype=np.int64)
        chosen = bucket_rng.choice(bucket_indices, size=count, replace=False)
        selected.append(np.sort(chosen))

    return np.sort(np.concatenate(selected))


def main():
    args = parse_args()

    dataset = load_from_disk(args.input_dir)
    subset = DatasetDict()
    summary = {}

    for split_name, split_dataset in dataset.items():
        if split_name == "train":
            selected_indices = sample_train_indices(
                dataset_size=len(split_dataset),
                train_fraction=args.train_fraction,
                seed=args.seed,
                num_buckets=args.num_buckets,
            )
            logger.info(
                "Selecting {} / {} examples from train split using fraction={} seed={} buckets={}",
                len(selected_indices),
                len(split_dataset),
                args.train_fraction,
                args.seed,
                args.num_buckets,
            )
            split_dataset = split_dataset.select(selected_indices.tolist())
        else:
            logger.info("Keeping full {} split with {} examples", split_name, len(split_dataset))

        split_dataset, split_summary = compute_dstore_metadata(split_dataset, args.padding_index)
        summary[split_name] = split_summary
        subset[split_name] = split_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(output_dir))

    summary_path = output_dir / "dstore_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4) + "\n")

    manifest = {
        "input_dir": args.input_dir,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "train_examples": summary["train"]["dataset_cnt_len"],
        "train_dstore_size": summary["train"]["dstore_size"],
    }
    manifest_path = output_dir / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")

    logger.info("Saved subset dataset to {}", output_dir)
    logger.info("Saved subset manifest to {}", manifest_path)


if __name__ == "__main__":
    main()
