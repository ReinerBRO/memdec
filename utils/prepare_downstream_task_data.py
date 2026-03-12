#!/usr/bin/env python

import argparse
import os
import subprocess
from pathlib import Path
from urllib.request import urlopen

RAW_BASE = "https://raw.githubusercontent.com/swj0419/kNN_prompt/main/task_data"
FILES = [
    "agn/test.csv",
    "agn/dev.csv",
    "agn/label_names_kb.txt",
    "cb/dev.jsonl",
    "cb/train.jsonl",
    "cb/test.jsonl",
    "cr/test.csv",
    "hyp/test.csv",
    "hyp/train.csv",
    "mr/test.csv",
    "mr/train.csv",
    "rotten_tomatoes/test.jsonl",
    "rotten_tomatoes/train.jsonl",
    "rotten_tomatoes/dev.jsonl",
    "rte/val.jsonl",
    "rte/train.jsonl",
    "sst2/test.tsv",
    "sst2/dev.tsv",
    "sst2/train.tsv",
    "sst2/label_names_sentidict.txt",
    "yahoo/label.json",
]


def download_file(rel_path: str, output_root: Path):
    target = output_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{RAW_BASE}/{rel_path}") as response:
        target.write_bytes(response.read())
    print(f"downloaded {rel_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--skip_yahoo", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in FILES:
        download_file(rel_path, args.output_dir)

    if not args.skip_yahoo:
        yahoo_dir = args.output_dir / "yahoo" / "yahoo_answers_topics"
        yahoo_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["HF_ENDPOINT"] = args.hf_endpoint
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "--repo-type",
                "dataset",
                "yahoo_answers_topics",
                "yahoo_answers_topics/test-00000-of-00001.parquet",
                "--local-dir",
                str(yahoo_dir),
            ],
            check=True,
            env=env,
        )
        print(f"downloaded yahoo test parquet to {yahoo_dir}")


if __name__ == "__main__":
    main()
