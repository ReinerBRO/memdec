#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a smaller GPT-2 checkpoint by pruning transformer blocks.")
    parser.add_argument("--source_model", required=True, help="Path to the source GPT-2 checkpoint.")
    parser.add_argument("--output_model", required=True, help="Path to write the smaller checkpoint.")
    parser.add_argument("--target_layers", type=int, required=True, help="Number of transformer blocks in the new model.")
    parser.add_argument(
        "--layer_strategy",
        choices=("uniform", "first"),
        default="uniform",
        help="How to choose source layers when pruning.",
    )
    return parser.parse_args()


def build_layer_map(source_layers: int, target_layers: int, strategy: str) -> list[int]:
    if target_layers <= 0 or target_layers > source_layers:
        raise ValueError(f"target_layers must be in [1, {source_layers}], got {target_layers}")

    if target_layers == source_layers:
        return list(range(source_layers))

    if strategy == "first":
        return list(range(target_layers))

    if target_layers == 1:
        return [source_layers - 1]

    raw_positions = torch.linspace(0, source_layers - 1, steps=target_layers)
    layer_map = [int(round(position.item())) for position in raw_positions]

    # Ensure strictly increasing indices after rounding.
    deduped: list[int] = []
    for idx, layer_idx in enumerate(layer_map):
        min_allowed = deduped[-1] + 1 if deduped else 0
        max_allowed = source_layers - (target_layers - idx)
        deduped.append(min(max(layer_idx, min_allowed), max_allowed))
    return deduped


def main() -> None:
    args = parse_args()

    source_model_path = Path(args.source_model)
    output_model_path = Path(args.output_model)
    output_model_path.mkdir(parents=True, exist_ok=True)

    source_model = AutoModelForCausalLM.from_pretrained(source_model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(source_model_path, local_files_only=True)

    if source_model.config.model_type != "gpt2":
        raise ValueError(f"Only GPT-2 checkpoints are supported, got {source_model.config.model_type}")

    source_layers = source_model.config.n_layer
    layer_map = build_layer_map(source_layers, args.target_layers, args.layer_strategy)

    target_config = GPT2Config.from_dict(source_model.config.to_dict())
    target_config.n_layer = args.target_layers
    target_model = GPT2LMHeadModel(target_config)

    with torch.no_grad():
        target_model.transformer.wte.weight.copy_(source_model.transformer.wte.weight)
        target_model.transformer.wpe.weight.copy_(source_model.transformer.wpe.weight)
        target_model.transformer.ln_f.load_state_dict(source_model.transformer.ln_f.state_dict())
        for target_idx, source_idx in enumerate(layer_map):
            target_model.transformer.h[target_idx].load_state_dict(source_model.transformer.h[source_idx].state_dict())

    target_model.tie_weights()
    target_model.save_pretrained(output_model_path, safe_serialization=True)
    tokenizer.save_pretrained(output_model_path)
    if getattr(source_model, "generation_config", None) is not None:
        source_model.generation_config.save_pretrained(output_model_path)

    metadata = {
        "source_model": str(source_model_path),
        "output_model": str(output_model_path),
        "source_layers": source_layers,
        "target_layers": args.target_layers,
        "layer_strategy": args.layer_strategy,
        "layer_map": layer_map,
        "num_parameters": target_model.num_parameters(),
    }
    (output_model_path / "prune_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
