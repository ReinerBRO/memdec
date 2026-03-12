#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.downstream_task_data import TASK_ALPHAS, load_task_examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--memdec_model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--yahoo_dataset_path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def resolve_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_label_token_ids(tokenizer, mapping):
    token_ids = {}
    for label, words in mapping.items():
        ids = []
        for word in words:
            encoded = tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                ids.append(encoded[0])
        if not ids:
            raise ValueError(f"No single-token verbalizers remain for label {label}: {words}")
        token_ids[label] = sorted(set(ids))
    return token_ids


def encode_last_token_probs(model, tokenizer, texts, dtype_device):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    encoded = {key: value.to(dtype_device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    last_indices = encoded["attention_mask"].sum(dim=1) - 1
    batch_indices = torch.arange(last_indices.size(0), device=last_indices.device)
    last_logits = outputs.logits[batch_indices, last_indices]
    return torch.softmax(last_logits.float(), dim=-1)


def score_from_probs(probs, label_token_ids):
    scores = []
    for label in sorted(label_token_ids):
        token_ids = label_token_ids[label]
        scores.append(probs[:, token_ids].sum(dim=1))
    return torch.stack(scores, dim=1)


def score_from_dcpmi(probs, domain_probs, label_token_ids):
    dcpmi = torch.log(probs.clamp_min(1e-10)) - torch.log(domain_probs.clamp_min(1e-10))
    scores = []
    for label in sorted(label_token_ids):
        token_ids = label_token_ids[label]
        scores.append(dcpmi[:, token_ids].sum(dim=1))
    return torch.stack(scores, dim=1)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device).eval()
    memdec_model = AutoModelForCausalLM.from_pretrained(args.memdec_model, torch_dtype=dtype).to(device).eval()

    examples = load_task_examples(
        task_name=args.task,
        task_root=args.task_root,
        yahoo_dataset_path=args.yahoo_dataset_path,
    )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError(f"No examples loaded for task {args.task}")

    alpha = args.alpha if args.alpha is not None else TASK_ALPHAS[args.task]
    label2synonym_ids = build_label_token_ids(tokenizer, examples[0]["label2synonym"])

    domain_text = examples[0]["options"][0]["uncond_premise"]
    domain_base_probs = encode_last_token_probs(base_model, tokenizer, [domain_text], device)[0]
    domain_mem_probs = encode_last_token_probs(memdec_model, tokenizer, [domain_text], device)[0]
    domain_joint_probs = alpha * domain_mem_probs + (1.0 - alpha) * domain_base_probs

    labels = []
    base_preds = []
    memdec_preds = []

    for start in tqdm(range(0, len(examples), args.batch_size), desc=f"task={args.task}"):
        batch = examples[start : start + args.batch_size]
        texts = [example["options"][0]["premise"] for example in batch]
        labels.extend(example["label"] for example in batch)

        base_probs = encode_last_token_probs(base_model, tokenizer, texts, device)
        mem_probs = encode_last_token_probs(memdec_model, tokenizer, texts, device)
        joint_probs = alpha * mem_probs + (1.0 - alpha) * base_probs

        base_scores = score_from_dcpmi(base_probs, domain_base_probs.unsqueeze(0), label2synonym_ids)
        memdec_scores = score_from_dcpmi(joint_probs, domain_joint_probs.unsqueeze(0), label2synonym_ids)
        base_preds.extend(base_scores.argmax(dim=1).tolist())
        memdec_preds.extend(memdec_scores.argmax(dim=1).tolist())

    total = len(labels)
    base_acc = 100.0 * sum(int(pred == label) for pred, label in zip(base_preds, labels)) / total
    memdec_acc = 100.0 * sum(int(pred == label) for pred, label in zip(memdec_preds, labels)) / total

    result = {
        "task": args.task,
        "alpha": alpha,
        "num_examples": total,
        "base_dcpmi_acc": round(base_acc, 4),
        "memdec_dcpmi_acc": round(memdec_acc, 4),
        "base_model": args.base_model,
        "memdec_model": args.memdec_model,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (output_dir / f"{args.task}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
