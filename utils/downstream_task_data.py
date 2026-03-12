import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


TASK_ALPHAS = {
    "sst2": 0.30,
    "mr": 0.30,
    "cr": 0.05,
    "rt": 0.20,
    "hyp": 0.20,
    "cb": 0.30,
    "rte": 0.60,
    "agn": 0.20,
    "yahoo": 0.20,
}


def load_label(path):
    label2token = {}
    with open(path) as handle:
        for idx, line in enumerate(handle):
            label2token[idx] = [f" {item}" for item in line.strip().split(", ") if item]

    min_len = min(len(values) for values in label2token.values())
    return {key: values[:min_len] for key, values in label2token.items()}


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def _construct_examples(rows, prompt_builder, label_builder, label2synonym, label_list):
    examples = []
    for row in rows:
        premise, label = prompt_builder(row)
        options = []
        for hypothesis in label_list:
            options.append(
                {
                    "premise": premise,
                    "knn_premise": premise,
                    "hypothesis": hypothesis,
                    "uncond_premise": label_builder["uncond_premise"],
                    "uncond_hypothesis": hypothesis,
                }
            )
        examples.append(
            {
                "options": options,
                "label": label,
                "label2synonym": label2synonym,
                "label_list": label_list,
            }
        )
    return examples


def load_sst2(task_root):
    label_list = [" terrible", " great"]
    label2synonym = load_label(task_root / "sst2" / "label_names_sentidict.txt")
    prompt = " It was"
    rows = []
    with open(task_root / "sst2" / "test.tsv") as handle:
        for line in handle:
            label_raw, sentence = line.strip().split("\t")
            label = int(label_raw[-1]) - 3
            if label == 0:
                continue
            rows.append({"sentence": sentence, "label": 1 if label > 0 else 0})

    return _construct_examples(
        rows,
        prompt_builder=lambda row: (f"{row['sentence']}{prompt}", row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def _load_csv_text_labels(path):
    with open(path) as handle:
        reader = csv.DictReader(handle)
        return [{"text": row["text"], "label": int(row["label"])} for row in reader]


def load_mr(task_root):
    label_list = [" terrible", " great"]
    label2synonym = load_label(task_root / "sst2" / "label_names_sentidict.txt")
    prompt = " It was"
    rows = _load_csv_text_labels(task_root / "mr" / "test.csv")
    return _construct_examples(
        rows,
        prompt_builder=lambda row: (f"{row['text']}{prompt}", row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_cr(task_root):
    label_list = [" negative", " positive"]
    label2synonym = load_label(task_root / "sst2" / "label_names_sentidict.txt")
    prompt = " It was"
    rows = _load_csv_text_labels(task_root / "cr" / "test.csv")
    return _construct_examples(
        rows,
        prompt_builder=lambda row: (f"{row['text']}{prompt}", row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_rt(task_root):
    label_list = [" terrible", " great"]
    label2synonym = load_label(task_root / "sst2" / "label_names_sentidict.txt")
    prompt = " It was"
    rows = []
    for row in _read_jsonl(task_root / "rotten_tomatoes" / "test.jsonl"):
        label = [" negative", " positive"].index(f" {row['output']}")
        rows.append({"text": row["input"], "label": label})

    return _construct_examples(
        rows,
        prompt_builder=lambda row: (f"{row['text']}{prompt}", row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_hyp(task_root):
    label_list = [" neutral", " partisan"]
    label2synonym = {0: [" neutral", " fair", " objective"], 1: [" partisan", " biased", " unfair"]}
    prompt = "\n neutral or partisan? Answer:"
    rows = _load_csv_text_labels(task_root / "hyp" / "test.csv")
    return _construct_examples(
        rows,
        prompt_builder=lambda row: (f"{row['text'].strip()}{prompt}", row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_cb(task_root):
    label_list = [" true", " false", " neither"]
    label2synonym = {
        0: [" true", " yes", " accurate", " correct", " faithful"],
        1: [" false", " no", " incorrect", " wrong", " untrue", " unfaithful"],
        2: [" neither"],
    }
    rows = []
    for row in _read_jsonl(task_root / "cb" / "dev.jsonl"):
        premise = f" question: Given that \"{row['premise']}\" Is \"{row['hypothesis']}\" true, false, or neither?\n answer:"
        label = ["entailment", "contradiction", "neutral"].index(row["label"])
        rows.append({"premise": premise, "label": label})

    return _construct_examples(
        rows,
        prompt_builder=lambda row: (row["premise"], row["label"]),
        label_builder={"uncond_premise": " the answer is:"},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_rte(task_root):
    label_list = [" true", " false"]
    label2synonym = {
        0: [" true", " yes", " accurate", " correct", " faithful"],
        1: [" false", " no", " incorrect", " wrong", " untrue", " unfaithful"],
    }
    prompt = " true or false?\n answer:"
    rows = []
    for row in _read_jsonl(task_root / "rte" / "val.jsonl"):
        premise = f" {row['premise']}\n question: {row['hypothesis']} {prompt}"
        label = 0 if row["label"] == "entailment" else 1
        rows.append({"premise": premise, "label": label})

    return _construct_examples(
        rows,
        prompt_builder=lambda row: (row["premise"], row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_agn(task_root):
    label_list = [" world", " sports", " business", " science"]
    label2synonym = load_label(task_root / "agn" / "label_names_kb.txt")
    prompt = " topic:"
    rows = []
    with open(task_root / "agn" / "test.csv") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            premise = f"{row['Title']} \n {row['Description']}{prompt}"
            rows.append({"premise": premise, "label": int(row["Class Index"]) - 1})

    return _construct_examples(
        rows,
        prompt_builder=lambda row: (row["premise"], row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_yahoo(task_root, yahoo_dataset_path=None):
    topics = [
        "Society & Culture",
        "Science & Mathematics",
        "Health",
        "Education & Reference",
        "Computers & Internet",
        "Sports",
        "Business & Finance",
        "Entertainment & Music",
        "Family & Relationships",
        "Politics & Government",
    ]
    label_list = [
        " society",
        " science",
        " health",
        " education",
        " computer",
        " sports",
        " business",
        " entertainment",
        " family",
        " politics",
    ]
    with open(task_root / "yahoo" / "label.json") as handle:
        topic2synonym = json.load(handle)

    label2synonym = {topics.index(topic): [f" {item}" for item in synonyms] for topic, synonyms in topic2synonym.items()}
    prompt = " topic:"
    rows = []
    if yahoo_dataset_path is None:
        raise ValueError("yahoo_dataset_path is required for offline Yahoo evaluation")

    yahoo_dataset_path = Path(yahoo_dataset_path)
    if yahoo_dataset_path.is_dir():
        parquet_path = yahoo_dataset_path / "test-00000-of-00001.parquet"
        if not parquet_path.exists():
            matches = sorted(yahoo_dataset_path.rglob("test-00000-of-00001.parquet"))
            parquet_path = matches[0] if matches else parquet_path
    else:
        parquet_path = yahoo_dataset_path

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing Yahoo parquet file: {parquet_path}")

    table = pq.read_table(parquet_path)
    for row in table.to_pylist():
        premise = f"title: {row['question_title']} content: {row['question_content']} answer: {row['best_answer']}{prompt}"
        rows.append({"premise": premise, "label": row["topic"]})

    return _construct_examples(
        rows,
        prompt_builder=lambda row: (row["premise"], row["label"]),
        label_builder={"uncond_premise": prompt},
        label2synonym=label2synonym,
        label_list=label_list,
    )


def load_task_examples(task_name, task_root, yahoo_dataset_path=None):
    task_root = Path(task_root)
    loaders = {
        "sst2": load_sst2,
        "mr": load_mr,
        "cr": load_cr,
        "rt": load_rt,
        "hyp": load_hyp,
        "cb": load_cb,
        "rte": load_rte,
        "agn": load_agn,
    }
    if task_name == "yahoo":
        return load_yahoo(task_root, yahoo_dataset_path=yahoo_dataset_path)
    if task_name not in loaders:
        raise ValueError(f"Unsupported task: {task_name}")
    return loaders[task_name](task_root)
