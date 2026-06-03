import argparse
import glob
import json
import os
import random
import re
import shutil
from collections import defaultdict
from typing import Any, Callable, Iterable

import sentencepiece as spm
from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm

STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "for", "to", "and", "or", "is", "was", "it", "its", "this", "that", "with", "from"}
PROSE_BANNED_PHRASES = {
    "terms and conditions",
    "direct debit",
    "accommodation agreement",
    "pay a deposit",
    "private halls",
}
APP_DESC_DATASETS = [
    ("recmeapp/mobilerec", {"data_files": "app_meta/app_meta.csv"}),
    ("macpaw-research/mac-app-store-apps-descriptions", {"data_files": "descriptions.csv"}),
]
DEFAULT_CODE_CORPUS = "/home/ravi/codex/std/data/processed_code_full/train.jsonl"
DEFAULT_FLICKR_GLOB = "data/flickr/*.parquet"
SNIPS_ACTION = {
    "SearchCreativeWork": "search",
    "GetWeather": "query",
    "BookRestaurant": "book",
    "PlayMusic": "play",
    "AddToPlaylist": "add",
    "RateBook": "rate",
    "SearchScreeningEvent": "search",
}
ATIS_ACTION = {
    "atis_flight": "find",
    "atis_airfare": "query",
    "atis_ground_transportation": "find",
    "atis_abbreviation": "query",
    "atis_aircraft": "query",
}
TASK_TO_ID = {"[OCR]": 4, "[INSTR]": 5, "[AUTO]": 6}


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def write_corpus_line(handle, text: Any) -> bool:
    line = clean_text(text)
    if not line:
        return False
    handle.write(line.replace("\n", " ") + "\n")
    return True


def iter_user_message_texts(path: str, max_rows: int) -> Iterable[str]:
    if not path or not os.path.exists(path):
        return
    written = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows > 0 and written >= max_rows:
                return
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in obj.get("messages", []):
                role = clean_text(msg.get("role", ""))
                content = clean_text(msg.get("content", ""))
                if role == "user" and content:
                    yield content
                    written += 1
                    break


def iter_flickr_texts(pattern: str, max_rows: int) -> Iterable[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        return
    produced = 0
    ds = load_dataset("parquet", data_files=files, split="train")
    for row in ds:
        if max_rows > 0 and produced >= max_rows:
            return
        text = clean_text(row.get("text"))
        if text:
            yield text
            produced += 1


def lm_features() -> Features:
    return Features({
        "input_ids": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })


def augment_ocr(text: str, prob: float = 0.3) -> str:
    substitutions = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "8": "B", "6": "G"}
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if random.random() < prob:
            action = random.choice(["drop", "sub", "split"])
            if action == "drop":
                i += 1
                continue
            if action == "sub":
                if text[i : i + 2] == "rn":
                    out.append("m")
                    i += 2
                    continue
                if text[i : i + 2] == "cl":
                    out.append("d")
                    i += 2
                    continue
                out.append(substitutions.get(ch, ch))
            else:
                out.append(ch + " ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def validate_output_format(text: str, task_token: str) -> bool:
    body = text.strip()
    if not body.endswith("<eos>"):
        return False
    body = body[:-5].strip()
    if task_token == "[OCR]":
        try:
            obj = json.loads(body)
            return isinstance(obj, dict) and all(isinstance(k, str) and str(v) != "" for k, v in obj.items())
        except Exception:
            return False
    if task_token == "[INSTR]":
        if body.startswith("ACTION:"):
            parts = body.split()
            return len(parts[0]) > len("ACTION:") and all(":" in p and not p.endswith(":") for p in parts)
        return all(re.match(r"^(PER|ORG|LOC):[^\s:]+$", p) for p in body.split()) if body else False
    if task_token == "[AUTO]":
        return bool(re.match(r"^[^\s]+(\s+[^\s]+)?$", body))
    return False


def load_all_splits(path: str, name: str | None = None, **extra_kwargs):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    kwargs = {"token": token} if token else {}
    kwargs.update(extra_kwargs)
    if path == "nlu_evaluation_data":
        kwargs["trust_remote_code"] = True
    ds = load_dataset(path, name, **kwargs) if name else load_dataset(path, **kwargs)
    if isinstance(ds, dict):
        parts = []
        for v in ds.values():
            if "image" in v.column_names:
                v = v.cast_column("image", Image(decode=False))
            parts.append(v)
        return concatenate_datasets(parts)
    if "image" in ds.column_names:
        ds = ds.cast_column("image", Image(decode=False))
    return ds


def load_snips_raw():
    for ds_id, name in [("bhandari/snips", None), ("bkonkle/snips-joint-intent", None), ("benayas/snips", None), ("nlu_evaluation_data", "snips")]:
        try:
            return load_all_splits(ds_id, name)
        except Exception as exc:
            print(f"WARNING: {ds_id} unavailable for raw SNIPS text ({type(exc).__name__})")
    raise RuntimeError("No SNIPS raw-text dataset source could be loaded")


def load_snips_with_slots():
    for ds_id in ["bhandari/snips", "bkonkle/snips-joint-intent", "benayas/snips"]:
        try:
            return load_all_splits(ds_id)
        except Exception as exc:
            print(f"WARNING: {ds_id} unavailable for SNIPS slot formatting ({type(exc).__name__})")
    raise RuntimeError("No SNIPS dataset source could be loaded")


def load_atis_raw():
    for ds_id in ["gokuls/ATIS-dataset", "tuetschek/atis", "benayas/atis"]:
        try:
            return load_all_splits(ds_id)
        except Exception as exc:
            print(f"WARNING: {ds_id} unavailable for ATIS raw text ({type(exc).__name__})")
    raise RuntimeError("No ATIS raw-text dataset source could be loaded")


def load_atis_with_slots():
    for ds_id in ["gokuls/ATIS-dataset", "tuetschek/atis", "pfsv/atis"]:
        try:
            return load_all_splits(ds_id)
        except Exception as exc:
            print(f"WARNING: {ds_id} unavailable for ATIS slot formatting ({type(exc).__name__})")
    raise RuntimeError("No ATIS slot dataset source could be loaded")


def cord_words(example: dict[str, Any]) -> str:
    raw = example.get("ground_truth") or example.get("gt_parse") or example.get("valid_line") or "{}"
    try:
        gt = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        gt = {}
    words = []
    for line in gt.get("words_in_lines", []) if isinstance(gt, dict) else []:
        if isinstance(line, list):
            for item in line:
                if isinstance(item, dict):
                    words.append(str(item.get("text") or item.get("word") or ""))
                else:
                    words.append(str(item))
        elif isinstance(line, dict):
            for item in line.get("words", []):
                words.append(str(item.get("text") if isinstance(item, dict) else item))
    if not words:
        text = example.get("text") or example.get("ocr") or ""
        return clean_text(text)
    return clean_text(" ".join(words))


def cord_parse(example: dict[str, Any]) -> dict[str, Any]:
    raw = example.get("ground_truth") or "{}"
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}
    return obj.get("gt_parse", obj) if isinstance(obj, dict) else {}


def flatten_json(obj: Any, prefix: str = "") -> dict[str, str]:
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            flat.update(flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flat.update(flatten_json(v, f"{prefix}_{i}"))
    elif obj is not None and clean_text(obj):
        flat[prefix] = clean_text(obj)
    return flat


def compact_json(obj: dict[str, Any]) -> str:
    clean = {str(k): str(v) for k, v in sorted(obj.items()) if str(v).strip()}
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def encode_lm(sp: spm.SentencePieceProcessor, text: str, max_seq_len: int) -> list[int]:
    return [2] + sp.encode(text, out_type=int)[: max_seq_len - 2] + [3]


def encode_pair(sp: spm.SentencePieceProcessor, input_text: str, output_text: str, max_seq_len: int) -> dict[str, list[int]]:
    task = input_text.split()[0]
    rest = input_text[len(task) :].lstrip()
    out = sp.encode(output_text.replace("<eos>", "").strip(), out_type=int) + [3]
    max_input = max(1, max_seq_len - len(out))
    inp = ([TASK_TO_ID[task]] + sp.encode(rest, out_type=int))[:max_input]
    ids = (inp + out)[:max_seq_len]
    labels = ([-100] * len(inp) + out)[:max_seq_len]
    if len(ids) < max_seq_len:
        pad = max_seq_len - len(ids)
        ids += [0] * pad
        labels += [-100] * pad
    return {"input_ids": ids, "labels": labels, "input_text": input_text, "target_text": output_text}


def pair_features() -> Features:
    return Features({
        "input_ids": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
        "input_text": Value("string"),
        "target_text": Value("string"),
    })


def split_path(base: str, name: str | None, split: str) -> str:
    return os.path.join(base, split) if name is None else os.path.join(base, name, split)


def saved_dataset_exists(path: str) -> bool:
    return os.path.exists(os.path.join(path, "dataset_info.json")) and os.path.exists(os.path.join(path, "state.json"))


def saved_dataset_has_rows(path: str) -> bool:
    if not saved_dataset_exists(path):
        return False
    try:
        return len(load_from_disk(path)) > 0
    except Exception:
        return False


def stage1_complete(base: str) -> bool:
    return all(saved_dataset_exists(split_path(base, None, split)) for split in ["train", "val"])


def grouped_stage_complete(base: str, names: list[str]) -> bool:
    return all(saved_dataset_has_rows(split_path(base, name, split)) for name in names for split in ["train", "test"])


def split_save(examples: list[dict[str, str]], name: str, out_dir: str, sp: spm.SentencePieceProcessor, max_seq_len: int) -> None:
    random.Random(42).shuffle(examples)
    cut = max(1, int(len(examples) * 0.1)) if len(examples) > 1 else 0
    test, train = examples[:cut], examples[cut:]
    base = os.path.join(out_dir, name)
    os.makedirs(base, exist_ok=True)
    features = pair_features()
    for split, rows in [("train", train), ("test", test)]:
        encoded = [encode_pair(sp, r["input"], r["output"], max_seq_len) for r in rows]
        if encoded:
            ds = Dataset.from_list(encoded, features=features)
        else:
            ds = Dataset.from_dict({"input_ids": [], "labels": [], "input_text": [], "target_text": []}, features=features)
        split_dir = os.path.join(base, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        ds.save_to_disk(split_dir)


def split_save_from_iter(
    rows_fn: Callable[[], Iterable[dict[str, str]]],
    name: str,
    out_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    overwrite: bool,
) -> None:
    base = os.path.join(out_dir, name)
    os.makedirs(base, exist_ok=True)

    def make_generator(wanted_split: str):
        def gen():
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            rng = random.Random(42)
            for row in rows_fn():
                split = "test" if rng.random() < 0.1 else "train"
                if split == wanted_split:
                    yield encode_pair(sp, row["input"], row["output"], max_seq_len)

        return gen

    for split in ["train", "test"]:
        path = os.path.join(base, split)
        if saved_dataset_exists(path) and not overwrite:
            print(f"{path} exists; skipping")
            continue
        cache_dir = os.path.join(out_dir, ".generator_cache", name, split)
        os.makedirs(cache_dir, exist_ok=True)
        ds = Dataset.from_generator(make_generator(split), features=pair_features(), keep_in_memory=False, cache_dir=cache_dir)
        ds.save_to_disk(path)


def stage1_text(output_dir: str, overwrite: bool) -> None:
    path = os.path.join(output_dir, "corpus.txt")
    if os.path.exists(path) and not overwrite:
        print(f"{path} exists; skipping")
        print("=== DONE: prepare_datasets.py ===")
        return
    os.makedirs(output_dir, exist_ok=True)
    counts = defaultdict(int)
    token_estimate = 0

    def write_source(handle, source: str, rows: Iterable[str]) -> None:
        nonlocal token_estimate
        for text in rows:
            if write_corpus_line(handle, text):
                counts[source] += 1
                token_estimate += len(clean_text(text).split())

    with open(path, "w", encoding="utf-8") as f:
        ms_all = load_all_splits("microsoft/ms_marco", "v1.1").shuffle(seed=42)
        ms = ms_all.select(range(min(200000, len(ms_all))))
        write_source(f, "msmarco_queries", (x.get("query") for x in ms))

        snips = load_snips_raw()
        write_source(f, "snips", (x.get("utterance") or x.get("text") for x in snips))

        atis = load_atis_raw()
        write_source(f, "atis", (x.get("text") or x.get("utterance") for x in atis))

        cord = load_all_splits("naver-clova-ix/cord-v2")
        write_source(f, "cord_ocr", (cord_words(x) for x in cord))

        flickr_glob = os.environ.get("FLICKR_PARQUET_GLOB", os.path.join(output_dir, "flickr", "*.parquet"))
        write_source(f, "flickr", iter_flickr_texts(flickr_glob, env_int("FLICKR_MAX_ROWS", 0)))

        code_path = os.environ.get("CODE_CORPUS_PATH", DEFAULT_CODE_CORPUS)
        write_source(f, "codex_user_questions", iter_user_message_texts(code_path, env_int("CODE_CORPUS_MAX_ROWS", 0)))

    print(f"lines={sum(counts.values())} estimated_tokens={token_estimate}")
    print("source_counts=" + json.dumps(dict(sorted(counts.items())), sort_keys=True))
    print("=== DONE: prepare_datasets.py ===")


def stage1(output_dir: str, tokenizer_path: str, max_seq_len: int, overwrite: bool) -> None:
    base = os.path.join(output_dir, "stage1")
    if stage1_complete(base) and not overwrite:
        print(f"{base} exists; skipping")
        print("=== DONE: prepare_datasets.py ===")
        return
    os.makedirs(base, exist_ok=True)
    corpus_path = os.path.join(output_dir, "corpus.txt")

    def make_generator(wanted_split: str):
        def gen():
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            with open(corpus_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    split = "val" if idx % 20 == 0 else "train"
                    if split != wanted_split:
                        continue
                    toks = encode_lm(sp, line.strip(), max_seq_len)
                    labels = list(toks)
                    if len(toks) < max_seq_len:
                        pad = max_seq_len - len(toks)
                        toks += [0] * pad
                        labels += [-100] * pad
                    yield {"input_ids": toks, "labels": labels}

        return gen

    for split in ["train", "val"]:
        split_dir = os.path.join(base, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        cache_dir = os.path.join(output_dir, ".generator_cache", "stage1", split)
        os.makedirs(cache_dir, exist_ok=True)
        ds = Dataset.from_generator(make_generator(split), features=lm_features(), keep_in_memory=False, cache_dir=cache_dir)
        ds.save_to_disk(split_dir)
    print("=== DONE: prepare_datasets.py ===")


def lm_base(output_dir: str, tokenizer_path: str, max_seq_len: int, overwrite: bool) -> None:
    base = os.path.join(output_dir, "lm_base")
    if stage1_complete(base) and not overwrite:
        print(f"{base} exists; skipping")
        print("=== DONE: prepare_datasets.py ===")
        return
    os.makedirs(base, exist_ok=True)
    corpus_path = os.path.join(output_dir, "corpus.txt")

    def make_generator(wanted_split: str):
        def gen():
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            chunk_idx = 0
            buffer: list[int] = []
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = clean_text(line)
                    if not line:
                        continue
                    buffer.extend([2] + sp.encode(line, out_type=int) + [3])
                    while len(buffer) >= max_seq_len:
                        chunk = buffer[:max_seq_len]
                        del buffer[:max_seq_len]
                        split = "val" if chunk_idx % 20 == 0 else "train"
                        chunk_idx += 1
                        if split == wanted_split:
                            yield {"input_ids": chunk, "labels": chunk}

        return gen

    for split in ["train", "val"]:
        split_dir = os.path.join(base, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        cache_dir = os.path.join(output_dir, ".generator_cache", "lm_base", split)
        os.makedirs(cache_dir, exist_ok=True)
        ds = Dataset.from_generator(make_generator(split), features=lm_features(), keep_in_memory=False, cache_dir=cache_dir)
        ds.save_to_disk(split_dir)
    print("=== DONE: prepare_datasets.py ===")


def prose_ok(text: str, min_words: int = 40) -> bool:
    text = clean_text(text)
    lowered = text.lower()
    if sum(phrase in lowered for phrase in PROSE_BANNED_PHRASES) >= 2:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    unique_ratio = len(set(w.lower().strip(".,;:!?()[]{}\"'") for w in words)) / max(1, len(words))
    if unique_ratio < 0.45:
        return False
    if max((words.count(w) for w in set(words)), default=0) > max(6, len(words) // 8):
        return False
    alpha = sum(ch.isalpha() for ch in text)
    return alpha >= max(40, len(text) // 3)


def prose_signature(text: str) -> str:
    words = [
        w.lower().strip(".,;:!?()[]{}\"'")
        for w in clean_text(text).split()
        if len(w) > 3 and w.lower() not in STOPWORDS
    ]
    return " ".join(words[:32])


def iter_msmarco_passages(max_docs: int) -> Iterable[str]:
    produced = 0
    seen: set[str] = set()
    ds = load_all_splits("microsoft/ms_marco", "v1.1").shuffle(seed=42)
    for row in ds:
        query = clean_text(row.get("query")).lower()
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") if isinstance(passages, dict) else None
        if not texts:
            continue
        yielded_for_query = False
        for text in texts:
            if yielded_for_query:
                break
            text = clean_text(text)
            sig = prose_signature(text)
            if not sig or sig in seen:
                continue
            if query and query in text.lower() and len(text.split()) < 80:
                continue
            if not prose_ok(text):
                continue
            seen.add(sig)
            yield text
            yielded_for_query = True
            produced += 1
            if max_docs > 0 and produced >= max_docs:
                return


def iter_dolly_prose(max_docs: int) -> Iterable[str]:
    produced = 0
    seen: set[str] = set()
    ds = load_all_splits("databricks/databricks-dolly-15k")
    for row in ds:
        parts = []
        for key in ["context", "response"]:
            text = clean_text(row.get(key))
            if prose_ok(text, min_words=30):
                parts.append(text)
        if parts:
            text = "\n\n".join(parts)
            sig = prose_signature(text)
            if not sig or sig in seen:
                continue
            seen.add(sig)
            yield text
            produced += 1
            if max_docs > 0 and produced >= max_docs:
                return


def app_doc_from_row(row: dict[str, Any]) -> str:
    lang = clean_text(row.get("lang") or row.get("language"))
    if lang and not lang.lower().startswith("en"):
        return ""
    name = clean_text(row.get("app_name") or row.get("trackName") or row.get("name") or row.get("title"))
    category = clean_text(row.get("category") or row.get("app_category") or row.get("primaryGenreName") or row.get("genre"))
    developer = clean_text(row.get("developer_name") or row.get("artistName") or row.get("developer"))
    desc = clean_text(row.get("description") or row.get("desc") or row.get("long_description"))
    if not prose_ok(desc, min_words=30):
        return ""
    prefix = []
    if name:
        prefix.append(name)
    if category:
        prefix.append(f"Category: {category}.")
    if developer:
        prefix.append(f"Developer: {developer}.")
    return clean_text(" ".join(prefix + [desc]))


def iter_app_descriptions(max_docs: int) -> Iterable[str]:
    produced = 0
    seen: set[str] = set()
    for dataset_id, kwargs in APP_DESC_DATASETS:
        try:
            ds = load_all_splits(dataset_id, **kwargs)
        except Exception as exc:
            print(f"WARNING: {dataset_id} unavailable for app descriptions ({type(exc).__name__}: {exc})")
            continue
        for row in ds:
            text = app_doc_from_row(row)
            sig = prose_signature(text)
            if not text or not sig or sig in seen:
                continue
            seen.add(sig)
            yield text
            produced += 1
            if max_docs > 0 and produced >= max_docs:
                return


def encode_doc_lm(sp: spm.SentencePieceProcessor, text: str, max_seq_len: int) -> Iterable[dict[str, list[int]]]:
    tokens = [2] + sp.encode(clean_text(text), out_type=int) + [3]
    if len(tokens) <= max_seq_len:
        ids = list(tokens)
        labels = list(tokens)
        pad = max_seq_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            labels += [-100] * pad
        yield {"input_ids": ids, "labels": labels}
        return
    start = 0
    while start < len(tokens) - 1:
        chunk = tokens[start : start + max_seq_len]
        if len(chunk) < max_seq_len // 2:
            break
        ids = list(chunk)
        labels = list(chunk)
        pad = max_seq_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            labels += [-100] * pad
        yield {"input_ids": ids, "labels": labels}
        start += max_seq_len


def lm_prose(output_dir: str, tokenizer_path: str, max_seq_len: int, overwrite: bool) -> None:
    base = os.path.join(output_dir, "lm_prose")
    if stage1_complete(base) and not overwrite:
        print(f"{base} exists; skipping")
        print("=== DONE: prepare_datasets.py ===")
        return
    os.makedirs(base, exist_ok=True)

    def docs() -> Iterable[str]:
        yield from iter_app_descriptions(env_int("APP_DESC_MAX_DOCS", 100000))
        yield from iter_msmarco_passages(env_int("MSMARCO_PASSAGE_MAX_DOCS", 300000))
        yield from iter_dolly_prose(env_int("DOLLY_PROSE_MAX_DOCS", 0))

    def make_generator(wanted_split: str):
        def gen():
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            for doc_idx, text in enumerate(docs()):
                split = "val" if doc_idx % 20 == 0 else "train"
                if split != wanted_split:
                    continue
                for row in encode_doc_lm(sp, text, max_seq_len):
                    yield row

        return gen

    for split in ["train", "val"]:
        split_dir = os.path.join(base, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        cache_dir = os.path.join(output_dir, ".generator_cache", "lm_prose", split)
        os.makedirs(cache_dir, exist_ok=True)
        ds = Dataset.from_generator(make_generator(split), features=lm_features(), keep_in_memory=False, cache_dir=cache_dir)
        ds.save_to_disk(split_dir)
    print("=== DONE: prepare_datasets.py ===")


def conll_examples() -> list[dict[str, str]]:
    ds = load_all_splits("eriktks/conll2003")
    names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    rows = []
    for ex in ds:
        toks, tags = ex["tokens"], ex["ner_tags"]
        spans, cur_type, cur = [], None, []
        start = 0
        for i, (tok, tag_id) in enumerate(zip(toks, tags)):
            tag = names[int(tag_id)] if isinstance(tag_id, int) and int(tag_id) < len(names) else str(tag_id)
            if tag.startswith("B-"):
                if cur_type in {"PER", "ORG", "LOC"} and cur:
                    spans.append((start, cur_type, "_".join(cur)))
                cur_type, cur, start = tag[2:], [tok], i
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur.append(tok)
            else:
                if cur_type in {"PER", "ORG", "LOC"} and cur:
                    spans.append((start, cur_type, "_".join(cur)))
                cur_type, cur = None, []
        if cur_type in {"PER", "ORG", "LOC"} and cur:
            spans.append((start, cur_type, "_".join(cur)))
        parts = [f"{typ}:{val}" for _, typ, val in sorted(spans)]
        if parts:
            out = " ".join(parts) + " <eos>"
            if validate_output_format(out, "[INSTR]"):
                rows.append({"input": "[INSTR] " + " ".join(toks), "output": out})
    return rows


def cord_examples() -> list[dict[str, str]]:
    rows = []
    for ex in load_all_splits("naver-clova-ix/cord-v2"):
        flat = flatten_json(cord_parse(ex))
        if flat:
            out = compact_json(flat) + " <eos>"
            if validate_output_format(out, "[OCR]"):
                rows.append({"input": "[OCR] " + cord_words(ex), "output": out})
    return rows


def sroie_examples() -> list[dict[str, str]]:
    rows = []
    tag_names = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE", "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]
    for ex in load_all_splits("darentang/sroie"):
        words = ex.get("words") or []
        text = clean_text(ex.get("text") or ex.get("ocr") or " ".join(words))
        vals = {}
        if words and ex.get("ner_tags"):
            tags = []
            for tag in ex.get("ner_tags"):
                tags.append(tag_names[int(tag)] if isinstance(tag, int) and int(tag) < len(tag_names) else str(tag))
            for span in bio_spans(words, tags):
                typ, val = span.split(":", 1)
                vals[typ.lower()] = val.replace("_", " ")
        else:
            for k in ["company", "date", "address", "total"]:
                label = ex.get("label") if isinstance(ex.get("label"), dict) else {}
                vals[k] = clean_text(ex.get(k) or label.get(k) or "")
        if text and all(vals.values()):
            out = json.dumps(vals, ensure_ascii=False, separators=(",", ":")) + " <eos>"
            if validate_output_format(out, "[OCR]"):
                rows.append({"input": "[OCR] " + text, "output": out})
    return rows


def funsd_examples() -> list[dict[str, str]]:
    rows = []
    for ex in load_all_splits("nielsr/funsd"):
        if ex.get("words") and ex.get("ner_tags"):
            words = [clean_text(w) for w in ex["words"]]
            label_names = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
            labels = [
                label_names[int(tag)] if isinstance(tag, int) and int(tag) < len(label_names) else str(tag)
                for tag in ex["ner_tags"]
            ]
            fields: dict[str, list[str]] = defaultdict(list)
            for span in bio_spans(words, labels):
                typ, val = span.split(":", 1)
                fields[typ.lower()].append(val.replace("_", " "))
            flat = {
                f"{typ}_{i + 1}": val
                for typ, values in fields.items()
                for i, val in enumerate(values)
                if val
            }
            if flat:
                out = compact_json(flat) + " <eos>"
                if validate_output_format(out, "[OCR]"):
                    rows.append({"input": "[OCR] " + clean_text(" ".join(words)), "output": out})
            continue

        entities = ex.get("form") or ex.get("entities") or []
        text_parts, q_by_id, a_by_id = [], {}, {}
        for ent in entities:
            words = ent.get("words", [])
            ent_text = clean_text(" ".join(w.get("text", "") if isinstance(w, dict) else str(w) for w in words))
            text_parts.append(ent_text)
            label = ent.get("label")
            if label == "question":
                q_by_id[ent.get("id")] = (ent_text, ent.get("linking", []))
            elif label == "answer":
                a_by_id[ent.get("id")] = ent_text
        pairs = {}
        for _, (q, links) in q_by_id.items():
            for link in links:
                ans_id = link[1] if isinstance(link, (list, tuple)) and len(link) > 1 else link
                if ans_id in a_by_id and q:
                    pairs[q] = a_by_id[ans_id]
        if pairs:
            out = compact_json(pairs) + " <eos>"
            if validate_output_format(out, "[OCR]"):
                rows.append({"input": "[OCR] " + clean_text(" ".join(text_parts)), "output": out})
    return rows


def snips_examples() -> list[dict[str, str]]:
    rows = []
    for ex in load_snips_with_slots():
        intent = str(ex.get("intent") or ex.get("category") or ex.get("label") or "")
        action = SNIPS_ACTION.get(intent)
        utter = clean_text(ex.get("utterance") or ex.get("input") or ex.get("text"))
        slots = []
        raw_slots = ex.get("slots", []) or ex.get("entities", []) or []
        if isinstance(raw_slots, str):
            slots = bio_spans(utter.split(), raw_slots.split())
        else:
            for slot in raw_slots:
                typ = clean_text(slot.get("slot_name") or slot.get("entity") or slot.get("type"))
                val = clean_text(slot.get("entity_value") or slot.get("value") or slot.get("text")).replace(" ", "_")
                if typ and val:
                    slots.append(f"{typ}:{val}")
        if action and utter and slots:
            out = "ACTION:" + action + " " + " ".join(slots) + " <eos>"
            if validate_output_format(out, "[INSTR]"):
                rows.append({"input": "[INSTR] " + utter, "output": out})
    return rows


def bio_spans(words: list[str], labels: list[Any]) -> list[str]:
    spans, cur_t, cur = [], None, []
    for w, lab in zip(words, labels):
        tag = str(lab)
        if tag.startswith("B-"):
            if cur_t and cur:
                spans.append(f"{cur_t}:{'_'.join(cur)}")
            cur_t, cur = tag[2:], [w]
        elif tag.startswith("I-") and cur_t == tag[2:]:
            cur.append(w)
        else:
            if cur_t and cur:
                spans.append(f"{cur_t}:{'_'.join(cur)}")
            cur_t, cur = None, []
    if cur_t and cur:
        spans.append(f"{cur_t}:{'_'.join(cur)}")
    return spans


def atis_examples() -> list[dict[str, str]]:
    rows = []
    for ex in load_atis_with_slots():
        text = clean_text(ex.get("text"))
        intent = str(ex.get("intent") or ex.get("intent_label") or "")
        if intent and not intent.startswith("atis_"):
            intent = "atis_" + intent
        action = ATIS_ACTION.get(intent, "query" if intent.startswith("atis_") else "")
        words = text.split()
        labels = ex.get("slot_labels") or ex.get("slots") or ex.get("ner") or []
        if isinstance(labels, str):
            labels = labels.split()
        spans = bio_spans(words, labels)
        if text and action and spans:
            out = "ACTION:" + action + " " + " ".join(spans) + " <eos>"
            if validate_output_format(out, "[INSTR]"):
                rows.append({"input": "[INSTR] " + text, "output": out})
    return rows


def dolly_examples() -> list[dict[str, str]]:
    rows = []
    for ex in load_all_splits("databricks/databricks-dolly-15k"):
        if ex.get("category") not in {"closed_qa", "information_extraction"} or not clean_text(ex.get("context")):
            continue
        ctx = " ".join(clean_text(ex["context"]).split()[:300])
        inp = f"[INSTR] Context: {ctx} Instruction: {clean_text(ex.get('instruction'))}"
        out = clean_text(ex.get("response")) + " <eos>"
        if out != " <eos>":
            rows.append({"input": inp, "output": out})
    return rows


def msmarco_auto_examples() -> list[dict[str, str]]:
    rows = []
    ms_all = load_all_splits("microsoft/ms_marco", "v1.1").shuffle(seed=42)
    ds = ms_all.select(range(min(200000, len(ms_all))))
    for ex in ds:
        words = clean_text(ex.get("query")).split()
        if len(words) >= 3:
            for k in range(2, len(words)):
                nxt = words[k]
                if nxt.lower() not in STOPWORDS:
                    out = nxt + " <eos>"
                    if validate_output_format(out, "[AUTO]"):
                        rows.append({"input": "[AUTO] " + " ".join(words[:k]), "output": out})
    random.Random(42).shuffle(rows)
    return rows[:500000]


def iter_msmarco_auto_rows(limit: int = 500000) -> Iterable[dict[str, str]]:
    produced = 0
    ms_all = load_all_splits("microsoft/ms_marco", "v1.1").shuffle(seed=42)
    ds = ms_all.select(range(min(200000, len(ms_all))))
    for ex in ds:
        words = clean_text(ex.get("query")).split()
        if len(words) < 3:
            continue
        for k in range(2, len(words)):
            nxt = words[k]
            if nxt.lower() in STOPWORDS:
                continue
            out = nxt + " <eos>"
            if validate_output_format(out, "[AUTO]"):
                yield {"input": "[AUTO] " + " ".join(words[:k]), "output": out}
                produced += 1
                if produced >= limit:
                    return


def stage2(output_dir: str, tokenizer_path: str, max_seq_len: int, overwrite: bool) -> None:
    base = os.path.join(output_dir, "stage2")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    builders = {
        "conll2003": conll_examples,
        "cord-v2": cord_examples,
        "sroie": sroie_examples,
        "funsd": funsd_examples,
        "snips": snips_examples,
        "atis": atis_examples,
        "dolly_filtered": dolly_examples,
    }
    if grouped_stage_complete(base, list(builders)) and not overwrite:
        print(f"{base} exists; skipping")
        print("=== DONE: prepare_datasets.py ===")
        return
    for name, fn in builders.items():
        has_complete_nonempty = all(saved_dataset_has_rows(split_path(base, name, split)) for split in ["train", "test"])
        if has_complete_nonempty and not overwrite:
            print(f"{os.path.join(base, name)} exists; skipping")
            continue
        rows = fn()
        print(f"{name}: {len(rows)} formatted examples")
        split_save(rows, name, base, sp, max_seq_len)
    print("=== DONE: prepare_datasets.py ===")


def stage3(output_dir: str, tokenizer_path: str, max_seq_len: int, overwrite: bool) -> None:
    base = os.path.join(output_dir, "stage3")
    copied_names = ["cord-v2", "sroie", "funsd", "snips", "atis", "dolly_filtered"]
    expected_names = copied_names + ["msmarco_auto"]
    if grouped_stage_complete(base, expected_names) and not overwrite:
        print(f"{base} exists; skipping")
        print("=== DONE: prepare_datasets.py ===")
        return
    os.makedirs(base, exist_ok=True)
    stage2_base = os.path.join(output_dir, "stage2")
    for name in copied_names:
        for split in ["train", "test"]:
            dest = os.path.join(base, name, split)
            if saved_dataset_has_rows(dest) and not overwrite:
                print(f"{dest} exists; skipping")
                continue
            if os.path.exists(dest):
                shutil.rmtree(dest)
            ds = load_from_disk(os.path.join(stage2_base, name, split))
            ds.save_to_disk(dest)
    split_save_from_iter(lambda: iter_msmarco_auto_rows(), "msmarco_auto", base, tokenizer_path, max_seq_len, overwrite)
    print("=== DONE: prepare_datasets.py ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1_text", "1", "lm_base", "lm_prose", "2", "3", "all"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage in {"1_text", "all"}:
        stage1_text(args.output_dir, args.overwrite)
    if args.stage in {"1", "all"}:
        stage1(args.output_dir, args.tokenizer_path, args.max_seq_len or 256, args.overwrite)
    if args.stage in {"lm_base", "all"}:
        lm_base(args.output_dir, args.tokenizer_path, args.max_seq_len or 256, args.overwrite)
    if args.stage in {"lm_prose", "all"}:
        lm_prose(args.output_dir, args.tokenizer_path, args.max_seq_len or 256, args.overwrite)
    if args.stage in {"2", "all"}:
        stage2(args.output_dir, args.tokenizer_path, args.max_seq_len or 512, args.overwrite)
    if args.stage in {"3", "all"}:
        stage3(args.output_dir, args.tokenizer_path, args.max_seq_len or 1024, args.overwrite)


if __name__ == "__main__":
    main()
