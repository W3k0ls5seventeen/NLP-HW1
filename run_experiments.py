"""HW1: two BoW methods, three averaged embeddings, and three-epoch BERT."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import time
import warnings
import zlib

# Set these before importing numeric libraries to avoid CPU oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 42
TOKEN_PATTERN = r"(?u)\b\w+\b"
TOKENIZER = re.compile(TOKEN_PATTERN)
METHODS = ["binary_bow", "word_frequency", "glove", "word2vec_ag", "word2vec_nyt", "bert"]


def tokenize(text):
    return TOKENIZER.findall(text.lower())


def stable_hash(word):
    return zlib.crc32(word.encode("utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics(y, prediction, labels):
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
            "per_class": classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0)}


def load_data(out):
    path = ROOT / "HW-1" / "nyt.csv"
    data = pd.read_csv(path)
    if list(data.columns) != ["text", "label"] or data.isna().any().any():
        raise ValueError("NYT must have complete text and label columns")
    labels = sorted(data.label.unique().tolist())
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    split_file = out / "split.json"
    if split_file.exists():
        split = json.loads(split_file.read_text())
        if split["nyt_sha256"] != fingerprint or split["seed"] != SEED:
            raise ValueError("Existing split belongs to different data or seed")
    else:
        train, held = train_test_split(np.arange(len(data)), test_size=0.2,
                                      random_state=SEED, shuffle=True, stratify=data.label)
        val, test = train_test_split(held, test_size=0.5, random_state=SEED,
                                    shuffle=True, stratify=data.label.iloc[held])
        split = {"seed": SEED, "nyt_sha256": fingerprint, "labels": labels,
                 "train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
        write_json(split_file, split)
    groups = [split[key] for key in ("train", "validation", "test")]
    assert len(set(sum(groups, []))) == len(data)
    assert sum(map(len, groups)) == len(data)
    frames = [data.iloc[idx].copy() for idx in groups]
    sets = [set(frame.text) for frame in frames]
    stats = {"rows": len(data), "labels": labels, "duplicate_text_rows": int(data.text.duplicated().sum()),
             "splits": {key: {"rows": len(frame), "class_counts": frame.label.value_counts().to_dict()}
                        for key, frame in zip(("train", "validation", "test"), frames)},
             "exact_text_overlap": {"train_validation": len(sets[0] & sets[1]),
                                    "train_test": len(sets[0] & sets[2]),
                                    "validation_test": len(sets[1] & sets[2])}}
    write_json(out / "data_summary.json", stats)
    return frames, labels


def record(out, name, result, test_frame, predictions):
    write_json(out / f"{name}.json", result)
    pd.DataFrame({"row_id": test_frame.index, "gold": test_frame.label.to_numpy(),
                  "prediction": predictions}).to_csv(out / f"{name}_predictions.csv", index=False)
    summarize(out)
    print(name, json.dumps({k: result["test"][k] for k in ("accuracy", "macro_f1")}), flush=True)


def summarize(out):
    rows = []
    for name in METHODS:
        file = out / f"{name}.json"
        if file.exists():
            r = json.loads(file.read_text())
            rows.append({"method": name, "validation_accuracy": r["validation"]["accuracy"],
                         "validation_macro_f1": r["validation"]["macro_f1"],
                         "test_accuracy": r["test"]["accuracy"], "test_macro_f1": r["test"]["macro_f1"],
                         "seconds": r["seconds"]})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


def fit_lr(features, frames, labels, standardize=False):
    xtrain, xval, xtest = features
    trials = []
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        print(f"LR fitting C={c}, features={xtrain.shape}", flush=True)
        estimator = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-4, random_state=SEED)
        model = make_pipeline(StandardScaler(), estimator) if standardize else estimator
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(xtrain, frames[0].label)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError(f"LR did not converge at C={c}; do not report unconverged results")
        validation = metrics(frames[1].label, model.predict(xval), labels)
        trials.append({"C": c, "accuracy": validation["accuracy"], "macro_f1": validation["macro_f1"],
                       "n_iter": estimator.n_iter_.tolist()})
        if best is None or validation["macro_f1"] > best[0]:
            best = (validation["macro_f1"], c, model, validation)
    _, c, model, validation = best
    prediction = model.predict(xtest)
    return model, prediction, {"selected_C": c, "C_trials": trials, "validation": validation,
                                "test": metrics(frames[2].label, prediction, labels)}


def run_bow(frames, labels, out, checkpoints):
    vectorizer = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, dtype=np.float64)
    begin = time.perf_counter()
    train = vectorizer.fit_transform(frames[0].text)
    count_features = [train] + [vectorizer.transform(frame.text) for frame in frames[1:]]
    vocabulary_seconds = time.perf_counter() - begin
    for name in ("binary_bow", "word_frequency"):
        begin = time.perf_counter()
        features = [matrix.copy() for matrix in count_features]
        if name == "binary_bow":
            for matrix in features:
                matrix.data.fill(1)
                assert set(np.unique(matrix.data)) <= {1}
        model, prediction, result = fit_lr(features, frames, labels)
        result.update({"seconds": time.perf_counter() - begin + vocabulary_seconds,
                       "vocabulary_size": len(vectorizer.vocabulary_), "train_nnz": train.nnz,
                       "token_pattern": TOKEN_PATTERN, "normalization": "none", "vocabulary_fit": "NYT training only"})
        joblib.dump({"vectorizer": vectorizer, "classifier": model, "binary": name == "binary_bow"},
                    checkpoints / f"{name}.joblib")
        record(out, name, result, frames[2], prediction)


def mean_embeddings(tokens, vectors):
    features = np.zeros((len(tokens), 100), dtype=np.float32)
    known = total = empty = 0
    for i, words in enumerate(tokens):
        valid = [word for word in words if word in vectors]
        total += len(words)
        known += len(valid)
        if valid:
            features[i] = np.mean([vectors[word] for word in valid], axis=0)
        else:
            empty += 1
    return features, {"known_tokens": known, "total_tokens": total,
                      "coverage": known / max(1, total), "zero_vector_documents": empty}


def run_embeddings(frames, labels, out, checkpoints, selected):
    from gensim.models import Word2Vec
    tokens = [[tokenize(text) for text in frame.text] for frame in frames]
    for name in selected:
        begin = time.perf_counter()
        info = {"dimension": 100, "pooling": "occurrence-weighted mean of in-vocabulary tokens"}
        if name == "glove":
            needed = {word for documents in tokens for words in documents for word in words}
            vectors = {}
            source = ROOT / "assets" / "glove.6B.100d.txt"
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    word, _, values = line.partition(" ")
                    if word in needed:
                        vector = np.fromstring(values, sep=" ", dtype=np.float32)
                        if vector.shape != (100,):
                            raise ValueError(f"Invalid GloVe vector for {word}")
                        vectors[word] = vector
            info["source"] = "glove.6B.100d.txt"
        else:
            if name == "word2vec_ag":
                corpus = [tokenize(text) for text in pd.read_csv(ROOT / "HW-1" / "ag.csv").text]
                info["corpus"] = "all 90000 AG News texts without labels"
            else:
                corpus = tokens[0]
                info["corpus"] = "NYT training split only"
            config = dict(vector_size=100, window=5, min_count=2, workers=1, sg=1,
                          negative=5, epochs=5, seed=SEED, sample=1e-3)
            print(f"Training {name} on {len(corpus)} documents", flush=True)
            embedding = Word2Vec(sentences=corpus, hashfxn=stable_hash, **config)
            embedding.save(str(checkpoints / f"{name}.model"))
            vectors = embedding.wv
            info["word2vec_config"] = config
            info["word2vec_hash"] = "CRC32"
            info["embedding_vocabulary_size"] = len(vectors)
            del corpus
        pairs = [mean_embeddings(documents, vectors) for documents in tokens]
        model, prediction, result = fit_lr([pair[0] for pair in pairs], frames, labels, standardize=True)
        result.update(info)
        result["coverage"] = dict(zip(("train", "validation", "test"), [pair[1] for pair in pairs]))
        result["seconds"] = time.perf_counter() - begin
        joblib.dump(model, checkpoints / f"{name}_lr.joblib")
        record(out, name, result, frames[2], prediction)


def run_bert(frames, labels, out, checkpoints):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    begin = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("BERT device", device, flush=True)
    source = ROOT / "assets" / "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    label2id = {label: i for i, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
        local_files_only=True, attn_implementation="sdpa").to(device)

    class NewsDataset(Dataset):
        def __init__(self, frame):
            self.encoded = tokenizer(frame.text.tolist(), padding="max_length", truncation=True,
                                     max_length=64, return_tensors="pt")
            self.y = torch.tensor([label2id[label] for label in frame.label])
        def __len__(self):
            return len(self.y)
        def __getitem__(self, index):
            return {**{key: value[index] for key, value in self.encoded.items()}, "labels": self.y[index]}

    datasets = [NewsDataset(frame) for frame in frames]
    generator = torch.Generator().manual_seed(SEED)
    loaders = [DataLoader(dataset, batch_size=32, shuffle=(i == 0), num_workers=0,
                          pin_memory=device.type == "cuda", generator=generator if i == 0 else None)
               for i, dataset in enumerate(datasets)]
    decay = [p for name, p in model.named_parameters() if p.requires_grad and not ("bias" in name or "LayerNorm.weight" in name)]
    no_decay = [p for name, p in model.named_parameters() if p.requires_grad and ("bias" in name or "LayerNorm.weight" in name)]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=2e-5)
    steps = 3 * len(loaders[0])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * steps), num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def predict(loader):
        model.eval()
        predicted = []
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(**batch).logits
                predicted.extend(logits.argmax(dim=-1).cpu().tolist())
        return np.array([labels[i] for i in predicted])

    history = []
    best_f1 = -1
    saved = checkpoints / "bert_best"
    for epoch in range(1, 4):
        model.train()
        loss_sum = 0.0
        for step, batch in enumerate(loaders[0], 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            loss_sum += loss.item() * len(batch["labels"])
            if step % 50 == 0:
                print(f"BERT epoch {epoch}/3 step {step}/{len(loaders[0])} loss {loss.item():.4f}", flush=True)
        validation = metrics(frames[1].label, predict(loaders[1]), labels)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(frames[0]), "validation": validation})
        print(f"BERT epoch {epoch} validation Macro-F1 {validation['macro_f1']:.6f}", flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(saved)
            tokenizer.save_pretrained(saved)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(saved, local_files_only=True).to(device)
    prediction = predict(loaders[2])
    result = {"seconds": time.perf_counter() - begin,
              "validation": history[best_epoch - 1]["validation"], "test": metrics(frames[2].label, prediction, labels),
              "history": history, "selected_epoch": best_epoch, "model_id": "google-bert/bert-base-uncased",
              "config": {"epochs": 3, "max_length": 64, "batch_size": 32, "learning_rate": 2e-5,
                         "weight_decay": 0.01, "warmup_ratio": 0.1, "optimizer": "AdamW", "clip_norm": 1.0},
              "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    record(out, "bert", result, frames[2], prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "bow", "embeddings", "glove", "word2vec", "bert"], default="all")
    args = parser.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    checkpoints = ROOT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    frames, labels = load_data(out)
    versions = {package: importlib.metadata.version(package) for package in
                ["numpy", "pandas", "scipy", "scikit-learn", "gensim", "transformers", "torch"]
                if importlib.util.find_spec(package.replace("-", "_")) is not None or package == "scikit-learn"}
    write_json(out / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                                           "packages": versions, "seed": SEED})
    if args.part in ("all", "bow"):
        run_bow(frames, labels, out, checkpoints)
    if args.part in ("all", "embeddings", "glove", "word2vec"):
        selected = {"all": ["glove", "word2vec_ag", "word2vec_nyt"],
                    "embeddings": ["glove", "word2vec_ag", "word2vec_nyt"],
                    "glove": ["glove"], "word2vec": ["word2vec_ag", "word2vec_nyt"]}[args.part]
        run_embeddings(frames, labels, out, checkpoints, selected)
    if args.part in ("all", "bert"):
        run_bert(frames, labels, out, checkpoints)


if __name__ == "__main__":
    main()
