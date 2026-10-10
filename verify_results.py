"""Recompute submitted metrics from predictions and verify the common split."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / "results"
    split = json.loads((out / "split.json").read_text())
    labels = split["labels"]
    nyt = pd.read_csv(ROOT / "HW-1" / "nyt.csv")
    assert hashlib.sha256((ROOT / "HW-1" / "nyt.csv").read_bytes()).hexdigest() == split["nyt_sha256"], "NYT source fingerprint differs"
    all_rows = split["train"] + split["validation"] + split["test"]
    assert len(all_rows) == len(nyt) == len(set(all_rows)), "Rows are missing or overlapping"
    methods = ["binary_bow", "word_frequency", "glove", "word2vec_ag", "word2vec_nyt", "bert"]
    summary = pd.read_csv(out / "summary.csv").set_index("method")
    for name in methods:
        predictions = pd.read_csv(out / f"{name}_predictions.csv")
        saved = json.loads((out / f"{name}.json").read_text())
        assert predictions.row_id.tolist() == split["test"], f"{name}: different test order"
        assert predictions.gold.tolist() == nyt.iloc[split["test"]].label.tolist(), f"{name}: gold labels differ"
        assert set(predictions.prediction) <= set(labels)
        accuracy = accuracy_score(predictions.gold, predictions.prediction)
        macro = f1_score(predictions.gold, predictions.prediction, labels=labels, average="macro")
        assert np.isclose(accuracy, saved["test"]["accuracy"], atol=1e-12)
        assert np.isclose(macro, saved["test"]["macro_f1"], atol=1e-12)
        assert np.isclose(accuracy, summary.loc[name, "test_accuracy"], atol=1e-12)
        assert np.isclose(macro, summary.loc[name, "test_macro_f1"], atol=1e-12)
        assert confusion_matrix(predictions.gold, predictions.prediction, labels=labels).tolist() == saved["test"]["confusion_matrix"]
        print(f"PASS {name}: accuracy={accuracy:.6f}, macro_f1={macro:.6f}")
    bert = json.loads((out / "bert.json").read_text())
    assert len(bert["history"]) == bert["config"]["epochs"] == 3
    assert bert["config"]["max_length"] == 64
    assert bert["selected_epoch"] == max(bert["history"], key=lambda row: row["validation"]["macro_f1"])["epoch"]
    print("PASS shared split, all six prediction files, and BERT training configuration")


if __name__ == "__main__":
    main()
