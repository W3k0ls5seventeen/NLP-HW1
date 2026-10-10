"""Generate the English PDF report from the six measured result files."""
from html import escape
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent
NAMES = {"binary_bow": "Binary Bag of Words", "word_frequency": "Word Frequency",
         "glove": "Pretrained GloVe", "word2vec_ag": "Word2Vec on AG News",
         "word2vec_nyt": "Word2Vec on NYT", "bert": "Fine tuned BERT"}


def main():
    results = {name: json.loads((ROOT / "results" / f"{name}.json").read_text()) for name in NAMES}
    stats = json.loads((ROOT / "results" / "data_summary.json").read_text())
    env = json.loads((ROOT / "results" / "environment.json").read_text())
    sources = json.loads((ROOT / "results" / "sources.json").read_text())
    ordered = sorted(results, key=lambda name: results[name]["test"]["macro_f1"], reverse=True)
    best = ordered[0]
    styles = {
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=14.8, spaceAfter=9),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=9.0, leading=12, spaceAfter=7),
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=27, spaceAfter=13),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=15, leading=19, spaceAfter=12),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=11.3, leading=15,
                                   spaceBefore=7, spaceAfter=7, keepWithNext=True),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9.2, leading=12),
        "cellbold": ParagraphStyle("cellbold", fontName="Helvetica-Bold", fontSize=9.2, leading=12),
    }
    story = []

    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))
    def h(text):
        p(text, "heading")
    def table(rows, widths):
        cells = [[Paragraph(escape(str(value)), styles["cellbold" if i == 0 else "cell"])
                  for value in row] for i, row in enumerate(rows)]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EDF2")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([t, Spacer(1, 10)])
    def page(title):
        story.append(PageBreak())
        p(title, "section")
    def value(name, key, split="test"):
        return results[name][split][key]
    def percent(v):
        return f"{100 * v:.2f}%"

    p("Homework 1 Text Classification", "title")
    p("Bag of Words Word Embeddings and BERT", "heading")
    p("Natural Language Processing  |  Experiment report  |  October 5 2026", "small")
    p(f"Six text classifiers were evaluated on a shared NYT test set: two Bag of Words models, "
      f"three averaged word embedding models, and a fine tuned BERT model. "
      f"{NAMES[best]} achieved the highest measured test Macro-F1 of {value(best, 'macro_f1'):.4f} "
      f"with Accuracy of {percent(value(best, 'accuracy'))}. The comparison evaluates the required "
      "representations under a common data split, while preserving their prescribed training procedures.")
    h("1 Data and evaluation protocol")
    p("The supplied NYT file contains 11,519 labeled news articles in three categories. AG News contains "
      "90,000 unlabeled texts and is used only to learn one Word2Vec model. All rows are retained. "
      "NYT is shuffled and stratified with random seed 42 into approximately 80% training, 10% validation "
      "and 10% test data. The same saved row indices are reused in all six experiments.")
    table([["Split", "Business", "Politics", "Sports", "Total"]] +
          [[name.title(), stats["splits"][name]["class_counts"]["business"],
            stats["splits"][name]["class_counts"]["politics"], stats["splits"][name]["class_counts"]["sports"],
            stats["splits"][name]["rows"]] for name in ("train", "validation", "test")], [120, 90, 90, 90, 110])
    p("The training set is the only source of the BoW vocabulary, NYT Word2Vec embeddings, feature "
      "standardization statistics and supervised model parameters. Validation Macro-F1 selects the "
      "Logistic Regression regularization strength and the BERT checkpoint. Test labels do not select "
      "hyperparameters, epochs or preprocessing choices. Models are not refitted on validation rows.")
    h("Evaluation metrics")
    p("Accuracy is the fraction of correct predictions. For class k, precision is TP/(TP+FP), recall "
      "is TP/(TP+FN), and F1 is 2PR/(P+R). Macro-F1 is the unweighted mean of the three class F1 scores, "
      "with zero division scored as zero. This matters because sports accounts for 75% of the test set. "
      "An always-sports classifier reaches 75.00% Accuracy but only 0.2857 Macro-F1.")
    h("Repeated texts")
    p("NYT contains 72 duplicate text rows. Under the prescribed row-level shuffle, 14 distinct exact "
      "texts occur in both training and validation, 10 in training and test, and 1 in validation and test. "
      "The row indices are disjoint, but the repeated content weakens the independence of evaluation. "
      "All methods face the same split, and the overlap is disclosed rather than silently removing rows.", "small")

    page("2 Methods and training settings")
    h("Two Bag of Words representations")
    vocab = results["binary_bow"]["vocabulary_size"]
    p(f"Text is lowercased and split into Unicode word tokens using the regular expression \\b\\w+\\b. "
      f"Single-character words and numbers are retained. There is no stemming, stopword filtering or "
      f"n-gram expansion. A training-only vocabulary of {vocab:,} words defines a sparse document vector. "
      "Binary BoW sets x_i to 1 if word i occurs and 0 otherwise. Word Frequency sets x_i to the raw "
      "occurrence count. For vocabulary [apple, banana], the text 'apple apple banana' becomes [1, 1] "
      "in binary form and [2, 1] in frequency form. Neither representation is normalized or TF-IDF weighted.")
    p("Both use L2-regularized multinomial Logistic Regression with the lbfgs solver, tolerance 1e-4 "
      "and a maximum of 2,000 iterations. C is chosen from {0.01, 0.1, 1, 10} by validation Macro-F1. "
      "A smaller C means stronger regularization. Classes are unweighted, and every candidate must converge.")
    h("Three averaged word embedding representations")
    p("Each document receives a 100-dimensional vector by averaging the embeddings of its valid token "
      "occurrences: e(d) = sum(e(w_i))/n_valid. Repeated words contribute repeatedly. Out-of-vocabulary "
      "tokens are skipped; a document with no valid tokens receives a zero vector. Each dimension is "
      "standardized using training means and standard deviations, then classified with the same "
      "Logistic Regression search as above.")
    p("<b>Pretrained GloVe.</b> The original glove.6B.100d.txt vectors are used [1]. The external "
      "vocabulary and weights are fixed. Loading the vectors needed for validation and test tokens "
      "performs a lookup in this pretrained vocabulary, without learning from those documents.")
    p("<b>Word2Vec on AG News.</b> Skip-gram embeddings are learned from all 90,000 unlabeled AG texts [2]. "
      "The model uses dimension 100, window 5, minimum count 2, 5 negative samples, subsampling 1e-3 "
      "and 5 embedding epochs. Seed 42, one worker and a stable CRC32 hash control initialization and "
      "training variation. NYT labels are used only by the downstream classifier.")
    p("<b>Word2Vec on NYT.</b> The same embedding configuration is applied to the 9,215 NYT training "
      "texts. Validation and test articles are excluded from embedding training. This isolates the "
      "comparison between an external news corpus and the task's own training corpus.")
    h("BERT fine tuning")
    p("The specified google-bert/bert-base-uncased checkpoint is fine tuned end to end with a three-class "
      "classification head [3, 4]. WordPiece tokenization truncates and pads each input to max_length=64, "
      "including special tokens. Training lasts exactly 3 epochs, with batch size 32, AdamW learning "
      "rate 2e-5, weight decay 0.01 except bias and LayerNorm weights, 10% linear warmup, linear learning "
      "rate decay and gradient norm clipping at 1.0. CUDA mixed precision is used. After all three "
      "epochs, the checkpoint with highest validation Macro-F1 is evaluated on the test set.")

    page("3 Experimental results")
    p("Table 1 reports the required metrics on the same 1,152 test articles. Validation metrics are "
      "shown to document model selection. Scores are fractions except where Accuracy is shown as a percentage.")
    table([["Representation", "Val Acc", "Val F1", "Test Acc", "Test F1"]] +
          [[NAMES[name], percent(value(name, "accuracy", "validation")), f"{value(name, 'macro_f1', 'validation'):.4f}",
            percent(value(name, "accuracy")), f"{value(name, 'macro_f1'):.4f}"] for name in NAMES], [200, 75, 75, 75, 75])
    p("Table 1  Accuracy and Macro-F1 for all required experiments", "small")
    h("Regularization selected on validation")
    p("The selected C values are " + "; ".join(f"{NAMES[name]}: {results[name]['selected_C']:g}" for name in list(NAMES)[:-1]) +
      ". Complete candidate scores and solver iteration counts are stored in the corresponding result JSON files.")
    h("Embedding token coverage")
    table([["Representation", "Train coverage", "Val coverage", "Test coverage", "Empty test docs"]] +
          [[NAMES[name]] + [percent(results[name]["coverage"][split]["coverage"]) for split in ("train", "validation", "test")] +
           [results[name]["coverage"]["test"]["zero_vector_documents"]] for name in ("glove", "word2vec_ag", "word2vec_nyt")],
          [180, 80, 80, 80, 80])
    p("Coverage counts occurrences, including repeated words. Empty documents receive zero vectors.", "small")
    h("BERT validation after each epoch")
    table([["Epoch", "Train loss", "Val Accuracy", "Val Macro-F1"]] +
          [[row["epoch"], f"{row['train_loss']:.4f}", percent(row["validation"]["accuracy"]),
            f"{row['validation']['macro_f1']:.4f}"] for row in results["bert"]["history"]], [65, 105, 165, 165])
    p(f"All three epochs were completed. Epoch {results['bert']['selected_epoch']} was selected by validation "
      "Macro-F1. Test results use this selected checkpoint. No epoch was chosen using test labels.")
    p("Elapsed times exclude environment installation and resource downloads. They are measured for "
      "this run and are not a controlled speed benchmark: " + "; ".join(f"{NAMES[name]} {results[name]['seconds']:.1f}s" for name in NAMES) + ".", "small")

    page("4 Analysis and limitations")
    h("Representation comparison")
    delta_acc = 100 * (value("word_frequency", "accuracy") - value("binary_bow", "accuracy"))
    delta_f1 = value("word_frequency", "macro_f1") - value("binary_bow", "macro_f1")
    p(f"Word Frequency improves over Binary BoW by {delta_acc:.2f} percentage points in Accuracy and "
      f"{delta_f1:.4f} in Macro-F1. The frequency model makes "
      f"{round((1-value('word_frequency','accuracy'))*1152)} test errors, compared with "
      f"{round((1-value('binary_bow','accuracy'))*1152)} for binary features. This small difference is "
      "consistent with useful repeated topical terms, but a single split does not establish a reliable "
      "general advantage. Both retain explicit lexical evidence that can be informative for news topics.")
    wdelta = value("word2vec_nyt", "macro_f1") - value("word2vec_ag", "macro_f1")
    p(f"NYT-trained Word2Vec achieves {wdelta:+.4f} Macro-F1 relative to AG-trained Word2Vec. "
      "The observed result is consistent with a benefit from matching the embedding corpus to the target "
      "domain. NYT has fewer documents, but document count is not token count. These results do not "
      "isolate training-token volume, news distribution or vocabulary coverage as causal factors. "
      "Mean pooling compresses a whole article "
      "into 100 numbers and discards word order and local context; this can dilute rare category-specific terms.")
    p(f"BERT reaches {percent(value('bert', 'accuracy'))} Accuracy and {value('bert', 'macro_f1'):.4f} Macro-F1. "
      "Its contextual representation differs from the fixed averages, but its input is limited to the "
      "first 64 WordPiece tokens including special tokens, while BoW and averaged embeddings use the "
      "whole text. The comparison therefore combines representation changes with different input budgets "
      "and training procedures. The observed ranking should not be generalized to unconstrained BERT models.")
    h("Per class test F1")
    table([["Representation", "Business", "Politics", "Sports"]] +
          [[NAMES[name]] + [f"{results[name]['test']['per_class'][label]['f1-score']:.4f}" for label in stats["labels"]]
           for name in NAMES], [230, 90, 90, 90])
    h("BERT test confusion matrix")
    table([["Gold / Predicted", "Business", "Politics", "Sports"]] +
          [[label.title()] + row for label, row in zip(stats["labels"], results["bert"]["test"]["confusion_matrix"])],
          [230, 90, 90, 90])
    h("Scope of the conclusions")
    p("The test set has only 143 business and 145 politics articles, so a few mistakes can move Macro-F1. "
      "These are single-seed results without confidence intervals or significance testing. Exact text "
      "overlap across splits may make the task easier. No claim is made that the split is independent "
      "at the story level or that the ranking transfers to new publishers, dates or categories. More "
      "reliable generalization estimates would require deduplicated or grouped splits and repeated runs.", "small")

    page("5 Reproducibility and references")
    h("Execution and saved evidence")
    p(f"The experiments used Python {env['python']}, Windows 11 and {results['bert']['gpu']}. Main versions "
      "were scikit-learn 1.6.1, Gensim 4.4.0, PyTorch 2.8.0 with CUDA 12.8 and Transformers 4.57.1. "
      "Seed 42 is used for the data split and training. GPU floating-point results can vary across "
      "hardware and software despite fixed seeds.")
    p("README.md documents installation, resource acquisition and execution. Run prepare_assets.py "
      "to obtain the required files, then run run_experiments.py --part all. The code saves the shared "
      "split and input fingerprint in results/split.json, aggregate results in results/summary.csv, "
      "per-method settings and metrics in JSON, and test row IDs, gold labels and predictions in CSV. "
      "verify_results.py recomputes Accuracy, Macro-F1 and confusion matrices from these predictions and "
      "checks the identical test-row ordering and the three-epoch BERT configuration.")
    p("GloVe was extracted from the 100-dimensional member of the original Stanford glove.6B.zip using "
      "HTTP byte ranges and ZIP integrity checking. BERT weights were delivered through a ModelScope "
      "mirror because the original large-file CDN was unavailable. The entire downloaded safetensors "
      "file was SHA-256 checked against the official Hugging Face file at the pinned revision; it is "
      "byte-for-byte identical. The tokenizer and configuration came from that Hugging Face revision.")
    p("BERT revision: " + escape(sources["bert"]["revision"]), "small")
    p("BERT SHA256: " + escape(sources["bert"]["sha256"]), "small")
    p("GloVe SHA256: " + escape(sources["glove"]["sha256"]), "small")
    h("Use of tools and AI assistance")
    p("The implementation uses scikit-learn for sparse vectorization, Logistic Regression and metrics; "
      "Gensim for Word2Vec; and PyTorch and Hugging Face Transformers for BERT fine tuning. AI assistance "
      "was used to implement and execute the experiments and draft the report. All reported experimental "
      "measurements were obtained by running the submitted code on the supplied data.")
    h("References")
    p("[1] Jeffrey Pennington, Richard Socher and Christopher Manning. 2014. GloVe: Global Vectors "
      "for Word Representation. EMNLP, pages 1532-1543. "
      '<link href="https://aclanthology.org/D14-1162/">https://aclanthology.org/D14-1162/</link>', "small")
    p("[2] Tomas Mikolov, Kai Chen, Greg Corrado and Jeffrey Dean. 2013. Efficient Estimation of Word "
      "Representations in Vector Space. arXiv:1301.3781. "
      '<link href="https://arxiv.org/abs/1301.3781">https://arxiv.org/abs/1301.3781</link>', "small")
    p("[3] Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. 2019. BERT: Pre-training "
      "of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, pages 4171-4186. "
      '<link href="https://aclanthology.org/N19-1423/">https://aclanthology.org/N19-1423/</link>', "small")
    p('[4] Hugging Face. google-bert/bert-base-uncased model card and pinned checkpoint. '
      '<link href="https://huggingface.co/google-bert/bert-base-uncased">'
      'https://huggingface.co/google-bert/bert-base-uncased</link>', "small")
    p("[5] Course Homework 1 specification and the supplied HW-1 datasets.", "small")

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(47, 25, "Homework 1 Text Classification")
        canvas.drawRightString(A4[0] - 47, 25, str(doc.page))

    target = ROOT / "HW1_Report.pdf"
    document = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=47, rightMargin=47,
                                 topMargin=42, bottomMargin=45,
                                 title="Homework 1 Text Classification", author="", subject="NYT classification experiments")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    main()
