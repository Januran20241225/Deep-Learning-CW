"""
Extract false positives and false negatives for the coursework entailment model.
Run this file from the project directory containing factcheck.py, factchecking_main.py,
and the data/ folder.
"""
import argparse
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factcheck import EntailmentModel, EntailmentFactChecker
from factchecking_main import read_passages, read_fact_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    print("Loading development examples...")
    fact_to_passage = read_passages(args.passages_path)
    examples = read_fact_examples(args.labels_path, fact_to_passage)
    print(f"Loaded {len(examples)} examples.")

    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if args.cuda:
        model.to("cuda")

    ent_model = EntailmentModel(model, tokenizer, args.cuda)
    checker = EntailmentFactChecker(ent_model, threshold=args.threshold)

    false_positives = []
    false_negatives = []

    print("Running predictions...")
    for index, example in enumerate(examples, start=1):
        gold = "NS" if example.label == "IR" else example.label
        predicted = checker.predict(example.fact, example.passages)

        record = {
            "example_index": index,
            "fact": example.fact,
            "original_gold_label": example.label,
            "gold_label": gold,
            "predicted_label": predicted,
            "passages": example.passages,
        }

        if gold == "NS" and predicted == "S":
            false_positives.append(record)
        elif gold == "S" and predicted == "NS":
            false_negatives.append(record)

        if index % 25 == 0:
            print(f"Processed {index}/{len(examples)}")

    selected_fp = false_positives[:args.limit]
    selected_fn = false_negatives[:args.limit]

    output = {
        "threshold": args.threshold,
        "total_examples": len(examples),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives": selected_fp,
        "false_negatives": selected_fn,
    }

    output_path = "entailment_errors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nExtraction complete.")
    print(f"False positives found: {len(false_positives)}")
    print(f"False negatives found: {len(false_negatives)}")
    print(f"Saved the first {len(selected_fp)} of each type to: {output_path}")
    print("Upload entailment_errors.json here for the error analysis.")


if __name__ == "__main__":
    main()
