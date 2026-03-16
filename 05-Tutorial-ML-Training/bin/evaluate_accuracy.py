#!/usr/bin/env python3

"""
Evaluate classification accuracy by comparing predictions against ground truth.

Reads predictions.json and crop_catalog.csv, computes accuracy metrics
including per-class precision, recall, F1, and confusion matrix.

Usage:
    ./evaluate_accuracy.py \
        --predictions predictions.json \
        --catalog crop_catalog.csv \
        --output accuracy_results.json
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_predictions(predictions_file: str) -> dict:
    """Load predictions JSON file."""
    with open(predictions_file, 'r') as f:
        return json.load(f)


def load_catalog(catalog_file: str) -> dict:
    """Load crop catalog CSV and return mapping of filename to category."""
    filename_to_category = {}
    with open(catalog_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            category = row.get('category', '')
            if filename and category:
                filename_to_category[filename] = category
    return filename_to_category


def compute_accuracy(predictions_file: str, catalog_file: str) -> dict:
    """Compute accuracy metrics comparing predictions to ground truth."""
    data = load_predictions(predictions_file)
    predictions = data.get('predictions', [])
    catalog = load_catalog(catalog_file)

    if not predictions:
        logger.warning("No predictions found")
        return {}

    if not catalog:
        logger.warning("No catalog entries found")
        return {}

    # Collect all labels
    all_labels = set()
    correct = 0
    total = 0
    unmatched = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    # For confusion matrix: confusion[true_label][predicted_label] = count
    confusion = defaultdict(lambda: defaultdict(int))

    for pred in predictions:
        if 'error' in pred:
            continue

        filename = pred.get('filename', '')
        predicted_class = pred.get('predicted_class', '')

        if filename not in catalog:
            unmatched += 1
            continue

        true_class = catalog[filename]
        all_labels.add(true_class)
        all_labels.add(predicted_class)

        total += 1
        per_class_total[true_class] += 1
        confusion[true_class][predicted_class] += 1

        if predicted_class == true_class:
            correct += 1
            per_class_correct[true_class] += 1

    if total == 0:
        logger.warning("No matched predictions found")
        return {}

    # Sort labels for consistent ordering
    labels = sorted(all_labels)

    # Build confusion matrix as 2D list
    matrix = []
    for true_label in labels:
        row = []
        for pred_label in labels:
            row.append(confusion[true_label][pred_label])
        matrix.append(row)

    # Compute per-class precision, recall, F1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        # FP: other classes predicted as this label
        fp = sum(confusion[other][label] for other in labels if other != label)
        # FN: this class predicted as other labels
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_total = per_class_total[label]
        class_correct = per_class_correct[label]
        accuracy = class_correct / class_total if class_total > 0 else 0.0

        per_class[label] = {
            "total": class_total,
            "correct": class_correct,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    overall_accuracy = correct / total if total > 0 else 0.0

    results = {
        "evaluated_at": datetime.now().isoformat(),
        "overall_accuracy": round(overall_accuracy, 4),
        "total_evaluated": total,
        "correct": correct,
        "incorrect": total - correct,
        "unmatched": unmatched,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classification accuracy against ground truth"
    )

    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Input predictions JSON file'
    )

    parser.add_argument(
        '--catalog', '-c',
        type=str,
        required=True,
        help='Input crop catalog CSV file with ground truth labels'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output accuracy results JSON file'
    )

    args = parser.parse_args()

    logger.info(f"Evaluating accuracy: predictions={args.predictions}, catalog={args.catalog}")

    results = compute_accuracy(args.predictions, args.catalog)

    if results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Accuracy results saved to {args.output}")
        logger.info(f"Overall accuracy: {results['overall_accuracy']:.2%} "
                     f"({results['correct']}/{results['total_evaluated']})")
        if results['unmatched'] > 0:
            logger.warning(f"Unmatched predictions (not in catalog): {results['unmatched']}")
    else:
        logger.error("No accuracy results computed")


if __name__ == "__main__":
    main()
