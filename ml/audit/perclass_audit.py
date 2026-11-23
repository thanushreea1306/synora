from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"C:\Users\thanu\synora\ml\reports\synora_baseline_v1_20261121")


def main() -> int:
    with (ROOT / "per_class_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [
        (line_number, row)
        for line_number, row in enumerate(rows, start=2)
        if row["model"] == "random_forest" and row["split"] == "test"
    ]
    worst = sorted(test_rows, key=lambda item: (float(item[1]["f1_score"]), item[1]["class_label"]))[:5]
    matrix = json.loads((ROOT / "confusion_matrices.json").read_text(encoding="utf-8"))["random_forest"]["test"]
    labels = [row["class_label"] for _, row in sorted(test_rows, key=lambda item: int(item[1]["model_index"]))]
    print("worst_test_f1")
    for line_number, row in worst:
        index = int(row["model_index"])
        confusions = [
            {
                "true": labels[index],
                "predicted": labels[target],
                "count": matrix[index][target],
            }
            for target in range(len(labels))
            if target != index and matrix[index][target] > 0
        ]
        print(
            json.dumps(
                {
                    "csv_line": line_number,
                    "class": row["class_label"],
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1_score": float(row["f1_score"]),
                    "support": int(row["support"]),
                    "confusions": confusions,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
