"""Arithmetic checks of reported table values, without claiming experimental verification."""

import csv
from pathlib import Path

from sanet.io import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    findings = []
    for table, precision, recall, f1 in [
        (4, "P (%)", "R (%)", "F1 (%)"),
        (6, "P (%)", "R (%)", "F1 (%)"),
        (7, "Precision", "Recall", "F1@50 (%)"),
    ]:
        with open(root / f"docs/manuscript/table-{table}.csv", newline="") as f:
            rows = list(csv.reader(f))
        headers = [h.strip() for h in rows[0]]
        for row in rows[1:]:
            values = dict(zip(headers, row))
            p, r, reported = (float(values[key]) for key in [precision, recall, f1])
            calculated = 2 * p * r / (p + r) if p + r else 0
            findings.append(
                {
                    "table": table,
                    "row": row[0],
                    "precision_percent": p,
                    "recall_percent": r,
                    "reported_f1_percent": reported,
                    "harmonic_f1_percent": calculated,
                    "difference_percentage_points": reported - calculated,
                    "review_if_same_micro_counts": abs(reported - calculated) > 0.15,
                }
            )
    write_json(
        root / "results/verification/manuscript-arithmetic.json",
        {
            "purpose": "Arithmetic check only; source observations/protocols unavailable",
            "caveat": "Macro-averaged P/R/F1 need not obey the harmonic relation. Flags warrant clarification, not replacement of reported observations.",
            "rows": findings,
        },
    )
    print(f"Checked {len(findings)} reported P/R/F1 rows")


if __name__ == "__main__":
    main()
