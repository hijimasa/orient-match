#!/usr/bin/env python3
"""Turn compare_methods CSV into the tables that go in the README.

Usage:  python3 benchmarks/summarize.py results.csv
"""
import csv
import sys
from collections import defaultdict

METHODS = ["orient_match", "bfncc_pyr", "bfncc", "fmt", "orb", "sift"]
LABEL = {
    "orient_match": "OrientMatch",
    "bfncc_pyr": "NCC coarse-to-fine",
    "bfncc": "NCC exhaustive 1°",
    "fmt": "Fourier-Mellin",
    "orb": "ORB + RANSAC",
    "sift": "SIFT + RANSAC",
}
POSITIVE = ["clean", "noise25", "noise50", "occlusion25", "occlusion50",
            "illum0.5", "jpeg20", "scale0.95", "scale1.05"]


def median(values):
    if not values:
        return float("nan")
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def auc(pos, neg):
    """Probability that a positive scores above a negative (rank statistic)."""
    if not pos or not neg:
        return float("nan")
    merged = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg = 0.5 * (i + 1 + j)          # 1-based average rank of the tie group
        rank_sum += sum(avg for k in range(i, j) if merged[k][1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main(path):
    rows = [r for r in csv.DictReader(open(path))
            if r.get("success") is not None and r.get("ms_frame") is not None]
    if not rows:
        sys.exit("no complete rows")

    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["condition"])].append(r)

    conditions = [c for c in POSITIVE if any(k[1] == c for k in by)]
    methods = [m for m in METHODS if any(k[0] == m for k in by)]

    print("### Success rate by condition (angle error <= 5 deg and position error <= 5 px)\n")
    header = "| Method | " + " | ".join(conditions) + " | overall |"
    print(header)
    print("|" + "---|" * (len(conditions) + 2))
    for m in methods:
        cells = []
        total_ok = total_n = 0
        for c in conditions:
            rs = by[(m, c)]
            ok = sum(int(r["success"]) for r in rs)
            cells.append(f"{100.0 * ok / len(rs):.1f}" if rs else "-")
            total_ok += ok
            total_n += len(rs)
        overall = f"**{100.0 * total_ok / total_n:.1f}**" if total_n else "-"
        print(f"| {LABEL[m]} | " + " | ".join(cells) + f" | {overall} |")

    print("\n### Accuracy where the method succeeded, and cost\n")
    print("| Method | median angle err [deg] | median position err [px] | successes | "
          "one-shot [ms] | per frame [ms] |")
    print("|---|---|---|---|---|---|")
    # Errors are taken over each method's own successes. A method that succeeds rarely is
    # therefore measured only on the cases it found easy, which is why the success-rate
    # table above has to be read together with this one.
    for m in methods:
        rs = [r for r in rows if r["method"] == m and r["condition"] in POSITIVE]
        ok = [r for r in rs if int(r["success"])]
        a = median([float(r["angle_err"]) for r in ok])
        p = median([float(r["pos_err"]) for r in ok])
        t = median([float(r["ms_total"]) for r in rs])
        f = median([float(r["ms_frame"]) for r in rs])
        print(f"| {LABEL[m]} | {a:.2f} | {p:.2f} | {len(ok)} | {t:.1f} | {f:.1f} |")
    print("\nThe true angles have a fractional part, so 0.25 degrees is the best a "
          "1 degree refinement grid can do: the correlation methods are at their "
          "quantization limit, not at their accuracy limit.")

    print("\n### Telling a present template from an absent one\n")
    print("| Method | AUC clean | AUC noise 25 | present median | absent median |")
    print("|---|---|---|---|---|")
    for m in methods:
        pos = [float(r["score"]) for r in by[(m, "clean")]]
        neg = [float(r["score"]) for r in by[(m, "negative")]]
        pos_n = [float(r["score"]) for r in by[(m, "noise25")]]
        neg_n = [float(r["score"]) for r in by[(m, "negative_noise25")]]
        print(f"| {LABEL[m]} | {auc(pos, neg):.3f} | {auc(pos_n, neg_n):.3f} | "
              f"{median(pos):.3f} | {median(neg):.3f} |")
    print("\nAUC is the probability that a present template outscores an absent one. "
          "Scores are not comparable across methods (the feature methods report an "
          "inlier count), only within a row.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
