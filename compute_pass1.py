#!/usr/bin/env python3
"""
compute_pass1.py

Reads CoderEval's Docker evaluation outputs (the *_out.jsonl files) for your
baseline and clarified predictions, and computes the FINAL publishable metric:

    pass@1(P)   = baseline
    pass@1(P')  = ours
    Δpass@1     = pass@1(P') − pass@1(P)          [your doc's secondary metric]
    ΔF_ext      = external-failure reduction       [your doc's PRIMARY metric]
    paired t-test (H1: ours > baseline) at α=0.05  [your doc's stat test]

CoderEval's *_out.jsonl adds a pass/fail field per task. This script auto-detects
common field names (is_pass / passed / correct / result). If yours differs, pass
--passfield <name>.

Usage:
    python compute_pass1.py --baseline pred_baseline.jsonl_out.jsonl \
                            --ours     pred_ours.jsonl_out.jsonl
"""
import argparse, json, numpy as np


PASS_FIELDS = ["is_pass", "passed", "correct", "pass", "result", "test_pass", "pass@1"]


def load_pass(path, passfield=None):
    """Return dict: task_id -> 1/0 pass."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = str(d.get("_id", d.get("id", "")))
            # find the pass field
            val = None
            if passfield and passfield in d:
                val = d[passfield]
            else:
                for k in PASS_FIELDS:
                    if k in d:
                        val = d[k]; break
            # normalize to 1/0
            if isinstance(val, bool):
                p = int(val)
            elif isinstance(val, (int, float)):
                p = int(val > 0)
            elif isinstance(val, str):
                p = int(val.lower() in ("true", "pass", "passed", "1", "yes"))
            elif isinstance(val, list):
                # some formats: list of per-candidate results; pass@1 = first
                p = int(bool(val[0])) if val else 0
            else:
                p = 0
            out[sid] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--passfield", default=None,
                    help="name of the pass/fail field if auto-detect fails")
    a = ap.parse_args()

    from scipy import stats

    base = load_pass(a.baseline, a.passfield)
    ours = load_pass(a.ours, a.passfield)

    # align on common task ids
    common = sorted(set(base) & set(ours))
    if not common:
        print("ERROR: no common task ids. Check the files / --passfield name.")
        print("Baseline sample keys:", list(base)[:3])
        return

    b = np.array([base[s] for s in common])
    o = np.array([ours[s] for s in common])

    p1_base = b.mean()
    p1_ours = o.mean()
    dpass = p1_ours - p1_base

    # External Failure Reduction (doc's primary metric):
    # F_ext = failure rate. Reduction = (F(P) - F(P')) / F(P)
    f_base = 1 - p1_base
    f_ours = 1 - p1_ours
    dF_ext = (f_base - f_ours) / f_base if f_base > 0 else 0.0

    print("========== FINAL pass@1 RESULTS (CoderEval Docker) ==========")
    print(f"Tasks evaluated (common): {len(common)}")
    print(f"pass@1  BASELINE (P):   {p1_base:.4f}  ({b.sum()}/{len(b)})")
    print(f"pass@1  OURS     (P'):  {p1_ours:.4f}  ({o.sum()}/{len(o)})")
    print(f"Δpass@1:                {dpass:+.4f}  ({100*dpass:+.1f} pts)")
    print(f"ΔF_ext (failure reduction): {dF_ext:+.1%}")

    if len(common) > 2 and not (b == o).all():
        t, p = stats.ttest_rel(o, b)
        p_one = p / 2 if t > 0 else 1 - p / 2
        print(f"\nPaired t-test (H1: ours > baseline):")
        print(f"  t = {t:.3f}, one-sided p = {p_one:.4f}  "
              f"{'-> SIGNIFICANT (p<0.05)' if p_one < 0.05 else '-> not significant'}")
    else:
        print("\n(t-test skipped: identical or too few samples)")

    print("\nThis is the REAL pass@1 metric your architecture specifies.")


if __name__ == "__main__":
    main()
