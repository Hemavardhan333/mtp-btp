#!/usr/bin/env python3
"""
make_coderEval_predictions.py

Converts your Stage 6 generated code into CoderEval's Docker input format so you
can compute TRUE pass@1.

CoderEval expects a JSONL file, one line per task:
    {"_id": "<task_id>", "generate_results": ["<code candidate>", ...]}

This produces TWO files:
    pred_baseline.jsonl  — code generated from the ORIGINAL prompt P
    pred_ours.jsonl      — code generated from the CLARIFIED prompt P'

You then feed EACH into CoderEval's Docker evaluation (see instructions printed
at the end) to get pass@1 for baseline vs ours.

Usage:
    python make_coderEval_predictions.py --stage6 stage6_results.xlsx
"""
import argparse, json, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage6", default="stage6_results.xlsx")
    ap.add_argument("--out_baseline", default="pred_baseline.jsonl")
    ap.add_argument("--out_ours", default="pred_ours.jsonl")
    a = ap.parse_args()

    df = pd.read_excel(a.stage6)
    need = {"sample_id", "code_baseline", "code_ours"}
    missing = need - set(df.columns)
    assert not missing, f"stage6 file missing columns: {missing}"

    nb = no = 0
    with open(a.out_baseline, "w") as fb, open(a.out_ours, "w") as fo:
        for _, r in df.iterrows():
            sid = str(r["sample_id"])
            cb = str(r["code_baseline"])
            co = str(r["code_ours"])
            # skip error rows
            if cb and not cb.startswith("[error"):
                fb.write(json.dumps({"_id": sid, "generate_results": [cb]}) + "\n")
                nb += 1
            if co and not co.startswith("[error"):
                fo.write(json.dumps({"_id": sid, "generate_results": [co]}) + "\n")
                no += 1

    print(f"Wrote {a.out_baseline}: {nb} tasks")
    print(f"Wrote {a.out_ours}:     {no} tasks")
    print("""
================= NEXT: run these in CoderEval's Docker =================

1. Download the CoderEval Docker image from the Drive link in their README:
   https://drive.google.com/drive/folders/1F8M7e25MgHZ3XJ4RSOGWindFSWC5QOvI

2. Import it (see their 'issue 4' for exact import steps), e.g.:
   docker load -i coderEval_image.tar
   docker run -it --name codereval <image_id> /bin/bash

3. Copy your two prediction files into the container:
   docker cp pred_baseline.jsonl codereval:/workspace/
   docker cp pred_ours.jsonl     codereval:/workspace/

4. Inside the container, run their evaluation program on EACH file
   (their script name is in the Docker's usage README — typically something
   that reads a *_Input.jsonl and writes *_Input.jsonl_out.jsonl with pass info).
   Run it once for pred_baseline.jsonl and once for pred_ours.jsonl.

5. Copy the two *_out.jsonl result files back out:
   docker cp codereval:/workspace/pred_baseline.jsonl_out.jsonl .
   docker cp codereval:/workspace/pred_ours.jsonl_out.jsonl .

6. Run compute_pass1.py (next script) on those two _out.jsonl files to get
   pass@1(baseline) vs pass@1(ours) + the paired t-test.
========================================================================
""")


if __name__ == "__main__":
    main()
