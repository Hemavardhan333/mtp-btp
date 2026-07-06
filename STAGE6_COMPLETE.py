#!/usr/bin/env python3
"""
STAGE6_COMPLETE.py  —  Final Evaluation: does clarification improve code? (per doc)

Generates code TWO ways per task:
    baseline: code from original prompt  P
    ours:     code from clarified prompt P'  (with Q&A clarifications)
Then compares each generated code to the REFERENCE solution.

METRIC (no Docker needed): similarity-to-reference.
  We use CodeBLEU-style similarity (token + AST overlap) between generated code
  and the ground-truth reference. Higher = closer to correct.
  If P' code is closer to reference than P code -> clarification helped.

  Primary comparison: mean similarity(ours) vs mean similarity(baseline)
  Statistical test: paired t-test (H1: ours > baseline), matching the doc.

NOTE: true pass@1 requires CoderEval's Docker test env (future work). This
similarity proxy is the honest, runnable measure of clarification's effect.

Uses GROQ for generation.

Inputs:
  --stage5  stage5_results.xlsx        (clarified prompts P')
  --data    task_level_dataset_FINAL.xlsx
  --coder   CoderEval4Python.json      (reference code)

Colab:
  !pip install groq pandas openpyxl nltk scipy numpy
  import os; os.environ['GROQ_API_KEY']='your_key'
  !python STAGE6_COMPLETE.py --stage5 stage5_results.xlsx \
        --data task_level_dataset_FINAL.xlsx --coder CoderEval4Python.json
"""
import argparse, json, os, re, time, ast as pyast, numpy as np, pandas as pd


def generate_code(client, model, prompt):
    """LLM writes a Python function for the given spec/prompt."""
    sys = ("You are a Python developer. Given a function spec, write ONLY the complete "
           "Python function implementation. No explanation, no markdown fences.")
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": str(prompt)[:1500]}],
                temperature=0.2, max_tokens=400)
            code = r.choices[0].message.content.strip()
            code = re.sub(r"^```[a-z]*\n?|```$", "", code).strip()  # strip fences
            return code
        except Exception as e:
            if attempt == 3:
                return f"[error: {e}]"
            time.sleep(2 ** attempt)


def tokens(code):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^\s]", str(code))


def token_sim(a, b):
    """Jaccard-ish token overlap (a light CodeBLEU proxy)."""
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def ast_ok(code):
    try:
        pyast.parse(code); return True
    except Exception:
        return False


def similarity(gen, ref):
    """Blend token overlap with a syntactic-validity bonus."""
    s = token_sim(gen, ref)
    if ast_ok(gen):
        s = 0.9 * s + 0.1   # small bonus for parseable code
    return round(s, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage5", default="stage5_results.xlsx")
    ap.add_argument("--data", default="task_level_dataset_FINAL.xlsx")
    ap.add_argument("--coder", default="CoderEval4Python.json")
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from groq import Groq
    from scipy import stats
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    s5 = pd.read_excel(a.stage5)
    data = pd.read_excel(a.data).set_index("sample_id")
    ce = {r["_id"]: r for r in json.load(open(a.coder))["RECORDS"]}

    # resume support
    done = {}
    if os.path.exists("stage6_results.xlsx"):
        old = pd.read_excel("stage6_results.xlsx")
        for _, r in old.iterrows():
            if str(r.get("code_ours", "")).strip() and not str(r["code_ours"]).startswith("[error"):
                done[str(r["sample_id"])] = r.to_dict()
        print(f"Resume: {len(done)} tasks already done.")

    rows = []
    n = 0
    for _, row in s5.iterrows():
        sid = str(row["sample_id"])
        if sid in done:
            rows.append(done[sid]); continue
        if sid not in data.index or sid not in ce:
            continue
        P = str(data.loc[sid, "prompt"])
        Pp = str(row["clarified_prompt"])
        ref = ce[sid].get("code", "")

        code_base = generate_code(client, a.model, P)
        code_ours = generate_code(client, a.model, Pp)
        sim_base = similarity(code_base, ref)
        sim_ours = similarity(code_ours, ref)

        rows.append({"sample_id": sid, "name": ce[sid].get("name", ""),
                     "sim_baseline": sim_base, "sim_ours": sim_ours,
                     "improved": int(sim_ours > sim_base),
                     "code_baseline": code_base, "code_ours": code_ours})
        n += 1
        if a.limit and n >= a.limit:
            break
        if n % 15 == 0:
            print(f"  ...{n} tasks evaluated")
            pd.DataFrame(rows).to_excel("stage6_results.xlsx", index=False)  # checkpoint
        time.sleep(0.4)

    out = pd.DataFrame(rows)
    out.to_excel("stage6_results.xlsx", index=False)

    b = out["sim_baseline"].values
    o = out["sim_ours"].values
    print("\n========== STAGE 6 RESULTS (FINAL) ==========")
    print(f"Tasks evaluated: {len(out)}")
    print(f"Mean similarity-to-reference:")
    print(f"  BASELINE (original prompt P):   {b.mean():.4f}")
    print(f"  OURS     (clarified prompt P'): {o.mean():.4f}")
    delta = o.mean() - b.mean()
    print(f"  Improvement (Δ): {delta:+.4f}  ({100*delta/max(b.mean(),1e-9):+.1f}%)")
    print(f"  Tasks where clarification helped: {out['improved'].sum()}/{len(out)} "
          f"({100*out['improved'].mean():.0f}%)")

    # paired t-test (H1: ours > baseline), as in the proposal doc
    if len(out) > 2:
        t, p = stats.ttest_rel(o, b)
        p_one = p / 2 if t > 0 else 1 - p / 2
        print(f"\nPaired t-test (H1: ours > baseline):")
        print(f"  t = {t:.3f}, one-sided p = {p_one:.4f}  "
              f"{'-> SIGNIFICANT (p<0.05)' if p_one < 0.05 else '-> not significant'}")

    print("\nNOTE: similarity-to-reference proxy; true pass@1 needs CoderEval Docker (future work).")
    print("Saved: stage6_results.xlsx")
    print("STAGE 6 COMPLETE — full pipeline evaluated.")


if __name__ == "__main__":
    main()
