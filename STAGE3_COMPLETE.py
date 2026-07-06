#!/usr/bin/env python3
"""
STAGE3_COMPLETE.py  —  Question Generator g_phi (per proposal doc).

For each UNRESOLVED ambiguity (escalated from Stage 2), generate ONE targeted
clarifying question using an LLM with a type-specific template M(t):
        Q_t = g_phi(P, t, M(t))

EVALUATION: BERTScore between the generated question and a "gold" question
derived from what the reference solution actually needed (oracle_context).
This measures whether the questions target the right missing information.

Uses GROQ (free, fast, OpenAI-compatible). Get a key at console.groq.com.

Inputs:
  --stage2  stage2_results.xlsx        (has escalated_to_Q per task)
  --data    task_level_dataset_FINAL.xlsx
  --coder   CoderEval4Python.json      (oracle_context for gold questions)

Colab:
  !pip install groq bert-score pandas openpyxl
  import os; os.environ['GROQ_API_KEY'] = 'your_key_here'
  !python STAGE3_COMPLETE.py --stage2 stage2_results.xlsx \
        --data task_level_dataset_FINAL.xlsx --coder CoderEval4Python.json
"""
import argparse, json, os, re, time, pandas as pd

TYPES = ["func", "param", "dep", "ctx"]

# type-specific templates M(t): how to ask about each ambiguity type
TEMPLATES = {
    "func":  "The prompt's intended behavior is unclear. Ask ONE concise question "
             "to clarify exactly what the function should do.",
    "param": "A parameter's type or format is undefined. Ask ONE concise question "
             "to clarify the parameter's expected type, format, or valid values.",
    "dep":   "A required class/function/module is not named. Ask ONE concise question "
             "to identify the specific dependency the implementation should use.",
    "ctx":   "Repository-level context is assumed but missing. Ask ONE concise question "
             "to obtain the missing contextual information needed to implement this.",
}


def make_question(client, model, prompt, t):
    """g_phi: generate one clarifying question for type t."""
    sys = ("You generate a SINGLE short clarifying question a developer would ask "
           "before implementing a function from an ambiguous spec. Output ONLY the "
           "question, no preamble.")
    user = f"{TEMPLATES[t]}\n\nSPEC:\n{prompt[:800]}\n\nYour one question:"
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user}],
                temperature=0.3, max_tokens=60)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 3:
                return f"[error: {e}]"
            time.sleep(2 ** attempt)  # backoff on rate limit


def gold_question(oracle_context, t):
    """Derive a reference 'ideal' question from what the solution needed."""
    try:
        d = json.loads(str(oracle_context).replace("'", '"'))
    except Exception:
        d = {}
    classes = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(d.get("classes", "")))
    apis = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(d.get("apis", "")))
    if t == "dep" and (classes or apis):
        names = ", ".join((classes + apis)[:3])
        return f"Which specific class or function should be used to implement this — for example {names}?"
    if t == "param":
        return "What is the exact type and expected format of each input parameter?"
    if t == "func":
        return "What is the precise intended behavior and expected output of this function?"
    if t == "ctx":
        names = ", ".join((classes + apis)[:3])
        return f"What surrounding context or definitions are needed{(' such as ' + names) if names else ''}?"
    return "What key information is missing to implement this correctly?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", default="stage2_results.xlsx")
    ap.add_argument("--data", default="task_level_dataset_FINAL.xlsx")
    ap.add_argument("--coder", default="CoderEval4Python.json")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N tasks (for testing)")
    a = ap.parse_args()

    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    s2 = pd.read_excel(a.stage2)
    data = pd.read_excel(a.data).set_index("sample_id")
    ce = {r["_id"]: r for r in json.load(open(a.coder))["RECORDS"]}

    gen_qs, gold_qs, meta = [], [], []
    n = 0
    for _, row in s2.iterrows():
        esc = str(row.get("escalated_to_Q", ""))
        if esc.strip() in ("", "(none)"):
            continue
        sid = str(row["sample_id"])
        if sid not in data.index or sid not in ce:
            continue
        prompt = str(data.loc[sid, "prompt"])
        oracle = ce[sid].get("oracle_context", "")
        # parse the escalated types, e.g. "dep(0.4), ctx(0.5)"
        types = [tt for tt in TYPES if re.search(rf"\b{tt}\b", esc)]
        for t in types:
            q = make_question(client, a.model, prompt, t)
            g = gold_question(oracle, t)
            gen_qs.append(q); gold_qs.append(g)
            meta.append({"sample_id": sid, "type": t, "generated_q": q, "gold_q": g})
        n += 1
        if a.limit and n >= a.limit:
            break
        if n % 10 == 0:
            print(f"  ...{n} tasks, {len(gen_qs)} questions generated")

    print(f"\nGenerated {len(gen_qs)} questions across {n} tasks.")

    # ---- BERTScore evaluation ----
    print("Computing BERTScore (generated vs gold questions)...")
    from bert_score import score as bertscore
    P, R, F1 = bertscore(gen_qs, gold_qs, lang="en", verbose=False)
    f1m = F1.mean().item()

    dfm = pd.DataFrame(meta)
    dfm["bertscore_f1"] = F1.numpy()
    dfm.to_excel("stage3_results.xlsx", index=False)

    print("\n========== STAGE 3 RESULTS ==========")
    print(f"Questions generated: {len(gen_qs)}")
    print(f"BERTScore F1 (vs gold questions): {f1m:.3f}")
    print("Per-type average BERTScore:")
    for t in TYPES:
        sub = dfm[dfm["type"] == t]["bertscore_f1"]
        if len(sub):
            print(f"  {t}: {sub.mean():.3f}  (n={len(sub)})")
    print("\nSaved: stage3_results.xlsx")
    print("STAGE 3 COMPLETE.")


if __name__ == "__main__":
    main()
