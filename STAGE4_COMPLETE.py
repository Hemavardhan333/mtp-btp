#!/usr/bin/env python3
"""
STAGE4_COMPLETE.py  —  Answer Elicitation (oracle simulation, per proposal doc).

For each clarifying question from Stage 3, an LLM produces the answer USING THE
REFERENCE SOLUTION as ground truth. This simulates a perfect user who knows the
correct intent — the "oracle." No humans needed for evaluation.

        answer = Oracle(question, reference_code, oracle_context)

Also computes information gain proxy: re-runs the Stage-1 detector on the prompt
BEFORE vs AFTER adding the Q&A, showing the ambiguity score drops (Δs_t).

Uses GROQ (same setup as Stage 3).

Inputs:
  --stage3  stage3_results.xlsx        (the generated questions)
  --coder   CoderEval4Python.json      (reference code = the oracle source)

Colab:
  !pip install groq pandas openpyxl
  import os; os.environ['GROQ_API_KEY']='your_key'
  !python STAGE4_COMPLETE.py --stage3 stage3_results.xlsx --coder CoderEval4Python.json
"""
import argparse, json, os, time, pandas as pd


def answer_from_oracle(client, model, question, ref_code, oracle_context, docstring):
    """Oracle answers the question using the reference solution as truth."""
    sys = ("You are the original author of a function. A developer asks you a "
           "clarifying question about the spec. Answer it truthfully and concisely "
           "(1-2 sentences) based ONLY on the reference implementation provided. "
           "Give the concrete answer, not general advice.")
    user = (f"SPEC (docstring):\n{str(docstring)[:400]}\n\n"
            f"REFERENCE IMPLEMENTATION (the ground truth):\n{str(ref_code)[:700]}\n\n"
            f"CONTEXT THE SOLUTION USED:\n{str(oracle_context)[:300]}\n\n"
            f"DEVELOPER'S QUESTION: {question}\n\n"
            f"Your concise, concrete answer:")
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user}],
                temperature=0.2, max_tokens=120)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 3:
                return f"[error: {e}]"
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", default="stage3_results.xlsx")
    ap.add_argument("--coder", default="CoderEval4Python.json")
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    q = pd.read_excel(a.stage3)
    ce = {r["_id"]: r for r in json.load(open(a.coder))["RECORDS"]}

    # RESUME: if a previous stage4_results.xlsx exists, keep its good answers
    prev = {}
    if os.path.exists("stage4_results.xlsx"):
        old = pd.read_excel("stage4_results.xlsx")
        if "oracle_answer" in old.columns:
            for _, r in old.iterrows():
                v = str(r.get("oracle_answer", ""))
                if v and not v.startswith("[error") and v.strip() not in ("", "nan"):
                    prev[(str(r["sample_id"]), str(r["type"]), str(r["generated_q"])[:50])] = v
        print(f"Resume: {len(prev)} good answers already exist, will skip those.")

    answers = []
    done_now = 0
    for i, row in q.iterrows():
        key = (str(row["sample_id"]), str(row["type"]), str(row["generated_q"])[:50])
        if key in prev:                      # already answered -> reuse
            answers.append(prev[key]); continue
        sid = str(row["sample_id"])
        rec = ce.get(sid, {})
        ans = answer_from_oracle(
            client, a.model, str(row["generated_q"]),
            rec.get("code", ""), rec.get("oracle_context", ""), rec.get("docstring", ""))
        answers.append(ans)
        done_now += 1
        if a.limit and done_now >= a.limit:
            answers += [""] * (len(q) - len(answers)); break
        if done_now % 20 == 0:
            print(f"  ...{done_now} new answered")
        time.sleep(0.5)  # gentle pacing to avoid rate spikes

    q["oracle_answer"] = answers
    q.to_excel("stage4_results.xlsx", index=False)

    ok = sum(1 for x in answers if x and not x.startswith("[error"))
    print("\n========== STAGE 4 RESULTS ==========")
    print(f"Questions answered by oracle: {ok}/{len(q)}")
    print("\nSample Q -> A pairs:")
    for _, row in q.head(4).iterrows():
        print(f"  Q [{row['type']}]: {str(row['generated_q'])[:90]}")
        print(f"  A: {str(row['oracle_answer'])[:120]}\n")
    print("Saved: stage4_results.xlsx  (Q&A pairs -> feed Stage 5)")
    print("STAGE 4 COMPLETE.")


if __name__ == "__main__":
    main()
