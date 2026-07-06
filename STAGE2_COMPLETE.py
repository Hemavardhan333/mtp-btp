#!/usr/bin/env python3
"""
STAGE2_COMPLETE.py  —  Context Localizer + Cascade Re-scoring (per proposal doc).

For each detected ambiguity t in A(P):
  1. RETRIEVE the most relevant repository snippet from the source file
        r*_t = argmax_r sim(enc(r), enc(t, P))          [dense retrieval + FAISS]
  2. RESOLVE if similar enough:
        Resolve(t,P,R) = 1 if sim >= theta  else 0
     - Resolve=1  -> auto-patch silently (no question)  -> A_res
     - Resolve=0  -> escalate to Stage 3                 -> A_unres
  3. CASCADE RE-SCORE remaining types after each resolution:
        s_t'^(i) = s_t'^(i-1) * (1 - I(t_i -> t'))
     drop any type whose score falls below tau (cascade collapse)

EVALUATION (measurable now, without running code):
  Retrieval quality vs oracle_context — does the retrieved snippet contain the
  classes/APIs the reference solution actually needed? Reports Recall@snippet.

Inputs:
  --data   task_level_dataset_FINAL.xlsx   (has prompt, sample_id)
  --coder  CoderEval4Python.json           (has file_content, oracle_context)
  --detector detector.pt                   (from Stage 1)
  --influence influence_matrix.json        (from Stage 1)

Colab:
  !pip install transformers torch faiss-cpu scikit-learn pandas openpyxl numpy
  !python STAGE2_COMPLETE.py --data task_level_dataset_FINAL.xlsx \
        --coder CoderEval4Python.json --detector detector.pt \
        --influence influence_matrix.json
"""
import argparse, json, re, ast, numpy as np, pandas as pd, torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

TYPES = ["func", "param", "dep", "ctx"]
DET_MODEL = "microsoft/graphcodebert-base"
EMB_MODEL = "microsoft/unixcoder-base"   # code-specialized embedder for retrieval


# ---------- Stage 1 detector (to get scores + do cascade) ----------
class Detector(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.enc = AutoModel.from_pretrained(DET_MODEL)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(self.enc.config.hidden_size, n)
    def forward(self, **x):
        return self.head(self.drop(self.enc(**x).last_hidden_state[:, 0]))


# ---------- the code embedder for retrieval ----------
class Embedder:
    def __init__(self, dev):
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(EMB_MODEL)
        self.model = AutoModel.from_pretrained(EMB_MODEL).to(dev).eval()
    @torch.no_grad()
    def embed(self, texts):
        if isinstance(texts, str): texts = [texts]
        vecs = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            e = self.tok(batch, truncation=True, padding=True, max_length=256,
                         return_tensors="pt").to(self.dev)
            out = self.model(**e).last_hidden_state[:, 0]   # CLS
            out = torch.nn.functional.normalize(out, dim=1)  # for cosine via dot
            vecs.append(out.cpu().numpy())
        return np.vstack(vecs)


def chunk_file(file_content, solution_code="", size=6):
    """Split a source file into overlapping line-window snippets to retrieve from.
    IMPORTANT: removes the reference solution itself, so retrieval finds
    SURROUNDING repository context, not the answer (prevents data leakage)."""
    text = str(file_content)
    sol = str(solution_code).strip()
    if sol and len(sol) > 20:
        sol_lines = set(l.strip() for l in sol.splitlines() if len(l.strip()) > 10)
        kept = [l for l in text.splitlines() if l.strip() not in sol_lines]
        text = "\n".join(kept)
    lines = text.splitlines()
    chunks = []
    for i in range(0, max(1, len(lines)), size // 2 or 1):
        chunk = "\n".join(lines[i:i+size]).strip()
        if len(chunk) > 15:
            chunks.append(chunk)
    return chunks or [text[:400]]


def parse_oracle(oracle_context):
    """Pull the set of classes+apis the reference solution needed."""
    try:
        # oracle_context is a JSON-ish string with apis/classes/vars lists
        s = oracle_context.replace("'", '"')
        d = json.loads(s)
        items = set()
        for key in ("apis", "classes"):
            v = d.get(key, "[]")
            if isinstance(v, str):
                items |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", v))
            elif isinstance(v, list):
                items |= set(v)
        return {x for x in items if len(x) > 1}
    except Exception:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(oracle_context)))


# ---------- type-specific query text (what we search FOR) ----------
QUERY = {
    "func":  "function behavior specification what the function should do",
    "param": "parameter type format meaning of input arguments",
    "dep":   "definition of required class function or module dependency import",
    "ctx":   "repository context helper utilities surrounding code definitions",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="task_level_dataset_FINAL.xlsx")
    ap.add_argument("--coder", default="CoderEval4Python.json")
    ap.add_argument("--detector", default="detector.pt")
    ap.add_argument("--influence", default="influence_matrix.json")
    ap.add_argument("--theta", type=float, default=0.70, help="resolution threshold (stricter = fewer, more confident auto-resolutions)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    import faiss

    # load everything
    df = pd.read_excel(a.data)
    df = df[df["prompt"].astype(str).str.strip().ne("")].reset_index(drop=True)
    ce = {r["_id"]: r for r in json.load(open(a.coder))["RECORDS"]}
    I = np.array(json.load(open(a.influence))["I"])

    tok = AutoTokenizer.from_pretrained(DET_MODEL)
    det = Detector().to(dev)
    ck = torch.load(a.detector, map_location=dev)
    det.load_state_dict(ck["model"]); det.eval()
    thr = ck.get("thresholds", {t: 0.5 for t in TYPES})

    emb = Embedder(dev)

    def scores(prompt):
        e = tok(prompt, truncation=True, padding="max_length", max_length=256,
                return_tensors="pt").to(dev)
        with torch.no_grad():
            return torch.sigmoid(det(**e)).cpu().numpy()[0]

    # ---- run Stage 2 over all annotated (positive) tasks ----
    rows = []
    resolved_total, escalated_total = 0, 0
    retr_hit, retr_total = 0, 0
    all_sims = []  # every retrieval similarity, for the threshold sweep

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        if sid not in ce:   # skip negatives / non-CoderEval
            continue
        rec_ce = ce[sid]
        prompt = str(row["prompt"])
        s = scores(prompt)
        A = [t for t in TYPES if s[TYPES.index(t)] >= thr[t]]
        if not A:
            continue

        # build FAISS index over this task's source file
        snippets = chunk_file(rec_ce.get("file_content", ""), rec_ce.get("code", ""))
        S = emb.embed(snippets)                # (n, d) normalized
        index = faiss.IndexFlatIP(S.shape[1])  # inner product = cosine (normalized)
        index.add(S.astype(np.float32))

        oracle = parse_oracle(rec_ce.get("oracle_context", ""))

        # score vector we mutate during cascade
        s_live = {t: float(s[TYPES.index(t)]) for t in TYPES}
        A_res, A_unres = [], []

        # resolve in the order the detector is most confident (proxy for rank)
        for t in sorted(A, key=lambda x: s_live[x], reverse=True):
            if s_live[t] < thr[t]:
                continue  # collapsed by an earlier cascade -> implicitly resolved
            q = emb.embed(QUERY[t] + " " + prompt[:150])
            D, Idx = index.search(q.astype(np.float32), 1)
            sim = float(D[0][0]); best = snippets[int(Idx[0][0])]
            all_sims.append(sim)

            # retrieval-quality check vs oracle: does snippet contain needed names?
            snip_names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", best))
            if oracle:
                retr_total += 1
                if snip_names & oracle:
                    retr_hit += 1

            if sim >= a.theta:
                A_res.append((t, round(sim, 3)))
                resolved_total += 1
                # cascade re-score the others
                i = TYPES.index(t)
                for tp in TYPES:
                    if tp != t:
                        s_live[tp] = s_live[tp] * (1 - I[i][TYPES.index(tp)])
            else:
                A_unres.append((t, round(sim, 3)))
                escalated_total += 1

        rows.append({
            "sample_id": sid, "name": rec_ce.get("name", ""),
            "detected": ", ".join(A),
            "auto_resolved": ", ".join(f"{t}({s})" for t, s in A_res) or "(none)",
            "escalated_to_Q": ", ".join(f"{t}({s})" for t, s in A_unres) or "(none)",
        })

    out = pd.DataFrame(rows)
    out.to_excel("stage2_results.xlsx", index=False)

    print("\n========== STAGE 2 RESULTS ==========")
    print(f"Tasks processed: {len(out)}")
    print(f"Ambiguities auto-resolved from repo (silent patch): {resolved_total}")
    print(f"Ambiguities escalated to questions (Stage 3):        {escalated_total}")
    tot = resolved_total + escalated_total
    if tot:
        print(f"Auto-resolution rate: {resolved_total/tot:.1%} "
              f"(these need NO user question — the cost saving)")
    if retr_total:
        print(f"\nRetrieval quality vs oracle_context:")
        print(f"  Recall@1 snippet: {retr_hit/retr_total:.1%} "
              f"(retrieved snippet contained a needed class/API)")
    print(f"\nθ (resolution threshold) = {a.theta}")

    # threshold sweep: auto-resolution rate at different θ (pick a defensible one)
    if all_sims:
        sims = np.array(all_sims)
        print("\nThreshold sweep (auto-resolution rate vs θ):")
        for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            rate = (sims >= th).mean()
            print(f"  θ={th:.2f}  ->  {rate:.1%} auto-resolved, {1-rate:.1%} escalated")
        print(f"  (mean similarity {sims.mean():.3f}, median {np.median(sims):.3f})")
        print("  Pick θ where auto-resolution is believable (~30-60%), rest -> questions.")

    print("\nSaved: stage2_results.xlsx")
    print("STAGE 2 COMPLETE.")


if __name__ == "__main__":
    main()