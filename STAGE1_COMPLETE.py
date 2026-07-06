#!/usr/bin/env python3
"""
STAGE1_COMPLETE.py  —  Runs ALL of Stage 1 end to end in one shot.

Pipeline (per proposal doc):
  1A  train detector f_theta -> scores s_t, detected set A(P), per-class P/R/F1
  I   learn influence matrix I(t->t') from data
  1B  ambiguity ranking rank(t) = alpha*s_t + beta*sum I(t->t')
  1C  top-k selection, with coverage-vs-cost evaluation for k=1,2,3
Saves: detector.pt, influence_matrix.json, stage1_results.json

Colab:
  !pip install transformers torch scikit-learn pandas openpyxl iterative-stratification
  !python STAGE1_COMPLETE.py --data task_level_dataset.xlsx --seeds 5

Notes:
  * labels are HUMAN-ANNOTATED (relabeled/corrected); columns unchanged.
  * small-data hardened: capped pos_weight, frozen lower layers, early stop, multi-seed.
"""
import argparse, json, logging, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, f1_score
logging.getLogger("transformers").setLevel(logging.ERROR)

TYPES = ["func", "param", "dep", "ctx"]
MODEL = "microsoft/graphcodebert-base"


# ---------------- model ----------------
class DS(Dataset):
    def __init__(self, texts, labels, tok, maxlen=256):
        self.t, self.y, self.tok, self.m = texts, labels, tok, maxlen
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, padding="max_length",
                     max_length=self.m, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.y[i], dtype=torch.float)


class Detector(nn.Module):
    def __init__(self, n=4, freeze=6):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        for p in self.enc.embeddings.parameters(): p.requires_grad = False
        for i, l in enumerate(self.enc.encoder.layer):
            if i < freeze:
                for p in l.parameters(): p.requires_grad = False
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(self.enc.config.hidden_size, n)
    def forward(self, **x):
        return self.head(self.drop(self.enc(**x).last_hidden_state[:, 0]))


def split(y, seed):
    idx = np.arange(len(y))
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        tr, te = next(MultilabelStratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y))
        tr2, va = next(MultilabelStratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]))
        return tr[tr2], tr[va], te
    except Exception:
        rng = np.random.RandomState(seed); rng.shuffle(idx); n = len(idx)
        return idx[:int(.7*n)], idx[int(.7*n):int(.8*n)], idx[int(.8*n):]


def train_seed(texts, Y, Yb, tok, dev, seed, epochs, bs, lr):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = split(Yb, seed)
    mk = lambda idx, sh: DataLoader(DS([texts[i] for i in idx], Y[idx], tok), batch_size=bs, shuffle=sh)
    dtr, dva, dte = mk(tr, True), mk(va, False), mk(te, False)
    model = Detector().to(dev)
    pos = Yb[tr].sum(0); neg = len(tr) - pos
    pw = torch.tensor(np.clip(neg/np.maximum(pos,1),1,3), dtype=torch.float).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    def ep(dl, tr_):
        model.train() if tr_ else model.eval(); P,T=[],[]
        with torch.set_grad_enabled(tr_):
            for x,y in dl:
                x={k:v.to(dev) for k,v in x.items()}; y=y.to(dev)
                lo=model(**x); loss=crit(lo,y)
                if tr_: opt.zero_grad(); loss.backward(); opt.step()
                P.append(torch.sigmoid(lo).detach().cpu().numpy()); T.append(y.cpu().numpy())
        return np.vstack(P), np.vstack(T)
    best,bs_state,bad=-1,None,0
    for e in range(epochs):
        ep(dtr,True); vp,vt=ep(dva,False)
        f=f1_score((vt>=.5).astype(int),(vp>=.5).astype(int),average="macro",zero_division=0)
        if f>best: best=f; bs_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=4: break
    if bs_state: model.load_state_dict(bs_state)
    vp,vt=ep(dva,False)
    thr=[]
    for j in range(4):
        bt,bf=0.5,-1
        for c in np.arange(0.2,0.75,0.05):
            ff=f1_score((vt[:,j]>=.5).astype(int),(vp[:,j]>=c).astype(int),zero_division=0)
            if ff>bf: bf,bt=ff,c
        thr.append(float(bt))
    tp,tt=ep(dte,False)
    pred=np.zeros_like(tp)
    for j in range(4): pred[:,j]=(tp[:,j]>=thr[j]).astype(int)
    per={t:precision_recall_fscore_support((tt[:,j]>=.5).astype(int),pred[:,j],
         average="binary",zero_division=0)[:3] for j,t in enumerate(TYPES)}
    macro=f1_score((tt>=.5).astype(int),pred,average="macro",zero_division=0)
    micro=f1_score((tt>=.5).astype(int),pred,average="micro",zero_division=0)
    return model, thr, per, macro, micro


def learn_influence(df):
    H=df[[t+"_hard" for t in TYPES]].values.astype(int)
    I=np.zeros((4,4))
    for i in range(4):
        ti=H[:,i]==1; n=ti.sum()
        for j in range(4):
            if i!=j and n>0: I[i,j]=round(((H[ti,j]==1).sum())/n,3)
    return I


def stage1bc_eval(df, model, thr, I, dev, tok, alpha=1.0, beta=0.5):
    """Run 1B ranking + 1C top-k over the dataset, report coverage vs cost."""
    model.eval()
    def score(p):
        e=tok(p,truncation=True,padding="max_length",max_length=256,return_tensors="pt").to(dev)
        with torch.no_grad(): s=torch.sigmoid(model(**e)).cpu().numpy()[0]
        return s
    Y=df[[t+"_hard" for t in TYPES]].values.astype(int)
    res={}
    for k in [1,2,3]:
        cov_num,cov_den,nq=0,0,0
        for idx,row in df.iterrows():
            true=[j for j in range(4) if Y[idx][j]==1]
            if not true: continue
            s=score(str(row["prompt"]))
            A=[j for j in range(4) if s[j]>=thr[j]]
            if not A: continue
            # rank
            rank={}
            for t in A:
                infl=sum(I[t][tp] for tp in A if tp!=t)
                rank[t]=alpha*s[t]+beta*infl
            R=sorted(A,key=lambda t:rank[t],reverse=True)
            Ak=set(R[:k])
            cov_num+=len(Ak & set(true)); cov_den+=len(true); nq+=len(Ak)
        res[k]={"coverage":round(cov_num/max(1,cov_den),3),
                "avg_questions":round(nq/max(1,len(df)),2)}
    return res


def export_scores(df, model, thr, I, dev, tok, alpha=1.0, beta=0.5, k=2,
                  out="prompt_scores.xlsx"):
    """
    Run the full Stage-1 pipeline on every prompt and export a readable table:
    per-type sigmoid score, detected flag, rank value, rank order, and top-k pick.
    This is the 'detector in action' artifact (for analysis + showing the TA),
    NOT a change to the training labels.
    """
    import pandas as pd
    model.eval()
    def score(p):
        e=tok(p,truncation=True,padding="max_length",max_length=256,return_tensors="pt").to(dev)
        with torch.no_grad(): s=torch.sigmoid(model(**e)).cpu().numpy()[0]
        return s

    rows=[]
    for idx,row in df.iterrows():
        p=str(row["prompt"])
        s=score(p)
        A=[TYPES[j] for j in range(4) if s[j]>=thr[j]]      # detected set A(P)
        # 1B ranking
        rankval={}
        for t in A:
            i=TYPES.index(t)
            infl=sum(I[i][TYPES.index(tp)] for tp in A if tp!=t)
            rankval[t]=round(alpha*float(s[TYPES.index(t)])+beta*infl,3)
        R=sorted(A,key=lambda t:rankval[t],reverse=True)     # ordered list
        Ak=R[:k]                                             # 1C top-k
        rec={
            "sample_id":row.get("sample_id",""),
            "name":row.get("name",""),
            "prompt":p[:300],
        }
        # per-type predicted probability (the sigmoid scores)
        for j,t in enumerate(TYPES):
            rec[f"score_{t}"]=round(float(s[j]),3)
        # detected flags
        for t in TYPES:
            rec[f"detected_{t}"]=int(t in A)
        # rank value per detected type (blank if not detected)
        for t in TYPES:
            rec[f"rank_{t}"]=rankval.get(t,"")
        rec["ranked_order"]=" > ".join(R) if R else "(none)"
        rec[f"top{k}_selected"]=", ".join(Ak) if Ak else "(none)"
        # ground-truth labels for comparison (from dataset)
        for t in TYPES:
            rec[f"true_{t}"]=int(row.get(f"{t}_hard",0))
        rows.append(rec)

    out_df=pd.DataFrame(rows)
    out_df.to_excel(out,index=False)
    print(f"  exported per-prompt scores -> {out}  ({len(out_df)} rows)")
    return out_df


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="task_level_dataset.xlsx")
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-5)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device:",dev)

    df=pd.read_excel(a.data)
    df=df[df["prompt"].astype(str).str.strip().ne("")].reset_index(drop=True)
    texts=df["prompt"].astype(str).tolist()
    Y=df[[t+"_hard" for t in TYPES]].values.astype(float)
    Yb=(Y>=.5).astype(int)
    tok=AutoTokenizer.from_pretrained(MODEL)

    # ---- Stage 1A: train, multi-seed ----
    print("\n========== STAGE 1A: Detector ==========")
    per_f1={t:[] for t in TYPES}; macros=[]; micros=[]
    best_model=best_thr=None; best_macro=-1
    for s in range(a.seeds):
        m,thr,per,ma,mi=train_seed(texts,Y,Yb,tok,dev,42+s,a.epochs,a.bs,a.lr)
        macros.append(ma); micros.append(mi)
        for t in TYPES: per_f1[t].append(per[t][2])
        print(f"  seed {42+s}: macro-F1 {ma:.3f} micro-F1 {mi:.3f}")
        if ma>best_macro: best_macro,best_model,best_thr=ma,m,thr
    print("\n  --- 1A results (mean +/- std, HUMAN-ANNOTATED labels) ---")
    for t in TYPES:
        ar=np.array(per_f1[t]); print(f"    {t:6s} F1 {ar.mean():.3f} +/- {ar.std():.3f}")
    print(f"    macro-F1 {np.mean(macros):.3f} +/- {np.std(macros):.3f}")
    print(f"    micro-F1 {np.mean(micros):.3f} +/- {np.std(micros):.3f}")

    # save best detector
    torch.save({"model":best_model.state_dict(),"thresholds":{t:best_thr[i] for i,t in enumerate(TYPES)},
                "types":TYPES},"detector.pt")

    # ---- Influence matrix ----
    print("\n========== Influence Matrix I(t->t') ==========")
    I=learn_influence(df)
    print("        "+"  ".join(f"{t:>6}" for t in TYPES))
    for i,t in enumerate(TYPES):
        print(f"  {t:>5} "+"  ".join(f"{I[i,j]:6.3f}" for j in range(4)))
    json.dump({"types":TYPES,"I":I.tolist()},open("influence_matrix.json","w"),indent=2)

    # ---- Stage 1B + 1C ----
    print("\n========== STAGE 1B + 1C: Ranking & Top-k ==========")
    bc=stage1bc_eval(df,best_model,best_thr,I,dev,tok)
    for k,v in bc.items():
        print(f"  k={k}: ambiguity coverage {v['coverage']} | avg questions/prompt {v['avg_questions']}")
    print("  (Goal: high coverage at small k = the cost saving from ranking.)")

    # ---- Export per-prompt scores (the 'detector in action' artifact) ----
    print("\n========== Exporting per-prompt scores ==========")
    export_scores(df, best_model, best_thr, I, dev, tok, out="prompt_scores.xlsx")

    json.dump({"stage1A":{"macro":float(np.mean(macros)),"macro_std":float(np.std(macros)),
                          "per_type":{t:float(np.mean(per_f1[t])) for t in TYPES}},
               "influence":I.tolist(),"stage1BC":bc},
              open("stage1_results.json","w"),indent=2)
    print("\nSaved: detector.pt, influence_matrix.json, stage1_results.json, prompt_scores.xlsx")
    print("STAGE 1 COMPLETE.")


if __name__=="__main__":
    main()
