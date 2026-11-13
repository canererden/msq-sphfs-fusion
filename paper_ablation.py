"""Ablations under the admissible setting (Table 7) and fold-averaged bootstrap intervals for the
reject-option comparison. Usage: python paper_ablation.py outputs_v2_nostain"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from fusion import load, to_objects, sphfs_score
from paper_analysis import acc, auc, aurc, sel_acc, MODELS
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs_v2_nostain"
adm = lambda d, t: ((d["mu"].max(-1) ** t + d["io"].max(-1) ** t + d["nu"].max(-1) ** t) <= 1 + 1e-9).all()
res = {}
for level in ("image", "patient"):
    r = {}
    for f in range(5):
        va, te = load(OUT, MODELS, f, "val"), load(OUT, MODELS, f, "test"); mags = sorted(set(te.mag))
        rel = np.array([balanced_accuracy_score(va.label, va[f"p{k}"] > 0.5) for k in range(3)]); kappa = rel / rel.sum(); b = int(rel.argmax())
        yv, dv, _, _ = to_objects(va, 3, level, mags); yt, dt, _, _ = to_objects(te, 3, level, mags)
        if level == "patient":
            wr = np.array([balanced_accuracy_score(va.label[va.mag == g], va.loc[va.mag == g, [f"p{k}" for k in range(3)]].mean(axis=1) > 0.5) for g in mags]); w = wr / wr.sum()
        else: w = np.ones(1)
        noab = lambda d: {"mu": d["mu"] + d["io"] / 2, "io": np.zeros_like(d["io"]), "nu": d["nu"] + d["io"] / 2}; one = lambda d: {c: d[c][..., b:b + 1] for c in d}
        for name, (tv, tt, kp) in {"full": (dv, dt, kappa), "equal": (dv, dt, np.full(3, 1 / 3)), "noab": (noab(dv), noab(dt), kappa), "single": (one(dv), one(dt), np.ones(1))}.items():
            t = next(t for t in range(1, 400) if adm(tv, t)); s = sphfs_score(tt, kp, w, t, t, t, 1); r.setdefault(name, []).append((t, acc(yt, s), auc(yt, s), aurc(yt, s)))
    res[level] = {k: {"t": [int(x[0]) for x in v], **{m: (float(np.mean([x[i] for x in v])), float(np.std([x[i] for x in v], ddof=1))) for i, m in ((1, "acc"), (2, "auc"), (3, "aurc"))}} for k, v in r.items()}
json.dump(res, open(f"{OUT}/ablation_admissible.json", "w"), indent=1)
# fold-averaged, patient-cluster bootstrap: performance-tuned rule minus mean probability (image level)
df = pd.read_csv(f"{OUT}/pooled_image.csv"); rng = np.random.default_rng(1)
stat = lambda d, m, fn: np.mean([fn(g.y.values, g[m].values) for _, g in d.groupby("fold")])
fns = {"aurc": aurc, "sel60": lambda y, s: sel_acc(y, s, 0.6), "sel50": lambda y, s: sel_acc(y, s, 0.5)}
by = {f: {p: g for p, g in d.groupby("group")} for f, d in df.groupby("fold")}; boot = {k: [] for k in fns}
for _ in range(1000):
    b = pd.concat([pp[k] for f, pp in by.items() for k in rng.choice(list(pp), len(pp))])
    if any(g.y.nunique() < 2 for _, g in b.groupby("fold")): continue
    for k, fn in fns.items(): boot[k].append(stat(b, "proposed", fn) - stat(b, "mean", fn))
ci = {k: [float(stat(df, "proposed", fn) - stat(df, "mean", fn))] + [float(x) for x in np.percentile(boot[k], [2.5, 97.5])] for k, fn in fns.items()}
json.dump(ci, open(f"{OUT}/bootstrap_fold_averaged.json", "w"), indent=1); print(res, ci)
