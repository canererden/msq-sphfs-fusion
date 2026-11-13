"""Fuse deep-model outputs with the m,s,q-SPHFS generalized hybrid weighted distance
and compare against standard ensembles.

Usage:
  python fusion.py --out outputs --models resnet50 densenet121 efficientnet_b0 --level patient
  python fusion.py --demo            (synthetic data, only checks that the code runs)

Mapping (theory -> data)
  membership / abstinence / non-membership : evidential belief(malignant) / uncertainty / belief(benign)
  hesitant elements                        : the K deep models
  probabilities kappa                      : model reliabilities (validation balanced accuracy, sum = 1)
  attributes x_1..x_n and weights w        : the magnifications (patient level) or one attribute (image level)
  ideal elements                           : "complete certainty" prototypes for malignant and benign
A case is assigned to the nearer prototype; |score-0.5| is the confidence used for the reject option.
m, s, q, lambda are tuned on the validation fold only.
"""
import argparse, itertools, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from msq_sphfs import hybrid_distance

GRID_P, GRID_L = (1, 2, 3, 4, 5), (1, 2, 3)

# ---------- data ----------
def load(out, models, fold, split):
    base = None
    for k, mname in enumerate(models):
        d = pd.read_csv(os.path.join(out, f"{mname}_fold{fold}_{split}.csv"))
        S = d.alpha_ben + d.alpha_mal
        d = d.assign(**{f"mu{k}": (d.alpha_mal - 1) / S, f"nu{k}": (d.alpha_ben - 1) / S,
                        f"io{k}": 2 / S, f"p{k}": d.alpha_mal / S}).drop(columns=["alpha_ben", "alpha_mal"])
        base = d if base is None else base.merge(d.drop(columns=["label", "patient", "mag"]), on="path", validate="1:1")
    return base

def to_objects(df, K, level, mags):
    """Return y, degrees dict of arrays (N, n, K), probs (N, K)."""
    cols = lambda c: [f"{c}{k}" for k in range(K)]
    if level == "image":
        deg = {c: df[cols(c)].values[:, None, :] for c in ("mu", "io", "nu")}
        return df.label.values, deg, df[cols("p")].values, df.mag.values
    g = df.groupby(["patient", "mag"])[cols("mu") + cols("io") + cols("nu") + cols("p")].mean()
    pats = sorted(df.patient.unique())
    full = g.reindex(pd.MultiIndex.from_product([pats, mags], names=["patient", "mag"]))
    if full.isna().any().any():   # patient without some magnification: use the patient's mean
        full = full.fillna(full.groupby(level=0).transform("mean"))
    N, n = len(pats), len(mags)
    deg = {c: full[cols(c)].values.reshape(N, n, K) for c in ("mu", "io", "nu")}
    y = df.groupby("patient").label.first().reindex(pats).values
    return y, deg, full[cols("p")].values.reshape(N, n, K).mean(1), None

# ---------- proposed method ----------
def sphfs_score(deg, kappa, w, m, s, q, lam):
    k = np.broadcast_to(kappa, deg["mu"].shape)
    a = (deg["mu"], k, deg["io"], k, deg["nu"], k)
    one, zero = np.ones_like(k), np.zeros_like(k)
    d_mal = hybrid_distance(a, (one, k, zero, k, zero, k), m, s, q, lam, w)
    d_ben = hybrid_distance(a, (zero, k, zero, k, one, k), m, s, q, lam, w)
    return d_ben / (d_ben + d_mal + 1e-12)

def tune(deg, y, kappa, w, tied=False):
    best = (-1, -1, None)
    for m, s, q, lam in itertools.product(GRID_P, GRID_P, GRID_P, GRID_L):
        if tied and not (m == s == q): continue
        sc = sphfs_score(deg, kappa, w, m, s, q, lam)
        key = (balanced_accuracy_score(y, sc > 0.5), roc_auc_score(y, sc) if len(set(y)) > 1 else 0)
        if key > best[:2]: best = (*key, (m, s, q, lam))
    return best[2]

# ---------- baselines ----------
def gompertz_rank(P):
    """Fuzzy rank-based fusion with the Gompertz function (Kundu et al., 2021), binary case."""
    conf = np.stack([1 - P, P], -1)                       # (N, K, 2)
    rank = 1 - np.exp(-np.exp(-2.0 * conf))
    final = rank.sum(1) * (1 - conf.mean(1))              # lower is better
    return final[:, 0] / (final.sum(1) + 1e-12)           # score for the malignant class

# ---------- metrics ----------
def ece(y, s, bins=10):
    conf = np.maximum(s, 1 - s); pred = (s > 0.5).astype(int); e = 0
    for lo in np.linspace(0.5, 1, bins + 1)[:-1]:
        mk = (conf >= lo) & (conf < lo + 0.5 / bins + 1e-9)
        if mk.any(): e += mk.mean() * abs((pred[mk] == y[mk]).mean() - conf[mk].mean())
    return e

def aurc(y, s):
    """Area under the risk-coverage curve (lower is better): error when the least
    confident cases are referred to the physician first."""
    order = np.argsort(-np.abs(s - 0.5)); err = ((s > 0.5).astype(int) != y)[order]
    return (np.cumsum(err) / np.arange(1, len(y) + 1)).mean()

def metrics(y, s):
    p = (s > 0.5).astype(int)
    return dict(acc=accuracy_score(y, p), bacc=balanced_accuracy_score(y, p), f1=f1_score(y, p, zero_division=0),
                auc=roc_auc_score(y, s) if len(set(y)) > 1 else np.nan, ece=ece(y, s), aurc=aurc(y, s))

# ---------- one fold ----------
def run_fold(out, models, fold, level):
    K = len(models); va, te = load(out, models, fold, "val"), load(out, models, fold, "test")
    mags = sorted(set(va.mag) | set(te.mag))
    rel = np.array([balanced_accuracy_score(va.label, va[f"p{k}"] > 0.5) for k in range(K)])
    kappa = rel / rel.sum(); uni = np.full(K, 1 / K)
    yv, dv, pv, _ = to_objects(va, K, level, mags); yt, dt, pt, mag_t = to_objects(te, K, level, mags)
    if level == "patient":   # magnification weights from validation images
        wr = np.array([balanced_accuracy_score(va.label[va.mag == g], va.loc[va.mag == g, [f"p{k}" for k in range(K)]].mean(axis=1) > 0.5) for g in mags])
        w = wr / wr.sum()
    else:
        w = np.ones(1)
    res = {}
    par = tune(dv, yv, kappa, w);                 res["Proposed m,s,q-SPHFS distance"] = (sphfs_score(dt, kappa, w, *par), par)
    par = tune(dv, yv, kappa, w, tied=True);      res["Ablation: m=s=q (T-spherical)"] = (sphfs_score(dt, kappa, w, *par), par)
    par = tune(dv, yv, uni, w);                   res["Ablation: equal probabilities"] = (sphfs_score(dt, uni, w, *par), par)
    noab = lambda d: {"mu": d["mu"] + d["io"] / 2, "io": np.zeros_like(d["io"]), "nu": d["nu"] + d["io"] / 2}
    par = tune(noab(dv), yv, kappa, w);           res["Ablation: no abstinence degree"] = (sphfs_score(noab(dt), kappa, w, *par), par)
    b = int(rel.argmax()); one = lambda d: {c: d[c][..., b:b + 1] for c in d}
    par = tune(one(dv), yv, np.ones(1), w);       res["Ablation: single model (no hesitancy)"] = (sphfs_score(one(dt), np.ones(1), w, *par), par)
    res["Best single model"] = (pt[:, b], None)
    res["Mean probability"] = (pt.mean(1), None)
    res["Weighted mean probability"] = (pt @ kappa, None)
    res["Majority vote"] = ((pt > 0.5).mean(1) * 0.98 + 0.01 * pt.mean(1) + 0.005, None)
    res["Stacking (logistic regression)"] = (LogisticRegression(max_iter=1000).fit(pv, yv).predict_proba(pt)[:, 1], None)
    res["Fuzzy rank (Gompertz)"] = (gompertz_rank(pt), None)
    rows = []
    for name, (s, par) in res.items():
        rows.append(dict(fold=fold, level=level, method=name, n=len(yt), params=par, **metrics(yt, s)))
    return rows

# ---------- decision matrix in the manuscript's notation ----------
def export_matrix(out, models, fold, n_show=5):
    """Write the m,s,q-SPHFS decision matrix of the test patients of one fold as
    <{mu|kappa,...},{iota|kappa,...},{nu|kappa,...}> per patient and magnification (cf. Table 1)."""
    K = len(models); va, te = load(out, models, fold, "val"), load(out, models, fold, "test")
    mags = sorted(set(te.mag))
    rel = np.array([balanced_accuracy_score(va.label, va[f"p{k}"] > 0.5) for k in range(K)]); kappa = rel / rel.sum()
    y, deg, _, _ = to_objects(te, K, "patient", mags)
    pats = sorted(te.patient.unique())
    fmt = lambda v: "{" + ", ".join(f"{x:.2f}|{k:.2f}" for x, k in zip(v, kappa)) + "}"
    rows = [dict(patient=pats[i], label="malignant" if y[i] else "benign",
                 **{f"{g}X": "<" + ", ".join(fmt(deg[c][i, j]) for c in ("mu", "io", "nu")) + ">" for j, g in enumerate(mags)})
            for i in range(len(pats))]
    df = pd.DataFrame(rows); path = os.path.join(out, f"decision_matrix_fold{fold}.csv"); df.to_csv(path, index=False)
    print(f"\nDecision matrix ({len(df)} patients) -> {path}; first {n_show}:")
    for r in rows[:n_show]:
        print(r["patient"], f"({r['label']})"); [print(f"   {g}X  {r[f'{g}X']}") for g in mags]

# ---------- demo data ----------
def make_demo(out, models, folds=5, seed=0):
    rng = np.random.default_rng(seed); os.makedirs(out, exist_ok=True)
    pats = pd.DataFrame({"patient": [f"p{i}" for i in range(80)], "label": rng.integers(0, 2, 80), "fold": np.arange(80) % folds})
    img = pats.loc[pats.index.repeat(40)].reset_index(drop=True)
    img["mag"] = np.tile([40, 100, 200, 400], len(img) // 4); img["path"] = [f"img{i}.png" for i in range(len(img))]
    hard = rng.random(len(img)) < 0.25
    for f in range(folds):
        for split, ff in (("val", (f + 1) % folds), ("test", f)):
            d = img[img.fold == ff]
            for j, mname in enumerate(models):
                sig = np.where(hard[d.index], 0.2, 1.5 + 0.3 * j) * (2 * d.label.values - 1) + rng.normal(0, 1.2, len(d))
                ev = np.where(hard[d.index], 2.0, 12.0) * rng.random(len(d)); pm = 1 / (1 + np.exp(-sig))
                d.assign(alpha_ben=1 + ev * (1 - pm), alpha_mal=1 + ev * pm)[["path", "label", "patient", "mag", "alpha_ben", "alpha_mal"]].to_csv(
                    os.path.join(out, f"{mname}_fold{f}_{split}.csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs"); ap.add_argument("--models", nargs="+", default=["resnet50", "densenet121", "efficientnet_b0"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--level", choices=["image", "patient", "both"], default="both")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        a.out = "demo_outputs"; make_demo(a.out, a.models)
    rows = []
    for lvl in (["image", "patient"] if a.level == "both" else [a.level]):
        for f in a.folds:
            rows += run_fold(a.out, a.models, f, lvl)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "fusion_results_by_fold.csv"), index=False)
    met = ["acc", "bacc", "f1", "auc", "ece", "aurc"]
    summ = df.groupby(["level", "method"], sort=False)[met].agg(["mean", "std"])
    summ.to_csv(os.path.join(a.out, "fusion_results_summary.csv"))
    pd.set_option("display.width", 220)
    print(df.groupby(["level", "method"], sort=False)[met].mean().round(4))
    print("\nTuned parameters (m, s, q, lambda) of the proposed method per fold:")
    print(df[df.method.str.startswith("Proposed")][["level", "fold", "params"]].to_string(index=False))
    export_matrix(a.out, a.models, a.folds[0])

if __name__ == "__main__":
    main()
