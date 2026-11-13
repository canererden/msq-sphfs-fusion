"""Extra analyses for the manuscript (pooled out-of-fold predictions, reject option,
bootstrap comparisons, parameter sensitivity, calibration, example decision matrix).
Reads the per-image outputs written by train_extract.py; changes nothing upstream.
Usage: python paper_analysis.py --out outputs_v2_nostain
"""
import argparse, json, os, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from fusion import load, to_objects, sphfs_score, tune, gompertz_rank, GRID_P, GRID_L
from msq_sphfs import hybrid_distance

MODELS = ["resnet50", "densenet121", "efficientnet_b0"]
COV = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

def platt(sv, yv, st):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(sv.reshape(-1, 1), yv)
    return lr.predict_proba(st.reshape(-1, 1))[:, 1]

def collect(out, level):
    rows, extra = [], {"params": {}, "kappa": {}, "w": {}, "sens": [], "t_adm": {}, "viol_adm": {}, "viol_tuned": {}, "viol_canon": {}}
    adm = lambda d, a, b, c: (d["mu"].max(-1) ** a + d["io"].max(-1) ** b + d["nu"].max(-1) ** c) <= 1 + 1e-9   # Definition 3.1
    for f in range(5):
        K = 3; va, te = load(out, MODELS, f, "val"), load(out, MODELS, f, "test")
        mags = sorted(set(va.mag) | set(te.mag))
        rel = np.array([balanced_accuracy_score(va.label, va[f"p{k}"] > 0.5) for k in range(K)]); kappa = rel / rel.sum()
        yv, dv, pv, _ = to_objects(va, K, level, mags); yt, dt, pt, mag_t = to_objects(te, K, level, mags)
        if level == "patient":
            wr = np.array([balanced_accuracy_score(va.label[va.mag == g], va.loc[va.mag == g, [f"p{k}" for k in range(K)]].mean(axis=1) > 0.5) for g in mags]); w = wr / wr.sum()
            ids = sorted(te.patient.unique()); grp = ids
        else:
            w = np.ones(1); ids = te.path.values; grp = te.patient.values
        par = tune(dv, yv, kappa, w); b = int(rel.argmax())
        extra["params"][f] = par; extra["kappa"][f] = kappa.round(4).tolist(); extra["w"][f] = np.round(w, 4).tolist()
        k = np.broadcast_to(kappa, dt["mu"].shape); a = (dt["mu"], k, dt["io"], k, dt["nu"], k)
        one, zero = np.ones_like(k), np.zeros_like(k)
        d_mal = hybrid_distance(a, (one, k, zero, k, zero, k), *par, w); d_ben = hybrid_distance(a, (zero, k, zero, k, one, k), *par, w)
        t = next(t for t in range(1, 400) if adm(dv, t, t, t).all())      # smallest tied exponent admissible on the validation fold
        extra["t_adm"][f] = t; extra["viol_adm"][f] = float(1 - adm(dt, t, t, t).mean()); extra["viol_tuned"][f] = float(1 - adm(dt, *par[:3]).mean()); extra["viol_canon"][f] = float(1 - adm(dt, 1, 1, 1).mean())
        da_mal = hybrid_distance(a, (one, k, zero, k, zero, k), t, t, t, 1, w); da_ben = hybrid_distance(a, (zero, k, zero, k, one, k), t, t, t, 1, w)
        srt = lambda d: {c: -np.sort(-d[c], axis=-1) for c in d}          # elements ordered by value (Definition 3.2)
        s_prop_v = sphfs_score(dv, kappa, w, *par)
        df = pd.DataFrame(dict(fold=f, id=ids, group=grp, y=yt,
            adm=sphfs_score(dt, kappa, w, t, t, t, 1), adm_cal=platt(sphfs_score(dv, kappa, w, t, t, t, 1), yv, sphfs_score(dt, kappa, w, t, t, t, 1)), da_mal=da_mal, da_ben=da_ben,
            proposed=sphfs_score(dt, kappa, w, *par), default=sphfs_score(dt, kappa, w, 1, 1, 1, 1),
            sorted=sphfs_score(srt(dt), kappa, w, *tune(srt(dv), yv, kappa, w)),
            proposed_cal=platt(s_prop_v, yv, sphfs_score(dt, kappa, w, *par)),
            mean=pt.mean(1), wmean=pt @ kappa, best=pt[:, b],
            stack=LogisticRegression(max_iter=1000).fit(pv, yv).predict_proba(pt)[:, 1], gompertz=gompertz_rank(pt),
            d_mal=d_mal, d_ben=d_ben, p0=pt[:, 0], p1=pt[:, 1], p2=pt[:, 2], best_idx=b, iota=dt["io"].mean((1, 2)), disagree=dt["mu"].std(-1).mean(1)))
        if level == "image": df["mag"] = mag_t
        rows.append(df)
        for m, q in itertools.product(GRID_P, GRID_P):                  # sensitivity: fix s and lambda at the tuned values
            sc = sphfs_score(dt, kappa, w, m, par[1], q, par[3]); extra["sens"].append((f, m, q, ((sc > 0.5) == yt).mean(), balanced_accuracy_score(yt, sc > 0.5)))
    return pd.concat(rows, ignore_index=True), extra

def acc(y, s): return ((s > 0.5).astype(int) == y).mean()
def bacc(y, s): return balanced_accuracy_score(y, s > 0.5)
def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(np.asarray(s)); n1 = (y == 1).sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def sens(y, s): return ((s > 0.5) & (y == 1)).sum() / (y == 1).sum()
def spec(y, s): return ((s <= 0.5) & (y == 0)).sum() / (y == 0).sum()
def rc(y, s):
    o = np.argsort(-np.abs(s - 0.5), kind="stable"); err = ((s > 0.5).astype(int) != y)[o]
    return np.cumsum(err) / np.arange(1, len(y) + 1)
def aurc(y, s): return rc(y, s).mean()
def sel_acc(y, s, c): n = max(1, int(round(c * len(y)))); return 1 - rc(y, s)[n - 1]
def ece(y, s, bins=10):
    conf = np.maximum(s, 1 - s); pred = (s > 0.5).astype(int); e = 0
    for lo in np.linspace(0.5, 1, bins + 1)[:-1]:
        mk = (conf >= lo) & (conf < lo + 0.5 / bins + 1e-9)
        if mk.any(): e += mk.mean() * abs((pred[mk] == y[mk]).mean() - conf[mk].mean())
    return e

def boot_diff(df, a, b, fn, n=1000, seed=0):
    """Patient-cluster bootstrap CI of fn(a) - fn(b) on pooled out-of-fold predictions."""
    rng = np.random.default_rng(seed); groups = df.group.unique(); idx = {g: np.where(df.group.values == g)[0] for g in groups}
    y, sa, sb = df.y.values, df[a].values, df[b].values; out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
        if len(set(y[ii])) < 2: continue
        out.append(fn(y[ii], sa[ii]) - fn(y[ii], sb[ii]))
    return float(fn(y, sa) - fn(y, sb)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs_v2_nostain"); a = ap.parse_args()
    R = {}
    for level in ("image", "patient"):
        df, ex = collect(a.out, level); df.to_csv(os.path.join(a.out, f"pooled_{level}.csv"), index=False)
        y = df.y.values; meth = ["adm", "proposed", "default", "sorted", "mean", "wmean", "best", "stack", "gompertz"]
        L = {"n": int(len(df)), "n_groups": int(df.group.nunique()), "params": {k: list(map(int, v)) for k, v in ex["params"].items()}, "kappa": ex["kappa"], "w": ex["w"], "t_adm": ex["t_adm"], "viol_adm": ex["viol_adm"], "viol_tuned": ex["viol_tuned"], "viol_canon": ex["viol_canon"]}
        L["pooled"] = {m: dict(acc=acc(y, df[m]), bacc=bacc(y, df[m]), sens=sens(y, df[m]), spec=spec(y, df[m]), auc=auc(y, df[m]), aurc=aurc(y, df[m].values), ece=ece(y, df[m].values),
                               sel={str(c): sel_acc(y, df[m].values, c) for c in COV}) for m in meth}
        L["pooled"]["proposed_cal"] = dict(ece=ece(y, df.proposed_cal.values), acc=acc(y, df.proposed_cal))
        L["boot"] = {f"proposed-{b}": {nm: boot_diff(df, "proposed", b, fn) for nm, fn in (("acc", acc), ("bacc", bacc), ("auc", auc), ("aurc", aurc))} for b in ("mean", "best", "stack")}
        L["pooled"]["adm_cal"] = dict(ece=ece(y, df.adm_cal.values), acc=acc(y, df.adm_cal))
        L["boot"].update({f"adm-{b}": {nm: boot_diff(df, "adm", b, fn) for nm, fn in (("acc", acc), ("bacc", bacc))} for b in ("mean", "best", "stack")})
        L["mcnemar_adm"] = dict(only_adm_correct=int((((df.adm > 0.5) == y) & ~((df["mean"] > 0.5) == y)).sum()), only_mean_correct=int((~((df.adm > 0.5) == y) & ((df["mean"] > 0.5) == y)).sum()))
        L["boot"]["default-mean"] = {nm: boot_diff(df, "default", "mean", fn) for nm, fn in (("acc", acc), ("auc", auc), ("aurc", aurc))}
        pa, pb = (df.proposed > 0.5) == y, (df["mean"] > 0.5) == y
        L["mcnemar"] = dict(only_proposed_correct=int((pa & ~pb).sum()), only_mean_correct=int((~pa & pb).sum()), both_wrong=int((~pa & ~pb).sum()))
        # reject option: what the rejected 20 % looks like
        o = np.argsort(-np.abs(df.adm.values - 0.5), kind="stable"); keep = np.zeros(len(df), bool); keep[o[:int(round(0.8 * len(df)))]] = True
        err = (df.adm > 0.5).astype(int) != y
        L["reject20"] = dict(errors_total=int(err.sum()), errors_rejected=int((err & ~keep).sum()), share_errors_caught=float((err & ~keep).sum() / err.sum()),
                             iota_kept=float(df.iota[keep].mean()), iota_rej=float(df.iota[~keep].mean()), dis_kept=float(df.disagree[keep].mean()), dis_rej=float(df.disagree[~keep].mean()),
                             err_rate_rej=float(err[~keep].mean()), err_rate_kept=float(err[keep].mean()))
        L["iota_err"] = dict(iota_correct=float(df.iota[~err].mean()), iota_wrong=float(df.iota[err].mean()), auc_iota_detects_error=float(roc_auc_score(err, df.iota)),
                             auc_margin_detects_error=float(np.mean([roc_auc_score((d.adm > 0.5).astype(int) != d.y, -np.abs(d.adm - 0.5)) for _, d in df.groupby("fold")])), auc_meanprob_detects_error=float(roc_auc_score((df["mean"] > 0.5).astype(int) != y, -np.abs(df["mean"] - 0.5))),
                             auc_disagree_detects_error=float(roc_auc_score(err, df.disagree)))
        s = pd.DataFrame(ex["sens"], columns=["fold", "m", "q", "acc", "bacc"]).groupby(["m", "q"])[["acc", "bacc"]].mean().reset_index()
        L["sens"] = s.round(4).values.tolist()
        if level == "image":
            L["by_mag"] = {int(g): {m: dict(acc=acc(d.y.values, d[m].values), auc=auc(d.y.values, d[m].values), aurc=aurc(d.y.values, d[m].values)) for m in ("adm", "proposed", "mean", "best")} for g, d in df.groupby("mag")}
        R[level] = L
    json.dump(R, open(os.path.join(a.out, "paper_analysis.json"), "w"), indent=1, default=float)
    for level in R:
        print("==", level, R[level]["n"], R[level]["n_groups"], R[level]["params"])
        for m, v in R[level]["pooled"].items():
            if "sel" in v: print(f"{m:9s} acc {v['acc']:.4f} bacc {v['bacc']:.4f} sens {v['sens']:.4f} spec {v['spec']:.4f} auc {v['auc']:.4f} aurc {v['aurc']:.4f} ece {v['ece']:.4f} | sel " + " ".join(f"{v['sel'][str(c)]:.4f}" for c in COV))
        print("cal", R[level]["pooled"]["proposed_cal"]); print("boot", json.dumps(R[level]["boot"], default=lambda x: round(x, 4)))
        print("mcnemar", R[level]["mcnemar"], R[level]["mcnemar_adm"], "t", R[level]["t_adm"], "viol", R[level]["viol_adm"], R[level]["viol_tuned"], "cal", R[level]["pooled"]["adm_cal"]); print("rej", R[level]["reject20"]); print("iota", R[level]["iota_err"])
    print("by_mag", json.dumps(R["image"]["by_mag"]))

if __name__ == "__main__":
    main()
