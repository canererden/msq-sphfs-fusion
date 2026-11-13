"""Read outputs/fusion_results_by_fold.csv (written by fusion.py) and print
  1) method x metric table, mean +- std over the test folds, per level
  2) fold-by-fold difference: proposed minus {mean, weighted mean, stacking} for acc, auc, aurc
Nothing is re-tuned here; this only tabulates what fusion.py produced.
"""
import sys, pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "outputs"
df = pd.read_csv(f"{out}/fusion_results_by_fold.csv")
met = ["acc", "bacc", "f1", "auc", "ece", "aurc"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for lvl, g in df.groupby("level", sort=False):
    a = g.groupby("method", sort=False)[met].agg(["mean", "std"])
    t = pd.DataFrame({m: a[m]["mean"].map("{:.4f}".format) + " ± " + a[m]["std"].map("{:.4f}".format) for m in met})
    print(f"\n=== {lvl} level (n per fold: {sorted(g.n.unique())}) ===\n{t.to_string()}")
    t.to_csv(f"{out}/summary_table_{lvl}.csv")
P = "Proposed m,s,q-SPHFS distance"
print("\n=== Proposed minus baseline, per fold (acc, auc: positive = proposed better; aurc: negative = proposed better) ===")
rows = []
for lvl, g in df.groupby("level", sort=False):
    w = g.pivot(index="fold", columns="method", values=["acc", "auc", "aurc"])
    for b in ("Mean probability", "Weighted mean probability", "Stacking (logistic regression)"):
        for m in ("acc", "auc", "aurc"):
            d = w[m][P] - w[m][b]; better = (d > 1e-12) if m != "aurc" else (d < -1e-12); worse = (d < -1e-12) if m != "aurc" else (d > 1e-12)
            rows.append(dict(level=lvl, baseline=b, metric=m, **{f"fold{f}": round(v, 4) for f, v in d.items()},
                             mean_diff=round(d.mean(), 4), folds_better=int(better.sum()), folds_equal=int((~better & ~worse).sum()),
                             folds_worse=int(worse.sum())))
r = pd.DataFrame(rows); print(r.to_string(index=False)); r.to_csv(f"{out}/proposed_vs_baselines_by_fold.csv", index=False)
