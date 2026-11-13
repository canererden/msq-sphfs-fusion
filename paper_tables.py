"""Fold-wise tables and figures for the manuscript from pooled_{image,patient}.csv."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paper_analysis import acc, bacc, auc, aurc, sens, spec, rc, sel_acc, ece, MODELS, COV
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs_v2_nostain"; FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
C = {"adm": "#2a78d6", "proposed": "#1baf7a", "mean": "#eb6834", "best": "#eda100", "stack": "#6250d6"}
LS = {"adm": "-", "proposed": (0, (5, 2)), "mean": (0, (1, 1.2)), "best": (0, (4, 1.5, 1, 1.5)), "stack": (0, (8, 2, 1, 2))}
NAME = {"adm": "Proposed (admissible)", "proposed": "Proposed (performance-tuned)", "default": "Proposed (canonical limit)", "mean": "Mean probability", "wmean": "Weighted mean probability",
        "best": "Best single model", "stack": "Stacking (logistic regression)", "gompertz": "Fuzzy rank (Gompertz)", "crisp": "Crisp distance-to-ideal ensemble"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
                     "xtick.color": "#555", "ytick.color": "#555", "axes.labelcolor": "#222", "grid.color": "#e3e3e0", "grid.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight"})

def crisp(P):
    """Product of Euclidean, Manhattan and cosine distances of the K class supports to the ideal vector 1_K."""
    def D(v):
        e = np.sqrt(((1 - v) ** 2).sum(1)); m = np.abs(1 - v).sum(1); c = 1 - v.sum(1) / (np.linalg.norm(v, axis=1) * np.sqrt(v.shape[1]) + 1e-12)
        return e * m * c
    dm, db = D(P), D(1 - P); return db / (db + dm + 1e-12)

R = {}
for level in ("image", "patient"):
    df = pd.read_csv(os.path.join(OUT, f"pooled_{level}.csv")); df["crisp"] = crisp(df[["p0", "p1", "p2"]].values)
    df.to_csv(os.path.join(OUT, f"pooled_{level}.csv"), index=False)
    meth = ["adm", "proposed", "default", "mean", "wmean", "best", "stack", "gompertz", "crisp"]; L = {}
    fn = dict(acc=acc, bacc=bacc, sens=sens, spec=spec, auc=auc, aurc=aurc, ece=ece)
    per = {m: {k: [f(d.y.values, d[m].values) for _, d in df.groupby("fold")] for k, f in fn.items()} for m in meth}
    L["fold"] = {m: {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in per[m].items()} for m in meth}
    L["sel"] = {m: {str(c): (float(np.mean(v)), float(np.std(v, ddof=1))) for c in COV for v in [[sel_acc(d.y.values, d[m].values, c) for _, d in df.groupby("fold")]]} for m in meth}
    L["aurc_folds_better_than_mean"] = {m: int(sum(a < b - 1e-12 for a, b in zip(per[m]["aurc"], per["mean"]["aurc"]))) for m in ("adm", "proposed", "default", "stack", "best")}
    L["pooled_acc"] = {m: (float(acc(df.y.values, df[m].values)), int(((df[m] > 0.5) == df.y).sum())) for m in meth}
    R[level] = L
    # ---- risk-coverage figure (fold-averaged) ----
    grid = np.linspace(0.3, 1.0, 71)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for m in ("adm", "proposed", "mean", "best", "stack"):
        cur = np.mean([[1 - sel_acc(d.y.values, d[m].values, c) for c in grid] for _, d in df.groupby("fold")], 0)
        ax.plot(grid * 100, cur * 100, color=C[m], ls=LS[m], lw=1.8, label=NAME[m])
    ax.set_xlabel("Coverage (% of cases decided automatically)"); ax.set_ylabel("Selective risk (% errors among decided)"); ax.grid(True, axis="y"); ax.set_xlim(30, 100); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left"); ax.set_title(f"{'Image' if level == 'image' else 'Patient'} level", fontsize=9.5, loc="left", color="#222")
    fig.savefig(os.path.join(FIG, f"risk_coverage_{level}.png")); plt.close(fig)

# single models (image level)
img = pd.read_csv(os.path.join(OUT, "pooled_image.csv")); y = img.y.values
R["single"] = {MODELS[k]: {n: (float(np.mean(v)), float(np.std(v, ddof=1))) for n, f in dict(acc=acc, bacc=bacc, sens=sens, spec=spec, auc=auc).items()
               for v in [[f(d.y.values, d[f"p{k}"].values) for _, d in img.groupby("fold")]]} for k in range(3)}
R["counts"] = dict(images=int(len(img)), benign=int((y == 0).sum()), malignant=int((y == 1).sum()), per_mag=img.groupby("mag").size().to_dict(),
                   folds=img.groupby("fold").agg(n=("y", "size"), mal=("y", "sum"), patients=("group", "nunique")).to_dict("index"))
# ---- abstinence / disagreement vs error (image level, proposed) ----
err = ((img.adm > 0.5).astype(int) != y)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, col, lab in ((axs[0], "iota", "Mean abstinence degree  ῑ  (quintile)"), (axs[1], "disagree", "Inter-model disagreement, s.d. of μ  (quintile)")):
    q = pd.qcut(img[col], 5, labels=False); rate = err.groupby(q).mean() * 100; edges = img[col].groupby(q).agg(["min", "max"])
    ax.bar(range(5), rate.values, width=0.62, color="#2a78d6"); ax.set_xticks(range(5)); ax.set_xticklabels([f"{a:.2f}–{b:.2f}" for a, b in edges.values], fontsize=7.5)
    for i, v in enumerate(rate.values): ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color="#222")
    ax.set_xlabel(lab); ax.grid(True, axis="y"); ax.set_axisbelow(True)
    R[f"err_by_{col}_quintile"] = [float(v) for v in rate.values]
axs[0].set_ylabel("Error rate (%)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "error_vs_uncertainty.png")); plt.close(fig)
# ---- sensitivity heatmap ----
A = json.load(open(os.path.join(OUT, "paper_analysis.json")))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, level in zip(axs, ("image", "patient")):
    s = pd.DataFrame(A[level]["sens"], columns=["m", "q", "acc", "bacc"]); M = s.pivot(index="m", columns="q", values="acc").values * 100
    R[f"sens_range_{level}"] = (float(M.min()), float(M.max()), float(M[0, 0]))
    im = ax.imshow(M, cmap="Blues", origin="lower", vmin=M.min() - 1.5, vmax=M.max() + 0.5)
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color="#0d366b" if M[i, j] < (M.min() + M.max()) / 2 else "white")
    ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6)); ax.set_xlabel("q (non-membership exponent)"); ax.set_ylabel("m (membership exponent)")
    ax.set_title(f"{'Image' if level == 'image' else 'Patient'} level: accuracy (%)", fontsize=9.5, loc="left", color="#222")
    for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "sensitivity_m_q.png")); plt.close(fig)
# ---- example patients ----
pat = pd.read_csv(os.path.join(OUT, "pooled_patient.csv")); pat["margin"] = (pat.adm - 0.5).abs(); pat["ok"] = (pat.adm > 0.5).astype(int) == pat.y
R["examples"] = pat.sort_values("margin")[["fold", "id", "y", "adm", "proposed", "mean", "da_mal", "da_ben", "iota", "disagree", "ok"]].round(4).to_dict("records")
json.dump(R, open(os.path.join(OUT, "paper_tables.json"), "w"), indent=1)
f3 = lambda t: f"{t[0]*100:.2f}±{t[1]*100:.2f}"
for level in ("image", "patient"):
    print("==", level, R[level]["pooled_acc"]); print("aurc folds better than mean:", R[level]["aurc_folds_better_than_mean"])
    for m, v in R[level]["fold"].items(): print(f"{m:9s}", " ".join(f"{k} {f3(v[k])}" for k in v), "| sel", " ".join(f3(R[level]["sel"][m][str(c)]) for c in COV[1:]))
print("single", {m: {k: f3(v) for k, v in d.items()} for m, d in R["single"].items()}); print(R["counts"])
print("iota quint", R["err_by_iota_quintile"], "dis quint", R["err_by_disagree_quintile"], R["sens_range_image"], R["sens_range_patient"])
