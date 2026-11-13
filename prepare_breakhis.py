"""Index BreakHis and build patient-level stratified folds.

Usage:  python prepare_breakhis.py --data_root "PATH/TO/BreaKHis_v1" --out breakhis_index.csv
File names look like  SOB_M_DC-14-2523-40-010.png :
  SOB_<B|M>_<subtype>-<year>-<slide/patient id>-<magnification>-<seq>.png
All images of one patient go to the same fold (no leakage between train/val/test).
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

PAT = re.compile(r"SOB_([BM])_([A-Z]+)-(\d+)-([0-9A-Za-z]+)-(\d+)-(\d+)\.png$", re.I)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="breakhis_index.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows, skipped = [], 0
    for p in Path(a.data_root).rglob("*.png"):
        mt = PAT.search(p.name)
        if not mt:
            skipped += 1; continue
        cls, sub, year, pid, mag, _ = mt.groups()
        rows.append(dict(name=p.name, path=str(p), label=int(cls.upper() == "M"), subtype=sub.upper(),
                         patient=f"{year}-{pid}", mag=int(mag)))
    # some distributions ship every image twice (binary and multiclass folder trees): keep one copy
    df = pd.DataFrame(rows).sort_values("path").drop_duplicates("name").drop(columns="name")
    if df.empty:
        raise SystemExit("No BreakHis images found under " + a.data_root)
    pat = df.groupby("patient").label.first().reset_index()
    mixed = df.groupby("patient").label.nunique()
    assert (mixed == 1).all(), "a patient has both labels: " + str(mixed[mixed > 1].index.tolist())
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    pat["fold"] = -1
    for f, (_, te) in enumerate(skf.split(pat.patient, pat.label)):
        pat.loc[te, "fold"] = f
    df = df.merge(pat[["patient", "fold"]], on="patient")
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images, {pat.shape[0]} patients, skipped {skipped} files")
    print(df.groupby(["label", "mag"]).size().unstack())
    print(pat.groupby(["fold", "label"]).size().unstack())

if __name__ == "__main__":
    main()
