"""Run all (fold, model) trainings one after another; finished runs are skipped, so the
script can be stopped and restarted at any time.

  python run_training.py --device cpu          (or cuda)
Console output of every run is appended to outputs/train_log.txt; a one-line summary
per finished run goes to outputs/progress.txt.
"""
import argparse, json, os, subprocess, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="breakhis_index.csv"); ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", nargs="+", default=["resnet50", "densenet121", "efficientnet_b0"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="auto"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bs_map", nargs="*", default=[], help="per-model batch size, e.g. densenet121=16 (4 GB GPU)")
    a, extra = ap.parse_known_args()   # unknown options (e.g. --epochs 20 --aug strong) go to train_extract.py
    bs_map = dict(s.split("=") for s in a.bs_map)
    os.makedirs(a.out, exist_ok=True)
    log, prog = os.path.join(a.out, "train_log.txt"), os.path.join(a.out, "progress.txt")
    for f in a.folds:
        for m in a.models:
            tag = f"{m}_fold{f}"
            if all(os.path.exists(os.path.join(a.out, f"{tag}_{s}")) for s in ("val.csv", "test.csv", "log.json")):
                continue
            with open(log, "a") as fh:
                fh.write(f"\n===== {tag}  {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"); fh.flush()
                rc = subprocess.call([sys.executable, "train_extract.py", "--index", a.index, "--model", m, "--fold", str(f),
                                      "--out", a.out, "--device", a.device, "--workers", str(a.workers)] + extra
                                     + (["--bs", bs_map[m]] if m in bs_map else []),
                                     stdout=fh, stderr=subprocess.STDOUT)
            if rc != 0:
                with open(prog, "a") as fh: fh.write(f"{time.strftime('%H:%M')} {tag} FAILED (exit {rc})\n")
                raise SystemExit(f"{tag} failed, see {log}")
            r = json.load(open(os.path.join(a.out, tag + "_log.json")))
            with open(prog, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M')} {tag} device={r['device']} best_val_acc={r['best_val_acc']:.4f} "
                         f"(epoch {r['best_epoch']}) test_acc={r['test_acc_at_best_val_epoch']:.4f} minutes={r['minutes']:.1f}\n")
        with open(prog, "a") as fh: fh.write(f"FOLD {f} DONE\n")
    with open(prog, "a") as fh: fh.write("ALL DONE\n")

if __name__ == "__main__":
    main()
