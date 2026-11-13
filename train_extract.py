"""Fine-tune one backbone with an evidential (Dirichlet) head on BreakHis and save
per-image Dirichlet parameters for the validation and test folds.

Usage (run once per model and fold):
  python train_extract.py --index breakhis_index.csv --model resnet50 --fold 0 --out outputs
Test fold = --fold, validation fold = (fold+1) % K, the rest is training data.
Evidential loss: Sensoy et al., "Evidential deep learning to quantify classification
uncertainty", NeurIPS 2018 (sum-of-squares form with annealed KL regulariser).
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

def build(name, mem_eff=False):
    if name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2"); m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "densenet121":
        # memory_efficient (gradient checkpointing): same maths, lets batch 32 fit a 4 GB GPU
        m = models.densenet121(weights="IMAGENET1K_V1", memory_efficient=mem_eff); m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1"); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1"); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    else:
        raise ValueError(name)
    return m

class Rot90:
    """Random rotation by a multiple of 90 degrees (histology has no canonical orientation)."""
    def __call__(self, img): return img.rotate(90 * int(torch.randint(0, 4, (1,))))

class StainNorm:
    """Per-image stain normalisation: optical density of every RGB channel is standardised to a fixed
    mean/std (data-set averages). Removes the global colour/staining-intensity statistics, which on
    BreakHis predict the label by themselves and make the networks fail on atypically stained patients."""
    MEAN, STD = np.array([0.26, 0.49, 0.28], np.float32), np.array([0.15, 0.24, 0.13], np.float32)
    def __call__(self, img):
        od = -np.log((np.asarray(img, dtype=np.float32) + 1) / 256)
        od = (od - od.mean((0, 1))) / (od.std((0, 1)) + 1e-6) * self.STD + self.MEAN
        return Image.fromarray(np.clip(256 * np.exp(-od) - 1, 0, 255).astype(np.uint8))

def bacc(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))

class DS(Dataset):
    def __init__(self, df, tf): self.df, self.tf = df.reset_index(drop=True), tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r.path).convert("RGB")), int(r.label), i

def kl_dirichlet(alpha):
    K = alpha.shape[1]; S = alpha.sum(1, keepdim=True)
    t1 = torch.lgamma(S) - torch.lgamma(alpha).sum(1, keepdim=True) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
    t2 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(1, keepdim=True)
    return t1 + t2

def edl_loss(logits, y, anneal):
    ev = F.softplus(logits); alpha = ev + 1; S = alpha.sum(1, keepdim=True)
    y1 = F.one_hot(y, 2).float(); p = alpha / S
    err = ((y1 - p) ** 2).sum(1, keepdim=True); var = (p * (1 - p) / (S + 1)).sum(1, keepdim=True)
    kl = kl_dirichlet(y1 + (1 - y1) * alpha)
    return (err + var + anneal * kl).mean()

@torch.no_grad()
def predict(model, dl, dev, tta=1):
    """Dirichlet parameters per image. tta = 1, 4 or 8: evidence averaged over rotations
    by 90 degrees (4) and their horizontal flips (8)."""
    model.eval(); out = []
    for x, _, _ in dl:
        x = x.to(dev); ev = 0
        for t in range(tta):
            v = torch.rot90(x, t % 4, (2, 3))
            if t >= 4: v = v.flip(3)
            if dev == "cpu": v = v.contiguous(memory_format=torch.channels_last)
            ev = ev + F.softplus(model(v)).float()
        out.append((ev / tta + 1).cpu().numpy())
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--model", default="resnet50")
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--img", type=int, default=224); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (off by default: fp16 gave NaN logits on the NVIDIA T1000 used here)")
    ap.add_argument("--balance", choices=["none", "class"], default="class",
                    help="class: sample training images so that both classes are equally frequent")
    ap.add_argument("--aug", choices=["basic", "strong"], default="strong",
                    help="strong: + 90-degree rotations, wider crop scale, stronger colour (stain) jitter")
    ap.add_argument("--stain", choices=["none", "od"], default="none", help="od: per-image stain normalisation (see StainNorm)")
    ap.add_argument("--select", choices=["acc", "bacc"], default="bacc", help="validation criterion for the checkpoint")
    ap.add_argument("--tta", type=int, choices=[1, 4, 8], default=8, help="test-time augmentation views for the saved outputs")
    ap.add_argument("--no_ckpt", action="store_true", help="do not write the .pt file (pilot runs)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    amp = a.amp and dev == "cuda"
    if dev == "cuda": torch.backends.cudnn.benchmark = True
    df = pd.read_csv(a.index); K = df.fold.nunique(); vf = (a.fold + 1) % K
    tr, va, te = df[~df.fold.isin([a.fold, vf])], df[df.fold == vf], df[df.fold == a.fold]
    if a.limit: tr, va, te = [d.sample(min(a.limit, len(d)), random_state=0) for d in (tr, va, te)]
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if a.aug == "strong":
        ttf = transforms.Compose([transforms.RandomResizedCrop(a.img, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(), Rot90(), transforms.ColorJitter(0.3, 0.3, 0.4, 0.1),
                                  transforms.ToTensor(), norm])
    else:
        ttf = transforms.Compose([transforms.RandomResizedCrop(a.img, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                                  transforms.ToTensor(), norm])
    etf = transforms.Compose([transforms.Resize((a.img, a.img)), transforms.ToTensor(), norm])
    if a.stain == "od":
        ttf.transforms.insert(0, StainNorm()); etf.transforms.insert(0, StainNorm())
    assert not (set(tr.patient) & set(va.patient)) and not (set(tr.patient) & set(te.patient)) \
        and not (set(va.patient) & set(te.patient)), "patient leakage between splits"
    mk = lambda d, tf, sh, sampler=None: DataLoader(DS(d, tf), a.bs, shuffle=sh and sampler is None, sampler=sampler,
                                                    num_workers=a.workers, pin_memory=dev == "cuda", drop_last=sh,
                                                    persistent_workers=a.workers > 0)
    sampler = None
    if a.balance == "class":
        cw = 1.0 / np.bincount(tr.label.values, minlength=2)
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(cw[tr.label.values], dtype=torch.double),
                                                         num_samples=len(tr), replacement=True)
    dtr, dva, dte = mk(tr, ttf, True, sampler), mk(va, etf, False), mk(te, etf, False)
    model = build(a.model, mem_eff=dev == "cuda").to(dev)
    if dev == "cpu": model = model.to(memory_format=torch.channels_last)   # faster convolutions on CPU
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(1, len(dtr)))
    scaler = torch.amp.GradScaler(enabled=amp)
    print(f"{a.model} fold {a.fold}: device {dev}, amp {amp}, train/val/test images {len(tr)}/{len(va)}/{len(te)}", flush=True)
    best, best_acc, best_state, best_ep, hist, t_start = -1, -1, None, -1, [], time.time()
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y, _ in dtr:
            x, y = x.to(dev), y.to(dev)
            if dev == "cpu": x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev, enabled=amp):
                logits = model(x)
            loss = edl_loss(logits.float(), y, min(1.0, ep / 10))
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {ep+1}; nothing saved")
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        al = predict(model, dva, dev); yv = va.label.values; pv = al.argmax(1)
        acc, vb = float((pv == yv).mean()), bacc(yv, pv); crit = vb if a.select == "bacc" else acc
        hist.append(dict(epoch=ep + 1, loss=tot / max(1, len(dtr)), val_acc=acc, val_bacc=vb, seconds=time.time() - t0))
        print(f"ep {ep+1}/{a.epochs} loss {tot/max(1,len(dtr)):.4f} val_acc {acc:.4f} val_bacc {vb:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if crit > best:
            best, best_ep, best_state = crit, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_acc = acc
    model.load_state_dict(best_state)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.model}_fold{a.fold}"
    if not a.no_ckpt: torch.save(best_state, os.path.join(a.out, tag + "_best.pt"))
    res = {}
    for name, d, dl in (("val", va, dva), ("test", te, dte)):
        al = predict(model, dl, dev, a.tta)
        assert np.isfinite(al).all(), "non-finite Dirichlet parameters"
        o = d[["path", "label", "patient", "mag"]].reset_index(drop=True).copy()
        o["alpha_ben"], o["alpha_mal"] = al[:, 0], al[:, 1]
        o.to_csv(os.path.join(a.out, f"{tag}_{name}.csv"), index=False)
        res[name + "_acc"] = float((al.argmax(1) == d.label.values).mean()); res[name + "_bacc"] = bacc(d.label.values, al.argmax(1))
    with open(os.path.join(a.out, tag + "_log.json"), "w") as fh:
        json.dump(dict(args=vars(a), device=dev, amp=amp, torch=torch.__version__, best_val_acc=best_acc, best_val_crit=best,
                       best_epoch=best_ep, val_acc_tta=res["val_acc"], val_bacc_tta=res["val_bacc"],
                       test_acc_at_best_val_epoch=res["test_acc"], test_bacc_at_best_val_epoch=res["test_bacc"],
                       n_train=len(tr), n_val=len(va), n_test=len(te),
                       minutes=(time.time() - t_start) / 60, history=hist), fh, indent=1)
    print(f"saved; best val {a.select} {best:.4f} (epoch {best_ep}); with tta {a.tta}: val acc {res['val_acc']:.4f} bacc {res['val_bacc']:.4f}, "
          f"test acc {res['test_acc']:.4f} bacc {res['test_bacc']:.4f}; {(time.time()-t_start)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
