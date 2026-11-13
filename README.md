# m,s,q-SPHFS distance fusion of evidential deep models (BreakHis)

Code and results for the paper

> Ş. Özlü, C. Erden, A. Sarkar. *Generalized hybrid distance measures on m,s,q-spherical probabilistic hesitant fuzzy sets: uncertainty-aware fusion of evidential deep models with a reject option for breast histopathology.* Submitted, 2026.

The repository contains the implementation of the generalized hybrid weighted distance for m,s,q-spherical probabilistic hesitant fuzzy sets, the training script for the evidential base models, the fusion and evaluation code, the patient-wise fold assignment, and the per-image model outputs from which every table and figure of the paper can be regenerated without retraining.

## Contents

| Path | Description |
|---|---|
| `msq_sphfs.py` | Generalized hybrid (weighted) distance, Eqs. (9)–(10) of the paper |
| `prepare_breakhis.py` | Indexes the images and builds patient-wise stratified folds |
| `train_extract.py`, `run_training.py` | Fine-tune ResNet-50, DenseNet-121 and EfficientNet-B0 with an evidential head; write per-image Dirichlet parameters |
| `fusion.py` | Proposed rule (performance-tuned setting), fusion baselines, metrics |
| `paper_analysis.py` | Admissible setting, pooled out-of-fold predictions, bootstrap intervals, calibration |
| `paper_ablation.py` | Ablations in the admissible setting, fold-averaged bootstrap |
| `paper_tables.py` | Fold-wise tables and Figures 2–4 |
| `data/breakhis_folds.csv` | Fold assignment of the 7,909 images (file name, label, subtype, patient, magnification, fold) |
| `outputs/main/` | Per-image outputs of the three models for every validation and test fold, pooled predictions, result files and figures of the main configuration |
| `outputs/other_configurations/` | Fold-wise fusion results of the three additional training configurations (Table 8) |

## Reproducing the tables and figures

Python 3.10 or later.

```
pip install -r requirements.txt
python fusion.py --out outputs/main --level both
python paper_analysis.py --out outputs/main
python paper_ablation.py outputs/main
python paper_tables.py outputs/main
```

These steps need no GPU and no images; they read the stored model outputs.

## Retraining the base models

The BreakHis images are not redistributed here. Obtain them from the dataset authors (Spanhol et al., 2016, https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and use the binary benign/malignant tree only.

```
python prepare_breakhis.py --data_root PATH/TO/classificacao_binaria --out breakhis_index.csv
python train_extract.py --index breakhis_index.csv --model resnet50 --fold 0 --out outputs/retrained
```

Run the second command for each of `resnet50`, `densenet121`, `efficientnet_b0` and each fold 0–4. `prepare_breakhis.py` with its default seed reproduces the fold assignment in `data/breakhis_folds.csv`. Training settings are described in the paper; `train_extract.py --help` lists the corresponding options.

## Data

BreakHis: F. A. Spanhol, L. S. Oliveira, C. Petitjean, L. Heutte, "A dataset for breast cancer histopathological image classification", *IEEE Transactions on Biomedical Engineering* 63(7), 1455–1462, 2016.

## License

Code is released under the MIT License (see `LICENSE`). The stored model outputs and result files may be reused with attribution to the paper.
