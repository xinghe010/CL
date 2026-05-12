# CL-TW-GNN

Supplementary code for

> **Logic-Constrained Graph Neural Networks via Gradient-Adaptive Refinement.**

> *Note on naming.* The folder prefix `CL` stands for **Constraint-Logic**, not contrastive learning. The model in the paper is referred to as **CL-TW-GNN** (or its attention variant CL-ASTW-GNN).

## TL;DR

A term-walk GNN trained with two extra signals:

1. **CNF rule loss** — task-specific implication rules (similarity, salient-symbol co-occurrence, argument-level compatibility) are compiled into CNF and grounded into propositional clauses; their satisfaction is added to the training loss.
2. **GradRe** (Gradient-Adaptive Refinement) — a quantization-aware backward pass that rescales the surrogate gradient by the quantization discrepancy. Replaces the straight-through estimator wherever hard CNF evaluation forces non-differentiable rounding.

Headline numbers: **89.00 %** on MPTP, **86.28 %** on CNF; **225 / 280** problems proved single-pass, **253 / 280** with ten feedback rounds.

---

## Quickstart

```bash
pip install -r requirements.txt

# Train + evaluate on MPTP2078
cd premise_selection/MPTP
python eval.py --exp_name cl_mptp --epochs 50 --batch_size 32 --lr 1e-3

# Same on the CNF-clausified counterpart
cd ../CNF
python eval.py --exp_name cl_cnf  --epochs 50 --batch_size 32 --lr 1e-3

# End-to-end E-prover loop (requires E-prover ≥ 2.6 on PATH)
cd ../../ATP_experiment/scripts
python feedback_loop.py --num_rounds 10
```

---

## Repository

```
supplementary material_CL/
├── ATP_experiment/        End-to-end loop with E-prover
│   ├── code/              Selector implementation (re-used by feedback_loop.py)
│   └── scripts/           run.py, feedback_loop.py, evaluate_*.py
└── premise_selection/
    ├── MPTP/              MPTP2078 training + evaluation
    └── CNF/               Same on the CNF variant
```

The two `premise_selection` sub-folders share a common module set (`model.py`, `graph.py`, `dataset.py`, `trainer.py`, `eval.py`) and differ only in the data they load.

---

## Where the paper lives in the code

| Paper concept                              | File / symbol                                      |
|--------------------------------------------|----------------------------------------------------|
| CNF clause representation                  | `ATP_experiment/code/sat.py` (`CNF`, `clause`, `literal`) |
| Compiled rule clauses                      | `ATP_experiment/code/add.cnf`, `add.atom2idx`      |
| Rule loss (`reg_cnf`)                      | `ste.py:reg_cnf`, called from `model.py`            |
| GradRe (EWGS quantizer)                    | `custom_modules.py:EWGS_discretizer`               |
| Gradient-scaling parameters                | `scales.py`                                        |
| Term-walk encoder + classifier             | `model.py:PremiseSelectionModel`                   |
| Rule-augmented logits                      | `model.py:vg_gen`                                  |

`model.py` exposes a single boolean flag, `--cnf`, which toggles the CNF rule loss; passing `--cnf 0` recovers the unconstrained baseline used in the ablation tables.

---

## Datasets

Pre-processed graph datasets ship with the archive — no external download is required:

* `premise_selection/MPTP/` — MPTP2078 formulas as DAGs
* `premise_selection/CNF/`  — CNF-clausified counterpart
* `ATP_experiment/dataset/` — held-out problems for the E-prover loop, plus `test_problems_provable.json`

---

## Reproducing the paper

| Row in paper                              | Command                                                  | Expected   |
|-------------------------------------------|----------------------------------------------------------|------------|
| Strongest constrained variant (MPTP)      | `eval.py` in `premise_selection/MPTP`                    | 88.60 %    |
| Strongest constrained variant (CNF)       | `eval.py` in `premise_selection/CNF`                     | 85.41 %    |
| GradRe-controlled variant (MPTP)          | `eval.py --gradre 1`                                     | 89.00 %    |
| GradRe-controlled variant (CNF)           | `eval.py --gradre 1`                                     | 86.28 %    |
| End-to-end, single pass                   | `feedback_loop.py --num_rounds 1`                        | 225 / 280  |
| End-to-end, ten feedback rounds           | `feedback_loop.py --num_rounds 10`                       | 253 / 280  |

Variation of ≤ 0.3 pp across hardware is expected (non-deterministic scatter ops).

---

## Citation

Please cite the accompanying paper. The BibTeX entry will be added once the work is officially published.
