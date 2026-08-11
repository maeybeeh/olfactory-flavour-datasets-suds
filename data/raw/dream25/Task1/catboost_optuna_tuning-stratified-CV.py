#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import optuna
import pandas as pd

os.environ.setdefault("TBB_NUM_THREADS", "9")
os.environ.setdefault("OMP_NUM_THREADS", "9")
os.environ.setdefault("MKL_NUM_THREADS", "9")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "9")

from catboost import CatBoostRegressor, Pool
from catboost.core import CatBoostError
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

META_COLS = {"stimulus", "Intensity_label"}
FLAG_COLS = ["flag_H_to_L", "flag_L_to_H"]

def load_feature_table(mordred_csv: str | Path, morgan_csv: str | Path) -> pd.DataFrame:
    mordred = pd.read_csv(mordred_csv)
    mordred = mordred.drop(columns=[c for c in mordred.columns if c.lower() == "smiles"], errors="ignore")
    mordred_cols = [c for c in mordred.columns if c != "molecule"]
    mordred[mordred_cols] = mordred[mordred_cols].apply(pd.to_numeric, errors="coerce")
    try:
        morgan = pd.read_csv(morgan_csv, encoding="utf-8")
    except UnicodeDecodeError:
        morgan = pd.read_csv(morgan_csv, encoding="latin-1")
    morgan = morgan.drop(columns=[c for c in morgan.columns if c.lower() == "smiles"], errors="ignore")
    bit_cols = [c for c in morgan.columns if c != "molecule"]
    morgan[bit_cols] = morgan[bit_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    if "molecule" not in mordred.columns or "molecule" not in morgan.columns:
        raise ValueError("Both descriptor files must contain a molecule column")
    merged = mordred.merge(morgan, on="molecule", how="inner", validate="one_to_one")
    merged.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
    const_cols = [c for c in merged.columns if c != "molecule" and merged[c].nunique(dropna=False) <= 1]
    if const_cols:
        merged = merged.drop(columns=const_cols)
    return merged

def build_rating_lookup(df: pd.DataFrame, rating_cols: List[str]) -> Dict[Tuple[int, str], np.ndarray]:
    return {(int(r.molecule), r.Intensity_label): r[rating_cols].to_numpy(float) for _, r in df.iterrows()}

def build_matrices(
    train_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    rating_cols: List[str],
    dil_lookup: Dict[Tuple[int, str], float],
    seed: int = 42, 
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, int]:
    lookup = build_rating_lookup(train_df, rating_cols)
    mols = [m for (m, _) in lookup if (m, "H") in lookup and (m, "L") in lookup]
    if len(mols) < 3:
        raise RuntimeError("Need ≥3 molecules with both H & L ratings present")
    feat_cols = [c for c in feat_df.columns if c != "molecule"]
    desc_idx = feat_df.set_index("molecule")
    desc_dim = len(feat_cols)
    X_rows, Y_rows, groups = [], [], []
    for m in mols:
        desc = desc_idx.loc[m, feat_cols].to_numpy(float) if m in desc_idx.index else np.full(desc_dim, np.nan, np.float32)
        H, L = lookup[(m, "H")], lookup[(m, "L")]
        dil_H = dil_lookup.get((m, "H"), 0.0)
        dil_L = dil_lookup.get((m, "L"), 0.0)
        X_rows.append(np.hstack([desc, [dil_H, dil_L], H, [1, 0]])); Y_rows.append(L); groups.append(m)
        X_rows.append(np.hstack([desc, [dil_L, dil_H], L, [0, 1]])); Y_rows.append(H); groups.append(m)
    X = np.nan_to_num(np.vstack(X_rows).astype(np.float32), nan=0.0)
    Y = np.vstack(Y_rows).astype(np.float32)
    groups_arr = np.asarray(groups)
    return X, Y, groups_arr, lookup, desc_dim

def mean_pearson_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    pears, coss = [], []
    for t, p in zip(y_true, y_pred):
        if np.nanstd(t) == 0 or np.nanstd(p) == 0:
            continue
        r = pearsonr(t, p)[0]; d = cos_dist(t, p)
        if np.isfinite(r) and np.isfinite(d):
            pears.append(r); coss.append(d)
    return (float(np.nanmean(pears)), float(np.nanmean(coss))) if pears else (0.0, 1.0)

def build_feature_row(
    mol: int, lab: str, known_vec: np.ndarray,
    desc_idx: pd.DataFrame, feat_cols: List[str],
    dil_lookup: Dict[Tuple[int, str], float],
) -> np.ndarray:
    desc = desc_idx.loc[mol, feat_cols].to_numpy(float) if mol in desc_idx.index else np.full(len(feat_cols), np.nan, np.float32)
    dil_H = dil_lookup.get((mol, "H"), 0.0); dil_L = dil_lookup.get((mol, "L"), 0.0)
    dil_pair = [dil_H, dil_L] if lab == "L" else [dil_L, dil_H]
    flags = [1, 0] if lab == "L" else [0, 1]
    return np.nan_to_num(np.hstack([desc, dil_pair, known_vec, flags]).astype(np.float32), nan=0.0)

def opposite_label(lab: str) -> str:
    return "H" if lab == "L" else "L"

def predict_template_with_fallback(
    template_df: pd.DataFrame, stim_df: pd.DataFrame, model: CatBoostRegressor,
    desc_df: pd.DataFrame, rating_cols: List[str],
    dil_lookup: Dict[Tuple[int, str], float],
    train_lookup: Dict[Tuple[int, str], np.ndarray] | None = None,
    use_fallback: bool = True,
) -> pd.DataFrame:
    feat_cols = [c for c in desc_df.columns if c != "molecule"]
    desc_idx = desc_df.set_index("molecule")
    stim_meta = stim_df.set_index("stimulus")
    preds = []
    for stim in template_df["stimulus"]:
        mol = int(stim_meta.loc[stim, "molecule"])
        target_lab = stim_meta.loc[stim, "Intensity_label"]
        known_from_template = template_df.loc[template_df["stimulus"] == stim, rating_cols].to_numpy(float).ravel()
        known_vec = None if np.isnan(known_from_template).all() else known_from_template
        if known_vec is None and use_fallback and train_lookup is not None:
            known_vec = train_lookup.get((mol, opposite_label(target_lab)), None)
        if known_vec is None:
            known_vec = np.zeros(len(rating_cols), dtype=np.float32)
        row = build_feature_row(mol, target_lab, known_vec, desc_idx, feat_cols, dil_lookup)
        preds.append(model.predict(row.reshape(1, -1)).ravel())
    out = template_df.copy()
    out[rating_cols] = np.vstack(preds)
    return out

def augment_training_data(
    X_tr: np.ndarray, Y_tr: np.ndarray, *,
    desc_dim: int, rating_dim: int,
    mask_known_prob: float, mask_label_prob: float, masked_label_weight: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_aug = X_tr.copy(); Y_aug = Y_tr.copy()
    weights = np.ones(len(X_tr), dtype=np.float32)
    known_start = desc_dim + 2; known_end = known_start + rating_dim
    if mask_known_prob > 0.0:
        mk = rng.rand(len(X_aug)) < mask_known_prob
        if mk.any():
            X_aug[mk, known_start:known_end] = 0.0
    if mask_label_prob > 0.0:
        my = rng.rand(len(Y_aug)) < mask_label_prob
        if my.any():
            Y_aug[my, :] = 0.0
            weights[my] = float(masked_label_weight)
    return X_aug, Y_aug, weights

def _resolve_devices_str(gpu_id: int) -> str:
    return "0" if os.environ.get("CUDA_VISIBLE_DEVICES") else str(gpu_id)

def _compute_molecule_sigmoid_fraction(
    train_df: pd.DataFrame,
    rating_cols: List[str],
    epsilon: float,
    sigmoid_cut: float,
) -> pd.Series:
    cols_needed = ["molecule", "Intensity_label"] + rating_cols
    df = train_df[cols_needed].copy()
    for c in rating_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby(["molecule", "Intensity_label"], dropna=True)[rating_cols].mean()
    try:
        L = agg.xs("L", level="Intensity_label")
        H = agg.xs("H", level="Intensity_label")
    except KeyError:
        raise RuntimeError("Both 'L' and 'H' rows are required per molecule to compute sigmoid fractions.")
    common = L.index.intersection(H.index)
    L = L.loc[common]; H = H.loc[common]
    delta_log2 = np.log2((H.values + epsilon) / (L.values + epsilon))
    sig_mask = (np.abs(delta_log2) > float(sigmoid_cut))
    sig_frac = sig_mask.mean(axis=1)
    return pd.Series(sig_frac, index=common, name="sigmoid_frac")

def _build_stratified_group_splits(
    groups: np.ndarray,
    mol_sig_frac: pd.Series,
    n_splits: int,
    seed: int,
    n_bins: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    unique_mols = np.unique(groups)
    sig = mol_sig_frac.reindex(unique_mols).fillna(0.0).to_numpy()

    if n_splits < 2:
        return []

    if n_bins <= 1:
        bins = np.zeros_like(sig, dtype=int)
    else:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(sig, qs)
        edges[0] = -np.inf; edges[-1] = np.inf
        if not np.all(np.diff(edges) > 0):
            edges = np.linspace(-np.inf, np.inf, n_bins + 1)
        bins = np.digitize(sig, edges[1:-1], right=False)

    mols_by_bin: Dict[int, List[int]] = {}
    for mol, b in zip(unique_mols, bins):
        mols_by_bin.setdefault(int(b), []).append(int(mol))

    folds_mols: List[List[int]] = [[] for _ in range(n_splits)]
    cursor = 0
    for b in sorted(mols_by_bin.keys()):
        lst = mols_by_bin[b]
        rng.shuffle(lst)
        for mol in lst:
            folds_mols[cursor % n_splits].append(mol)
            cursor += 1

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_mols = folds_mols[k]
        if len(val_mols) == 0:
            continue
        val_idx = np.where(np.isin(groups, val_mols))[0]
        if len(val_idx) == 0:
            continue
        train_idx = np.where(~np.isin(groups, val_mols))[0]
        splits.append((train_idx, val_idx))

    if len(splits) != n_splits:
        gkf = GroupKFold(n_splits=min(max(2, n_splits), len(unique_mols)))
        splits = list(gkf.split(np.zeros(len(groups)), np.zeros(len(groups)), groups))

    return splits

def make_objective(
    X: np.ndarray, Y: np.ndarray, groups: np.ndarray, gpu_id: int, seed: int, *,
    desc_dim: int, mask_known_prob: float, mask_label_prob: float, masked_label_weight: float,
    thread_count: int, rating_dim: int, gpu_ram_part: float = 0.30,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
):
    n_groups = len(np.unique(groups))
    if splits is None or len(splits) < 2:
        gkf = GroupKFold(n_splits=min(max(2, 5), n_groups))
        splits_list = list(gkf.split(X, Y, groups))
    else:
        splits_list = list(splits)

    devices_str = _resolve_devices_str(gpu_id)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function": "MultiRMSE",
            "task_type": "GPU",
            "devices": devices_str,
            "boosting_type": "Plain",
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "iterations": trial.suggest_int("iterations", 1500, 4000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
            "bootstrap_type": "Bayesian",
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "gpu_ram_part": float(gpu_ram_part),
            "thread_count": thread_count,
            "random_seed": seed,
            "od_type": "Iter",
            "od_wait": 100,
            "verbose": False,
            "allow_writing_files": False,
        }
        print(f"[W{thread_count}|Trial {trial.number:03}] depth={params['depth']} "
              f"lr={params['learning_rate']:.4g} iters={params['iterations']} "
              f"l2={params['l2_leaf_reg']:.4g} bagT={params['bagging_temperature']:.3g}")

        val_pears, val_coss = [], []
        for fold_id, (tr_idx, va_idx) in enumerate(splits_list):
            rng = np.random.RandomState(seed + trial.number * 9973 + fold_id * 101)
            X_tr, Y_tr = X[tr_idx], Y[tr_idx]
            X_va, Y_va = X[va_idx], Y[va_idx]
            X_aug, Y_aug, w_aug = augment_training_data(
                X_tr, Y_tr, desc_dim=desc_dim, rating_dim=rating_dim,
                mask_known_prob=mask_known_prob, mask_label_prob=mask_label_prob,
                masked_label_weight=masked_label_weight, rng=rng,
            )
            model = CatBoostRegressor(**params)
            try:
                model.fit(Pool(X_aug, Y_aug, weight=w_aug), eval_set=Pool(X_va, Y_va), verbose=False)
            except CatBoostError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda error 2" in msg:
                    trial.set_user_attr("oom", True)
                    raise optuna.exceptions.TrialPruned()
                raise
            pred = model.predict(X_va)
            pear, cos = mean_pearson_cosine(Y_va, pred)
            val_pears.append(pear); val_coss.append(cos)

        mean_val_pear = float(np.mean(val_pears)) if len(val_pears) else 0.0
        mean_val_cos = float(np.mean(val_coss)) if len(val_coss) else 1.0
        trial.set_user_attr("val_pear", mean_val_pear)
        trial.set_user_attr("val_cos", mean_val_cos)
        print(f"[Trial {trial.number:03}] Val Pearson={mean_val_pear:.4f} Cos={mean_val_cos:.4f}")
        return 1.0 - mean_val_pear

    return objective

def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--train", default="../data/Task1/TASK1_training_All.csv")
    ap.add_argument("--stim", default="../data/Task1/TASK1_Stimulus_definition.csv")
    ap.add_argument("--test_template", default="../data/Task1/TASK1_test_set_Submission_form.csv")
    ap.add_argument("--mordred", default="../data/Task1/Mordred_Descriptors.csv")
    ap.add_argument("--morgan", default="../data/Task1/Morgan_fingerprints_utf8.csv")
    ap.add_argument("--study_csv", default="../output/Task1/catboost_tuning_study.csv")
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=42) 
    ap.add_argument("--mask_known_prob", type=float, default=0.0)
    ap.add_argument("--mask_label_prob", type=float, default=0.0)
    ap.add_argument("--masked_label_weight", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TBB_NUM_THREADS", "4")))
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gpu_ram_part", type=float, default=0.30)
    ap.add_argument("--storage", default=None)
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--stratify_regime", action="store_true")
    ap.add_argument("--sigmoid_cut", type=float, default=7.0)
    ap.add_argument("--epsilon", type=float, default=1e-4)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--writer_id", type=int, default=0)

    args, _ = ap.parse_known_args(sys.argv[1:])
    if args.gpus is not None:
        gpus = [int(x) for x in str(args.gpus).replace(" ", "").split(",") if x != ""]
    elif args.gpu is not None:
        gpus = [int(args.gpu)]
    else:
        gpus = [0]
    setattr(args, "gpus", gpus)
    return args

def main() -> None:
    args = parse_cli()

    worker_seed = int(args.seed) + 1009 * int(args.worker_id)
    print(f"[CFG] worker_id={args.worker_id} seed={args.seed} → worker_seed={worker_seed}")

    train_df = pd.read_csv(args.train)
    stim_def = pd.read_csv(args.stim)
    test_tmpl = pd.read_csv(args.test_template)

    if "dilution" not in stim_def.columns:
        raise RuntimeError("'dilution' column missing in TASK1_Stimulus_definition.csv")
    stim_def["dilution"] = pd.to_numeric(stim_def["dilution"], errors="coerce").fillna(0)
    dilution_lookup = {(int(r.molecule), r.Intensity_label): float(r.dilution) for _, r in stim_def.iterrows()}

    train_df = train_df.merge(stim_def[["stimulus", "molecule", "Intensity_label"]], on="stimulus", how="left")

    meta_cols = META_COLS | set(FLAG_COLS) | {"molecule"}
    train_rating = [c for c in train_df.columns if c not in meta_cols]
    rating_cols = [c for c in test_tmpl.columns if c in train_rating]
    if not rating_cols:
        raise RuntimeError("No rating columns after intersecting train and test template!")

    feat_df = load_feature_table(args.mordred, args.morgan)
    X, Y, groups, _, desc_dim = build_matrices(train_df, feat_df, rating_cols, dilution_lookup, seed=args.seed)
    train_lookup = {(int(r.molecule), r.Intensity_label): r[rating_cols].to_numpy(float) for _, r in train_df.iterrows()}

    splits = None
    if args.stratify_regime:
        sig_frac = _compute_molecule_sigmoid_fraction(
            train_df=train_df, rating_cols=rating_cols,
            epsilon=args.epsilon, sigmoid_cut=args.sigmoid_cut
        )
        n_splits = min(max(2, args.cv_splits), len(np.unique(groups)))
        splits = _build_stratified_group_splits(
            groups=groups, mol_sig_frac=sig_frac,
            n_splits=n_splits, seed=args.seed, n_bins=3
        )

    sampler = optuna.samplers.TPESampler(
        seed=worker_seed,
        multivariate=True,
        group=True,
        n_startup_trials=0,
        n_ei_candidates=64,
        constant_liar=True,
    )

    if args.storage and args.study_name:
        study = optuna.create_study(direction="minimize", storage=args.storage,
                                    study_name=args.study_name, load_if_exists=True, sampler=sampler)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)

    objective = make_objective(
        X, Y, groups, gpu_id=args.gpus[0], seed=worker_seed,
        desc_dim=desc_dim, mask_known_prob=args.mask_known_prob,
        mask_label_prob=args.mask_label_prob, masked_label_weight=args.masked_label_weight,
        thread_count=args.threads, rating_dim=len(rating_cols),
        gpu_ram_part=args.gpu_ram_part, splits=splits,
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    if int(args.worker_id) != int(args.writer_id):
        print(f"[WriterGuard] worker_id={args.worker_id} != writer_id={args.writer_id} → skip writing.")
        study.trials_dataframe().to_csv(args.study_csv.replace(".csv", f"_worker{args.worker_id}.csv"), index=False)
        return
    
    if args.storage and args.study_name:
        study = optuna.load_study(storage=args.storage, study_name=args.study_name)

    
    best = study.best_trial
    print("\n======= BEST TRIAL =======")
    print(f"Pearson r (val) : {1 - best.value:.4f}")
    print(f"Cosine  (val)   : {best.user_attrs.get('val_cos'):.4f}")
    print("params          :", json.dumps(best.params, indent=2))

    rng_final = np.random.RandomState(worker_seed + 987_123)
    X_full_aug, Y_full_aug, w_full = augment_training_data(
        X, Y, desc_dim=desc_dim, rating_dim=len(rating_cols),
        mask_known_prob=args.mask_known_prob, mask_label_prob=args.mask_label_prob,
        masked_label_weight=args.masked_label_weight, rng=rng_final,
    )

    final_params = best.params | {
        "loss_function": "MultiRMSE",
        "task_type": "GPU",
        "devices": _resolve_devices_str(args.gpus[0]),
        "boosting_type": "Plain",
        "bootstrap_type": "Bayesian",
        "random_seed": worker_seed,
        "od_type": "Iter",
        "od_wait": 100,
        "verbose": False,
        "thread_count": args.threads,
        "allow_writing_files": False,
        "gpu_ram_part": float(args.gpu_ram_part),
    }

    model = CatBoostRegressor(**final_params)
    model.fit(Pool(X_full_aug, Y_full_aug, weight=w_full))

    test_pred = predict_template_with_fallback(
        test_tmpl, stim_def, model, feat_df, rating_cols,
        dil_lookup=dilution_lookup, train_lookup=train_lookup, use_fallback=True
    )
    os.makedirs(os.path.dirname(args.study_csv), exist_ok=True)
    out_dir = "../output/Task1"
    os.makedirs(out_dir, exist_ok=True)
    
    out_csv = os.path.join(out_dir, "TASK1_test_predictions_DIRECT_Final.csv")
    test_pred.to_csv(out_csv, index=False)


    study.trials_dataframe().to_csv(args.study_csv, index=False)
    print(f"\nSaved: {out_csv}, {args.study_csv}")

if __name__ == "__main__":
    main()
