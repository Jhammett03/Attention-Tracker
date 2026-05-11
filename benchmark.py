import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import cnn_time_model
import transformer_model

N_RUNS  = 10
N_FOLDS = 5

CSV_DATASETS = {
    "old":  "dataset.csv",
    "iris": "dataset_iris.csv",
    "gaze": "dataset_gaze.csv",
}

NPZ_DATASETS = {
    "old":  "sequence_dataset.npz",
    "iris": "sequence_dataset_iris.npz",
    "gaze": "sequence_dataset_gaze.npz",
}

_FEATURE_CANDIDATES = [
    ["mean_left_ear", "mean_right_ear", "std_left_ear", "std_right_ear",
     "mean_left_gaze_x", "mean_left_gaze_y", "mean_right_gaze_x", "mean_right_gaze_y",
     "var_left_gaze_x", "var_left_gaze_y", "var_right_gaze_x", "var_right_gaze_y",
     "mean_head_x", "mean_head_y", "var_head_x", "var_head_y",
     "mean_bb_tl_x", "mean_bb_tl_y", "mean_bb_tr_x", "mean_bb_tr_y",
     "mean_bb_bl_x", "mean_bb_bl_y", "mean_bb_br_x", "mean_bb_br_y",
     "var_bb_tl_x", "var_bb_tl_y", "var_bb_tr_x", "var_bb_tr_y",
     "var_bb_bl_x", "var_bb_bl_y", "var_bb_br_x", "var_bb_br_y"],
    ["mean_left_ear", "mean_right_ear", "std_left_ear", "std_right_ear",
     "mean_left_iris_x", "mean_left_iris_y", "mean_right_iris_x", "mean_right_iris_y",
     "var_left_iris_x", "var_left_iris_y", "var_right_iris_x", "var_right_iris_y"],
    ["mean_left_ear", "mean_right_ear", "std_left_ear", "std_right_ear",
     "left_x_variance", "left_y_variance", "right_x_variance", "right_y_variance"],
]


def _detect_feature_cols(df):
    cols = set(df.columns)
    for candidates in _FEATURE_CANDIDATES:
        if all(c in cols for c in candidates):
            return candidates
    raise ValueError(f"unrecognised CSV schema. columns: {list(df.columns)}")


def run_logreg_csv(path, n_runs=N_RUNS, n_folds=N_FOLDS):
    df = pd.read_csv(path)
    cols = _detect_feature_cols(df)
    X = df[cols].values
    y = LabelEncoder().fit_transform(df["label"].values)

    run_accs, run_f1s = [], []
    for seed in range(n_runs):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_accs, fold_f1s = [], []
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=1000)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            pred = clf.predict(sc.transform(X[te]))
            fold_accs.append(accuracy_score(y[te], pred))
            fold_f1s.append(f1_score(y[te], pred, average="macro"))
        run_accs.append(np.mean(fold_accs))
        run_f1s.append(np.mean(fold_f1s))
    return np.array(run_accs), np.array(run_f1s)


def run_logreg_npz(path, n_runs=N_RUNS, n_folds=N_FOLDS):
    data = np.load(path, allow_pickle=True)
    X = data["X"].reshape(data["X"].shape[0], -1)
    y = LabelEncoder().fit_transform(data["y"])

    run_accs, run_f1s = [], []
    for seed in range(n_runs):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_accs, fold_f1s = [], []
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=1000)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            pred = clf.predict(sc.transform(X[te]))
            fold_accs.append(accuracy_score(y[te], pred))
            fold_f1s.append(f1_score(y[te], pred, average="macro"))
        run_accs.append(np.mean(fold_accs))
        run_f1s.append(np.mean(fold_f1s))
    return np.array(run_accs), np.array(run_f1s)


def run_cnn(path, n_runs=N_RUNS, n_folds=N_FOLDS):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = LabelEncoder().fit_transform(data["y"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_accs, run_f1s = [], []
    for seed in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_accs, fold_f1s = [], []
        for tr, te in skf.split(X, y):
            X_tr, X_te = X[tr], X[te]
            N_tr, T, C = X_tr.shape
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr.reshape(-1, C)).reshape(N_tr, T, C)
            X_te_s = sc.transform(X_te.reshape(-1, C)).reshape(X_te.shape[0], T, C)
            X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32).permute(0, 2, 1).to(device)
            X_te_t = torch.tensor(X_te_s, dtype=torch.float32).permute(0, 2, 1).to(device)
            y_tr_t = torch.tensor(y[tr], dtype=torch.long).to(device)
            y_te_t = torch.tensor(y[te], dtype=torch.long).to(device)
            model = cnn_time_model.train_one_fold(X_tr_t, y_tr_t, X_te_t, y_te_t, device)
            model.eval()
            with torch.no_grad():
                pred = torch.argmax(model(X_te_t), dim=1).cpu().numpy()
            fold_accs.append(accuracy_score(y[te], pred))
            fold_f1s.append(f1_score(y[te], pred, average="macro"))
        run_accs.append(np.mean(fold_accs))
        run_f1s.append(np.mean(fold_f1s))
    return np.array(run_accs), np.array(run_f1s)


def run_transformer(path, n_runs=N_RUNS, n_folds=N_FOLDS):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = LabelEncoder().fit_transform(data["y"])
    device = torch.device("cpu")

    run_accs, run_f1s = [], []
    for seed in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_accs, fold_f1s = [], []
        for tr, te in skf.split(X, y):
            X_tr, X_te = transformer_model.standardize_train_test(X[tr], X[te])
            acc, f1, _, _ = transformer_model.train_one_fold(X_tr, y[tr], X_te, y[te], device)
            fold_accs.append(acc)
            fold_f1s.append(f1)
        run_accs.append(np.mean(fold_accs))
        run_f1s.append(np.mean(fold_f1s))
    return np.array(run_accs), np.array(run_f1s)


def _fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def main():
    configs = [
        ("LogReg (CSV)",       run_logreg_csv, CSV_DATASETS),
        ("LogReg (Flattened)", run_logreg_npz, NPZ_DATASETS),
        ("TemporalCNN",        run_cnn,         NPZ_DATASETS),
        ("Transformer",        run_transformer, NPZ_DATASETS),
    ]

    rows = []
    for model_name, runner, datasets in configs:
        for dataset_name, path in datasets.items():
            if not os.path.exists(path):
                print(f"[skip] {path} not found")
                rows.append(dict(model=model_name, dataset=dataset_name,
                                 acc_mean=float("nan"), acc_std=float("nan"),
                                 f1_mean=float("nan"), f1_std=float("nan")))
                continue

            print(f"{model_name:22s} | {dataset_name:4s} | {N_RUNS} runs × {N_FOLDS} folds ...", end=" ", flush=True)
            accs, f1s = runner(path)
            rows.append(dict(model=model_name, dataset=dataset_name,
                             acc_mean=np.mean(accs), acc_std=np.std(accs),
                             f1_mean=np.mean(f1s), f1_std=np.std(f1s)))
            print(f"acc {_fmt(np.mean(accs), np.std(accs))}  f1 {_fmt(np.mean(f1s), np.std(f1s))}")

    col_w = 70
    print("\n" + "=" * col_w)
    print(f"{'Model':<22} {'Dataset':<8} {'Accuracy':>17} {'Macro F1':>17}")
    print("-" * col_w)
    prev_model = None
    for r in rows:
        if r["model"] != prev_model and prev_model is not None:
            print("-" * col_w)
        prev_model = r["model"]
        acc = _fmt(r["acc_mean"], r["acc_std"]) if not np.isnan(r["acc_mean"]) else "N/A"
        f1  = _fmt(r["f1_mean"],  r["f1_std"])  if not np.isnan(r["f1_mean"])  else "N/A"
        print(f"{r['model']:<22} {r['dataset']:<8} {acc:>17} {f1:>17}")
    print("=" * col_w)
    print(f"\n{N_RUNS} runs × {N_FOLDS}-fold CV  |  values are mean ± std across runs")


if __name__ == "__main__":
    main()
