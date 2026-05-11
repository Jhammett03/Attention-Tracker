import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=6, seq_len=150, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def standardize_train_test(X_train, X_test):
    n_train, t, d = X_train.shape
    n_test = X_test.shape[0]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(n_train * t, d)).reshape(n_train, t, d)
    X_test = scaler.transform(X_test.reshape(n_test * t, d)).reshape(n_test, t, d)
    return X_train, X_test


def train_one_fold(X_train, y_train, X_test, y_test, device):
    input_dim = X_train.shape[2]
    model = TransformerClassifier(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=64, shuffle=False)

    best_test_acc = 0.0
    best_preds = None
    best_trues = None
    patience = 15
    bad = 0

    for _ in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = torch.argmax(model(xb.to(device)), dim=1).cpu().numpy()
                preds.append(p)
                trues.append(yb.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        acc = accuracy_score(trues, preds)

        if acc > best_test_acc + 1e-4:
            best_test_acc = acc
            best_preds = preds.copy()
            best_trues = trues.copy()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_test_acc, f1_score(best_trues, best_preds, average="macro"), best_trues, best_preds


def build_predictor(npz_path, epochs=50):
    """Train on all data and return (model, scaler, le, device) ready for inference."""
    device = torch.device("cpu")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_raw = data["y"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    N, T, D = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(N * T, D)).reshape(N, T, D)

    model = TransformerClassifier(input_dim=D, seq_len=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(SeqDataset(X_scaled, y), batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}")

    model.eval()
    print(f"  trained on {N} samples — classes: {list(le.classes_)}")
    return model, scaler, le, device


def save_model(npz_path="sequence_dataset_gaze.npz", output="transformer_model.pt"):
    """Train on all data and save to disk for use in realtime.py."""
    model, scaler, le, _ = build_predictor(npz_path)
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": model.input_proj.in_features,
        "seq_len": model.pos_enc.pos_embedding.shape[1],
        "scaler": scaler,
        "le": le,
    }, output)
    print(f"saved transformer → {output}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="train on all data and save model")
    parser.add_argument("--npz", default="sequence_dataset_gaze.npz")
    args = parser.parse_args()

    if args.save:
        save_model(npz_path=args.npz)
        return

    device = torch.device("cpu")

    data = np.load("sequence_dataset_gaze.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = LabelEncoder().fit_transform(data["y"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    accs = []
    f1s = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = standardize_train_test(X_train, X_test)
        acc, macro_f1, trues, preds = train_one_fold(X_train, y_train, X_test, y_test, device)

        accs.append(acc)
        f1s.append(macro_f1)

        print(f"\nfold {fold}: acc {acc:.4f}, macro f1 {macro_f1:.4f}")
        print(confusion_matrix(trues, preds))
        print(classification_report(trues, preds, digits=4))

    print("\ntransformer (5-fold)")
    print("accuracy mean:", float(np.mean(accs)))
    print("accuracy std :", float(np.std(accs)))
    print("macro f1 mean:", float(np.mean(f1s)))
    print("macro f1 std :", float(np.std(f1s)))


if __name__ == "__main__":
    main()
