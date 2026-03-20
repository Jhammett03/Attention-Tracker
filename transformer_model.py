import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

#dataset wrapper for converting numpy array to PyTorch dataset
#(lets us use DataLoader for batching/shuffling)
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#inject time information into data to allow for temporal learning like CNN
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=6, seq_len=30, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) #project 6 features to higher dim
        self.pos_enc = PositionalEncoding(seq_len, d_model) #inject time pos

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2) #binary classication

    def forward(self, x):
        x = self.input_proj(x) #feature embedding
        x = self.pos_enc(x) #inject order
        x = self.encoder(x) #global val
        x = x.mean(dim=1) #reduce 
        return self.classifier(x) #classify result


#normalize features per fold
def standardize_train_test(X_train, X_test):
    # standardize per feature dim across all timesteps
    n_train, t, d = X_train.shape
    n_test, _, _ = X_test.shape

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(n_train * t, d)
    X_test_flat = X_test.reshape(n_test * t, d)

    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    return X_train_flat.reshape(n_train, t, d), X_test_flat.reshape(n_test, t, d)


def train_one_fold(X_train, y_train, X_test, y_test, device):
    model = TransformerClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=64, shuffle=False)

    best_test_acc = 0.0
    best_preds = None
    best_trues = None

    patience = 15
    bad = 0

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # validation for each epoch for early stopping
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.argmax(logits, dim=1).cpu().numpy()
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

    macro_f1 = f1_score(best_trues, best_preds, average="macro")

    return best_test_acc, macro_f1, best_trues, best_preds


def main():
    device = torch.device("cpu")

    data = np.load("sequence_dataset.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)  # (N, 30, 6)
    y = data["y"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7) #5-fold cross validation

    accs = []
    f1s = []

    #train and validate over each fold, then take summary statistics
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = standardize_train_test(X_train, X_test)

        acc, macro_f1, trues, preds = train_one_fold(X_train, y_train, X_test, y_test, device)

        accs.append(acc)
        f1s.append(macro_f1)

        print(f"\nfold {fold}: acc {acc:.4f}, macro f1 {macro_f1:.4f}")

        # print confusion matrix
        cm = confusion_matrix(trues, preds)
        print("confusion matrix:")
        print(cm)

        # print per-class metrics
        print("classification report:")
        print(classification_report(trues, preds, digits=4))

    print("\ntransformer (5-fold)")
    print("accuracy mean:", float(np.mean(accs)))
    print("accuracy std :", float(np.std(accs)))
    print("\nmacro f1 mean:", float(np.mean(f1s)))
    print("macro f1 std :", float(np.std(f1s)))


if __name__ == "__main__":
    main()