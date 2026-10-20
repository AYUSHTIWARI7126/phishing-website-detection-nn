"""
Phishing Website Detection using Neural Computation techniques.

Trains and compares: Perceptron, ADALINE, MLP (back-prop), RBF network and
Kohonen SOM - all implemented from scratch in nn_models.py - plus an
activation-function experiment and a scikit-learn MLP as a sanity baseline.

usage:
    python train.py                       # synthetic URL dataset
    python train.py --csv data.csv --label-col Result   # your own dataset
"""
import argparse
import os
import time

import numpy as np

from dataset import generate_synthetic, load_csv, save_csv
from features import FEATURE_NAMES
from nn_models import MLP, RBFNetwork, SOM, Adaline, Perceptron, StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

OUT = "results"


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "cm": np.array([[tn, fp], [fn, tp]])}


def split(X, y, val=0.15, test=0.15, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    n_te, n_va = int(len(X) * test), int(len(X) * val)
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


# ---------------------------------------------------------------- plots
def plot_all(models, results, mlp_act_runs, som, feature_names, X_te, y_te):
    os.makedirs(OUT, exist_ok=True)

    # 1. learning curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(models["Perceptron"].history)
    ax[0].set(title="Perceptron: misclassifications / epoch", xlabel="epoch")
    ax[1].plot(models["ADALINE"].history)
    ax[1].set(title="ADALINE: MSE / epoch", xlabel="epoch")
    h = models["MLP (ReLU)"].history
    ax[2].plot(h["train_loss"], label="train")
    ax[2].plot(h["val_loss"], label="validation")
    ax[2].set(title="MLP: cross-entropy loss", xlabel="epoch")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/learning_curves.png", dpi=120)

    # 2. activation function comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in mlp_act_runs.items():
        ax.plot(m.history["val_loss"], label=name)
    ax.set(title="MLP validation loss vs activation function", xlabel="epoch",
           ylabel="BCE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/activation_comparison.png", dpi=120)

    # 3. model comparison bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(results)
    x = np.arange(len(names))
    for k, met in enumerate(["accuracy", "precision", "recall", "f1"]):
        ax.bar(x + k * 0.2 - 0.3, [results[n][met] for n in names], 0.2, label=met)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Test-set performance")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison.png", dpi=120)

    # 4. confusion matrix of the MLP
    cm = results["MLP (ReLU)"]["cm"]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["legit", "phish"])
    ax.set_yticks([0, 1], ["legit", "phish"])
    ax.set(xlabel="predicted", ylabel="actual", title="MLP confusion matrix")
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=120)

    # 5. SOM: U-matrix and phishing-fraction map
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    im = ax[0].imshow(som.u_matrix(), cmap="bone_r")
    ax[0].set_title("SOM U-matrix (cluster borders)")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(som.phish_frac.reshape(som.rows, som.cols), cmap="coolwarm", vmin=0, vmax=1)
    ax[1].set_title("SOM: fraction of phishing URLs per neuron")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(f"{OUT}/som_maps.png", dpi=120)

    # 6. feature importance (permutation importance on MLP)
    mlp = models["MLP (ReLU)"]
    base = np.mean(mlp.predict(X_te) == y_te)
    rng = np.random.default_rng(0)
    drops = []
    for j in range(X_te.shape[1]):
        Xp = X_te.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(base - np.mean(mlp.predict(Xp) == y_te))
    order = np.argsort(drops)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feature_names[i] for i in order], np.array(drops)[order])
    ax.set(title="MLP permutation feature importance", xlabel="accuracy drop")
    fig.tight_layout()
    fig.savefig(f"{OUT}/feature_importance.png", dpi=120)
    plt.close("all")
    print(f"\nPlots saved in ./{OUT}/")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="optional real dataset CSV")
    ap.add_argument("--label-col", help="label column name (default: last column)")
    ap.add_argument("--n", type=int, default=4000, help="synthetic dataset size")
    args = ap.parse_args()

    if args.csv:
        X, y, names = load_csv(args.csv, args.label_col)
        print(f"Loaded {args.csv}: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        X, y, urls = generate_synthetic(args.n)
        names = FEATURE_NAMES
        save_csv("phishing_dataset.csv", X, y, urls)
        print(f"Synthetic dataset: {X.shape[0]} URLs, {X.shape[1]} features "
              f"(saved to phishing_dataset.csv)")
    print(f"Class balance: {int(y.sum())} phishing / {int(len(y) - y.sum())} legitimate")

    X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y)
    scaler = StandardScaler().fit(X_tr)          # fit on training data only
    X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
    d = X.shape[1]

    models = {
        "Perceptron": Perceptron(lr=0.01, epochs=50),
        "ADALINE": Adaline(lr=0.0005, epochs=50),
        "MLP (ReLU)": MLP([d, 32, 16, 1], activation="relu", lr=0.01, epochs=300),
        "RBF network": RBFNetwork(n_centers=60),
        "Kohonen SOM": SOM(rows=10, cols=10, epochs=15),
    }

    results = {}
    print("\nTraining models...")
    for name, model in models.items():
        t0 = time.time()
        print(f"  -> {name}")
        if isinstance(model, MLP):
            model.fit(X_tr, y_tr, X_va, y_va, verbose=True)
        else:
            model.fit(X_tr, y_tr)
        results[name] = metrics(y_te, model.predict(X_te))
        results[name]["time"] = time.time() - t0

    # experiment: effect of the hidden-layer activation function
    print("\nActivation-function experiment (MLP 32-16, same settings):")
    act_runs = {}
    for act in ["sigmoid", "tanh", "relu"]:
        m = MLP([d, 32, 16, 1], activation=act, lr=0.01 if act == "relu" else 0.05,
                epochs=300).fit(X_tr, y_tr, X_va, y_va)
        act_runs[act] = m
        r = metrics(y_te, m.predict(X_te))
        print(f"  {act:8s} test acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  "
              f"epochs={len(m.history['val_loss'])}")

    # experiment: number of hidden neurons
    print("\nHidden-layer size experiment (single hidden layer, ReLU):")
    for h in [2, 4, 8, 16, 64]:
        m = MLP([d, h, 1], activation="relu", lr=0.01, epochs=300).fit(X_tr, y_tr, X_va, y_va)
        r = metrics(y_te, m.predict(X_te))
        print(f"  hidden={h:3d}  test acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")

    # baseline: scikit-learn MLP to check our from-scratch implementation
    try:
        from sklearn.neural_network import MLPClassifier
        sk = MLPClassifier((32, 16), max_iter=500, early_stopping=True, random_state=0)
        sk.fit(X_tr, y_tr)
        results["sklearn MLP"] = metrics(y_te, sk.predict(X_te))
        results["sklearn MLP"]["time"] = 0.0
    except ImportError:
        pass

    print("\n" + "=" * 66)
    print(f"{'Model':<16}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Time(s)':>10}")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:<16}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
              f"{r['recall']:>9.4f}{r['f1']:>8.4f}{r['time']:>10.2f}")
    print("=" * 66)
    cm = results["MLP (ReLU)"]["cm"]
    print(f"MLP confusion matrix  [[TN FP] [FN TP]] = {cm.tolist()}")

    os.makedirs(OUT, exist_ok=True)
    models["MLP (ReLU)"].save(f"{OUT}/mlp_model.npz", scaler, names)
    print(f"Best MLP saved to {OUT}/mlp_model.npz  (use predict.py)")

    if plt is not None:
        plot_all(models, results, act_runs, models["Kohonen SOM"], names, X_te, y_te)
    else:
        print("\n(matplotlib not installed - skipping plots: pip install matplotlib)")


if __name__ == "__main__":
    main()
