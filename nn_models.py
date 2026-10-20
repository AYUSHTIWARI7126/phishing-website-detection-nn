"""
Neural network models implemented FROM SCRATCH with NumPy.

NCA syllabus topics covered:
  1. Perceptron          - Rosenblatt learning rule, step activation
  2. ADALINE             - Widrow-Hoff / delta rule, gradient descent on MSE
  3. Multi-Layer Perceptron (MLP)
                         - forward propagation, back-propagation,
                           activation functions (sigmoid / tanh / ReLU),
                           binary cross-entropy loss, mini-batch SGD,
                           momentum, L2 regularisation (weight decay),
                           early stopping, Xavier/He weight initialisation
  4. RBF network         - k-means centres, Gaussian basis functions,
                           output weights by (regularised) least squares
  5. Kohonen SOM         - competitive / unsupervised learning,
                           neighbourhood function, decaying learning rate
"""
import numpy as np


# ---------------------------------------------------------------- utilities
class StandardScaler:
    """z-score normalisation: x' = (x - mean) / std"""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


ACTIVATIONS = {
    #  name : (f(z),                        f'(z) expressed via a = f(z) and z)
    "sigmoid": (sigmoid,                        lambda a, z: a * (1 - a)),
    "tanh":    (np.tanh,                        lambda a, z: 1 - a ** 2),
    "relu":    (lambda z: np.maximum(0, z),     lambda a, z: (z > 0).astype(float)),
}


# ---------------------------------------------------------------- 1. Perceptron
class Perceptron:
    """Single neuron, hard-limit (step) activation.
    Learning rule:  w <- w + eta * (t - y) * x
    Converges only if the data is linearly separable."""

    def __init__(self, lr=0.01, epochs=50, seed=0):
        self.lr, self.epochs = lr, epochs
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.w = self.rng.normal(0, 0.01, X.shape[1])
        self.b = 0.0
        self.history = []  # number of misclassifications per epoch
        for _ in range(self.epochs):
            errors = 0
            for i in self.rng.permutation(len(X)):
                pred = 1.0 if X[i] @ self.w + self.b >= 0 else 0.0
                update = self.lr * (y[i] - pred)
                if update != 0:
                    self.w += update * X[i]
                    self.b += update
                    errors += 1
            self.history.append(errors)
        return self

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(float)


# ---------------------------------------------------------------- 2. ADALINE
class Adaline:
    """Adaptive Linear Neuron. Trained on the *linear* output with the
    delta rule (LMS):  w <- w + eta * (t - net) * x ,  targets in {-1,+1}."""

    def __init__(self, lr=0.001, epochs=50, seed=0):
        self.lr, self.epochs = lr, epochs
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        t = np.where(y == 1, 1.0, -1.0)
        self.w = self.rng.normal(0, 0.01, X.shape[1])
        self.b = 0.0
        self.history = []  # MSE per epoch
        for _ in range(self.epochs):
            for i in self.rng.permutation(len(X)):
                err = t[i] - (X[i] @ self.w + self.b)
                self.w += self.lr * err * X[i]
                self.b += self.lr * err
            self.history.append(np.mean((t - (X @ self.w + self.b)) ** 2))
        return self

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(float)


# ---------------------------------------------------------------- 3. MLP
class MLP:
    """Fully-connected feed-forward network trained with back-propagation.

    layers      e.g. [20, 16, 8, 1]  (input, hidden..., output)
    hidden act  sigmoid | tanh | relu
    output      sigmoid neuron  -> P(phishing)
    loss        binary cross-entropy  L = -[t log y + (1-t) log(1-y)]
    optimiser   mini-batch gradient descent with momentum
                v <- beta*v - eta*dL/dW ;  W <- W + v
    """

    def __init__(self, layers, activation="relu", lr=0.05, momentum=0.9,
                 l2=1e-4, epochs=200, batch_size=32, patience=20, seed=0):
        self.layers = layers
        self.act_name = activation
        self.act, self.dact = ACTIVATIONS[activation]
        self.lr, self.momentum, self.l2 = lr, momentum, l2
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.rng = np.random.default_rng(seed)
        self._init_weights()

    def _init_weights(self):
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layers[:-1], self.layers[1:]):
            # He init for ReLU, Xavier/Glorot init for sigmoid/tanh
            scale = np.sqrt(2.0 / n_in) if self.act_name == "relu" else np.sqrt(1.0 / n_in)
            self.W.append(self.rng.normal(0, scale, (n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ---- forward pass: store every z (net input) and a (activation)
    def forward(self, X):
        a, zs, activations = X, [], [X]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = sigmoid(z) if l == len(self.W) - 1 else self.act(z)
            zs.append(z)
            activations.append(a)
        return zs, activations

    # ---- backward pass: chain rule, layer by layer from output to input
    def backward(self, zs, activations, t):
        m = len(t)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        # sigmoid output + cross-entropy => delta = y - t  (nice simplification)
        delta = activations[-1] - t.reshape(-1, 1)
        for l in reversed(range(len(self.W))):
            gW[l] = activations[l].T @ delta / m + self.l2 * self.W[l]
            gb[l] = delta.mean(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * self.dact(activations[l], zs[l - 1])
        return gW, gb

    def _step(self, gW, gb):
        for l in range(len(self.W)):
            self.vW[l] = self.momentum * self.vW[l] - self.lr * gW[l]
            self.vb[l] = self.momentum * self.vb[l] - self.lr * gb[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]

    @staticmethod
    def bce(p, t):
        p = np.clip(p.ravel(), 1e-12, 1 - 1e-12)
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_loss, best_params, wait = np.inf, None, 0
        for epoch in range(self.epochs):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), self.batch_size):
                bi = idx[s:s + self.batch_size]
                zs, acts = self.forward(X[bi])
                self._step(*self.backward(zs, acts, y[bi]))

            self.history["train_loss"].append(self.bce(self.predict_proba(X), y))
            if X_val is not None:
                vl = self.bce(self.predict_proba(X_val), y_val)
                self.history["val_loss"].append(vl)
                self.history["val_acc"].append(np.mean(self.predict(X_val) == y_val))
                # early stopping on validation loss
                if vl < best_loss - 1e-5:
                    best_loss, wait = vl, 0
                    best_params = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                else:
                    wait += 1
                    if wait >= self.patience:
                        if verbose:
                            print(f"    early stopping at epoch {epoch + 1}")
                        break
            if verbose and (epoch + 1) % 20 == 0:
                msg = f"    epoch {epoch + 1:4d}  train_loss={self.history['train_loss'][-1]:.4f}"
                if X_val is not None:
                    msg += f"  val_loss={vl:.4f}  val_acc={self.history['val_acc'][-1]:.4f}"
                print(msg)
        if best_params is not None:
            self.W, self.b = best_params
        return self

    def predict_proba(self, X):
        return self.forward(X)[1][-1].ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(float)

    def save(self, path, scaler, feature_names):
        np.savez(path, layers=np.array(self.layers), act=self.act_name,
                 mean=scaler.mean, std=scaler.std, names=np.array(feature_names),
                 **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        model = cls(list(d["layers"]), activation=str(d["act"]))
        n = len(model.layers) - 1
        model.W = [d[f"W{i}"] for i in range(n)]
        model.b = [d[f"b{i}"] for i in range(n)]
        scaler = StandardScaler()
        scaler.mean, scaler.std = d["mean"], d["std"]
        return model, scaler, list(d["names"])


# ---------------------------------------------------------------- 4. RBF network
class RBFNetwork:
    """Radial Basis Function network.
    Hidden layer : phi_j(x) = exp(-||x - c_j||^2 / (2 sigma^2))
    Centres c_j  : found by k-means clustering (unsupervised stage)
    Output layer : linear, weights solved by ridge least squares
                   w = (Phi^T Phi + lambda I)^-1 Phi^T t"""

    def __init__(self, n_centers=40, reg=1e-3, kmeans_iters=50, seed=0):
        self.k, self.reg, self.iters = n_centers, reg, kmeans_iters
        self.rng = np.random.default_rng(seed)

    def _kmeans(self, X):
        C = X[self.rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.iters):
            assign = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j]
                             for j in range(self.k)])
            if np.allclose(newC, C):
                break
            C = newC
        return C

    def _phi(self, X):
        d2 = ((X[:, None] - self.C[None]) ** 2).sum(-1)
        Phi = np.exp(-d2 / (2 * self.sigma ** 2))
        return np.hstack([Phi, np.ones((len(X), 1))])  # bias column

    def fit(self, X, y):
        self.C = self._kmeans(X)
        # common width heuristic: sigma = d_max / sqrt(2k)
        dmax = np.max(np.linalg.norm(self.C[:, None] - self.C[None], axis=-1))
        self.sigma = dmax / np.sqrt(2 * self.k)
        Phi = self._phi(X)
        A = Phi.T @ Phi + self.reg * np.eye(Phi.shape[1])
        self.w = np.linalg.solve(A, Phi.T @ y)
        return self

    def predict_proba(self, X):
        return np.clip(self._phi(X) @ self.w, 0, 1)

    def predict(self, X):
        return (self._phi(X) @ self.w >= 0.5).astype(float)


# ---------------------------------------------------------------- 5. Kohonen SOM
class SOM:
    """Self-Organising Map (unsupervised, competitive learning).
    For each sample: find Best Matching Unit (BMU), then
        w_j <- w_j + eta(t) * h_j(t) * (x - w_j)
        h_j(t) = exp(-dist_grid(j, BMU)^2 / (2 sigma(t)^2))
    eta and sigma decay exponentially over time.
    After training, each neuron is labelled by majority vote of the training
    samples mapped to it -> the SOM can then also be used as a classifier."""

    def __init__(self, rows=10, cols=10, lr=0.5, sigma=None, epochs=20, seed=0):
        self.rows, self.cols, self.lr0, self.epochs = rows, cols, lr, epochs
        self.sigma0 = sigma or max(rows, cols) / 2
        self.rng = np.random.default_rng(seed)
        self.grid = np.array([[i, j] for i in range(rows) for j in range(cols)], float)

    def bmu(self, X):
        d = ((X[:, None] - self.W[None]) ** 2).sum(-1)
        return np.argmin(d, axis=1)

    def fit(self, X, y=None):
        self.W = X[self.rng.choice(len(X), self.rows * self.cols)].copy()
        T = self.epochs * len(X)
        tau = T / np.log(self.sigma0)
        t = 0
        self.history = []  # quantisation error per epoch
        for _ in range(self.epochs):
            for i in self.rng.permutation(len(X)):
                lr = self.lr0 * np.exp(-t / T)
                sig = self.sigma0 * np.exp(-t / tau)
                b = np.argmin(((self.W - X[i]) ** 2).sum(1))
                h = np.exp(-((self.grid - self.grid[b]) ** 2).sum(1) / (2 * sig ** 2))
                self.W += lr * h[:, None] * (X[i] - self.W)
                t += 1
            self.history.append(self.quantization_error(X))
        if y is not None:
            self._label_neurons(X, y)
        return self

    def quantization_error(self, X):
        return np.mean(np.linalg.norm(X - self.W[self.bmu(X)], axis=1))

    def _label_neurons(self, X, y):
        b = self.bmu(X)
        n = self.rows * self.cols
        self.hits = np.bincount(b, minlength=n)
        self.phish_frac = np.bincount(b, weights=y, minlength=n) / np.maximum(self.hits, 1)
        default = float(y.mean() >= 0.5)
        self.labels = np.where(self.hits > 0, (self.phish_frac >= 0.5).astype(float), default)

    def predict(self, X):
        return self.labels[self.bmu(X)]

    def u_matrix(self):
        """Average distance of each neuron to its grid neighbours (cluster borders)."""
        W = self.W.reshape(self.rows, self.cols, -1)
        U = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                nb = [W[a, b] for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                      if 0 <= a < self.rows and 0 <= b < self.cols]
                U[i, j] = np.mean([np.linalg.norm(W[i, j] - v) for v in nb])
        return U
