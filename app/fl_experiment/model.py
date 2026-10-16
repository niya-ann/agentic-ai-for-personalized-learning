"""
model.py — a tiny logistic regression implemented in plain NumPy.

Why not scikit-learn's LogisticRegression here: FedAvg needs to read and write
raw weight vectors every round (get_parameters / set_parameters), which is the
one thing sklearn's fit() doesn't expose cleanly for incremental/federated
training. A two-line linear model keeps this fully transparent and keeps a
round's local training fast (milliseconds) on CPU.
"""
import numpy as np


class NumpyLogisticRegression:
    def __init__(self, n_features, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0

    def get_parameters(self):
        return [self.w.copy(), np.array([self.b])]

    def set_parameters(self, params):
        self.w = params[0].copy()
        self.b = float(params[1][0])

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def local_train(self, X, y, epochs=5, lr=0.1, batch_size=256, seed=0):
        """One local training round (a few epochs of mini-batch gradient descent
        on this client's own partition only -- no other client's data is ever
        seen here)."""
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                xb, yb = X[batch_idx], y[batch_idx]
                preds = self.predict_proba(xb)
                error = preds - yb
                grad_w = xb.T @ error / len(xb) + 0.001 * self.w  # small L2
                grad_b = error.mean()
                self.w -= lr * grad_w
                self.b -= lr * grad_b
        return len(X)

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        acc = float((preds == y).mean())
        eps = 1e-12
        bce = float(-np.mean(y * np.log(proba + eps) + (1 - y) * np.log(1 - proba + eps)))
        return acc, bce, preds, proba


def fedavg(param_list, sample_counts):
    """Weighted average of client weight vectors, weighted by each client's
    local dataset size -- the standard FedAvg aggregation rule."""
    total = sum(sample_counts)
    avg_w = sum(p[0] * n for p, n in zip(param_list, sample_counts)) / total
    avg_b = sum(p[1] * n for p, n in zip(param_list, sample_counts)) / total
    return [avg_w, avg_b]
