"""
client.py — one Flower NumPyClient per simulated institution.

Each client only ever sees its own partition's features/labels. It receives the
current global weights, trains locally, and returns updated weights + how many
samples it trained on (used for FedAvg's weighted average) -- raw data never
leaves this object.
"""
import numpy as np
import flwr as fl

from sklearn.metrics import f1_score, roc_auc_score 
from app.fl_experiment.model import NumpyLogisticRegression


class RiskClient(fl.client.NumPyClient):
    def __init__(self, institution_id, X_train, y_train, X_test, y_test, n_features):
        self.institution_id = institution_id
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.model = NumpyLogisticRegression(n_features)

    def get_parameters(self, config=None):
        return self.model.get_parameters()

    def fit(self, parameters, config=None):
        self.model.set_parameters(parameters)
        n = self.model.local_train(
            self.X_train, self.y_train,
            epochs=config.get("local_epochs", 5) if config else 5,
            lr=config.get("lr", 0.1) if config else 0.1,
        )
        return self.model.get_parameters(), n, {"institution_id": self.institution_id}

    def evaluate(self, parameters, config=None):
        self.model.set_parameters(parameters)
        acc, bce, preds, proba = self.model.evaluate(self.X_test, self.y_test)
        f1 = float(f1_score(self.y_test, preds))
        roc_auc = float(roc_auc_score(self.y_test, proba))
        return bce, len(self.X_test), {"accuracy": acc, "f1": f1, "roc_auc": roc_auc, "institution_id": self.institution_id}
