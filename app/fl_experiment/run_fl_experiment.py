"""
run_fl_experiment.py — the Federated Learning demo (Section 11).

Institutions are simulated as isolated data partitions on a single machine using
the Flower framework's NumPyClient abstraction; this demonstrates the federated
training and aggregation mechanism, not a production multi-site deployment.

Design note on *how* rounds are driven: this script calls each institution's
Flower NumPyClient.fit()/evaluate() directly in a manual FedAvg loop, rather than
going through flwr.simulation.start_simulation() (which pulls in Ray). This keeps
the experiment dependency-light and reliable on an ordinary laptop while still
using genuine Flower Client objects, real local-only training, and real weighted
weight aggregation -- the actual FedAvg mechanism your report needs to
demonstrate. If you later want the full Ray-based simulation runtime, client.py's
RiskClient is already a drop-in fl.client.NumPyClient for that.

Run:  python -m app.fl_experiment.run_fl_experiment
Output: db_files/fl_results.json, db_files/fl_accuracy_chart.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.data.oulad_prep import load_features, FEATURE_COLS, INSTITUTIONS
from app.fl_experiment.client import RiskClient
from app.fl_experiment.model import NumpyLogisticRegression, fedavg
from sklearn.metrics import f1_score, roc_auc_score 

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "db_files")

N_ROUNDS = 15
LOCAL_EPOCHS = 5
LR = 0.15
SEED = 42


def build_clients():
    df = load_features()
    scaler = StandardScaler().fit(df[FEATURE_COLS].values)

    clients = []
    for inst in INSTITUTIONS:
        part = df[df["institution_id"] == inst]
        X = scaler.transform(part[FEATURE_COLS].values)
        y = part["label"].values.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        clients.append(RiskClient(inst, X_train, y_train, X_test, y_test, n_features=len(FEATURE_COLS)))
    return clients, scaler, df


def run_federated(clients):
    global_model = NumpyLogisticRegression(n_features=len(FEATURE_COLS), seed=SEED)
    history = []

    for rnd in range(1, N_ROUNDS + 1):
        global_params = global_model.get_parameters()

        # --- each institution trains locally on its own partition only ---
        client_params, sample_counts = [], []
        for c in clients:
            new_params, n_samples, _meta = c.fit(global_params, {"local_epochs": LOCAL_EPOCHS, "lr": LR})
            client_params.append(new_params)
            sample_counts.append(n_samples)

        # --- server-side FedAvg: weighted average of client weights ---
        aggregated = fedavg(client_params, sample_counts)
        global_model.set_parameters(aggregated)

        # --- evaluate the newly aggregated global model on each client's held-out test set ---
        round_accs = []
        round_f1s = []
        round_roc_aucs = []

        for c in clients:
            _loss, _n, metrics = c.evaluate(global_model.get_parameters())

            round_accs.append(metrics["accuracy"])
            round_f1s.append(metrics["f1"])
            round_roc_aucs.append(metrics["roc_auc"])

        avg_acc = float(np.mean(round_accs))
        avg_f1 = float(np.mean(round_f1s))
        avg_roc_auc = float(np.mean(round_roc_aucs))

        history.append({
            "round": rnd,
            "avg_accuracy": avg_acc,
            "avg_f1": avg_f1,
            "avg_roc_auc": avg_roc_auc,
        })

        print(f"Round {rnd:2d}: accuracy={avg_acc:.4f}, F1={avg_f1:.4f}, ROC-AUC={avg_roc_auc:.4f}")

    return global_model, history


def run_centralized_baseline(df, scaler):
    """Same tiny model, but trained on ALL institutions' data pooled together --
    the comparison point that shows what federation gives up (a little accuracy,
    typically) in exchange for not centralising raw records."""
    X = scaler.transform(df[FEATURE_COLS].values)
    y = df["label"].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    model = NumpyLogisticRegression(n_features=len(FEATURE_COLS), seed=SEED)
    history = []
    for rnd in range(1, N_ROUNDS + 1):
        model.local_train(X_train, y_train, epochs=LOCAL_EPOCHS, lr=LR, seed=rnd)
        acc, _bce, preds, proba = model.evaluate(X_test, y_test)

        f1 = float(f1_score(y_test, preds))
        roc_auc = float(roc_auc_score(y_test, proba))

        history.append({
            "round": rnd,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc_auc
        })
    return model, history


def main():
    clients, scaler, df = build_clients()
    fed_model, fed_history = run_federated(clients)
    _, central_history = run_centralized_baseline(df, scaler)

    results = {
        "n_institutions": len(INSTITUTIONS),
        "institutions": INSTITUTIONS,
        "n_rounds": N_ROUNDS,
        "federated_history": fed_history,
        "centralized_history": central_history,
        "final_federated_accuracy": fed_history[-1]["avg_accuracy"],
        "final_federated_f1": fed_history[-1]["avg_f1"],
        "final_federated_roc_auc": fed_history[-1]["avg_roc_auc"],
        "final_centralized_accuracy": central_history[-1]["accuracy"],
        "final_centralized_f1": central_history[-1]["f1"],
        "final_centralized_roc_auc": central_history[-1]["roc_auc"],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "fl_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- the FL evidence chart: federated vs centralized accuracy per round ---
    plt.figure(figsize=(8, 5))
    plt.plot([h["round"] for h in fed_history], [h["avg_accuracy"] for h in fed_history],
              marker="o", label="Federated (FedAvg, 4 institutions)")
    plt.plot([h["round"] for h in central_history], [h["accuracy"] for h in central_history],
              marker="s", label="Centralized baseline (pooled data)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (held-out test set)")
    plt.title("Federated vs Centralized Risk-Prediction Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "fl_accuracy_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nFinal federated accuracy:   {results['final_federated_accuracy']:.4f}")
    print(f"Final federated F1:         {results['final_federated_f1']:.4f}")
    print(f"Final federated ROC-AUC:    {results['final_federated_roc_auc']:.4f}")
    print(f"\nFinal centralized accuracy: {results['final_centralized_accuracy']:.4f}")
    print(f"Final centralized F1:       {results['final_centralized_f1']:.4f}")
    print(f"Final centralized ROC-AUC:  {results['final_centralized_roc_auc']:.4f}")
    print(f"Saved: {chart_path}")

if __name__ == "__main__":
    main()
