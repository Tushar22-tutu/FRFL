"""Run a reproducible federated experiment over precomputed face embeddings."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from frfl.aggregation import drift_weighted_average, l2_parameter_drift, sample_weighted_average
from frfl.data import dirichlet_partition_indices, load_embeddings, split_embeddings
from frfl.metrics import classification_metrics
from frfl.model import EmbeddingClassifier
from frfl.trainer import train_local_model

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--x", default="data/features/X.npy")
    p.add_argument("--y", default="data/features/y.npy")
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=0.01, help="FedProx coefficient; 0 disables proximal penalty")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration for non-IID clients")
    p.add_argument("--temperature", type=float, default=1.0, help="Drift weighting temperature")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/federated_metrics.json")
    return p.parse_args()

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(X, dtype=torch.float32)).argmax(dim=1).cpu().numpy()
    return classification_metrics(y, pred)

def main():
    args = parse_args()
    if args.clients < 2 or args.rounds < 1 or args.local_epochs < 1:
        raise SystemExit("clients must be >=2; rounds and local-epochs must be >=1")
    if args.lr <= 0 or args.mu < 0 or args.temperature < 0:
        raise SystemExit("lr must be positive; mu and temperature must be non-negative")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    X, y, encoder = load_embeddings(args.x, args.y)
    X_train, X_test, y_train, y_test = split_embeddings(X, y, args.test_size, args.seed)
    partitions = dirichlet_partition_indices(y_train, args.clients, args.alpha, args.seed)
    # Maintain independent global models for a fair same-split aggregation comparison.
    drift_model = EmbeddingClassifier(X.shape[1], len(encoder.classes_))
    fedavg_model = EmbeddingClassifier(X.shape[1], len(encoder.classes_))
    fedavg_model.load_state_dict(drift_model.state_dict())
    history = []
    for round_no in range(1, args.rounds + 1):
        round_results = {}
        for method, global_model in (("drift_weighted", drift_model), ("fedavg", fedavg_model)):
            reference = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
            local_states, counts, losses = [], [], []
            for client_no, ids in enumerate(partitions):
                state, count, loss = train_local_model(
                    global_model, X_train[ids], y_train[ids],
                    epochs=args.local_epochs, batch_size=args.batch_size,
                    learning_rate=args.lr, mu=args.mu,
                    seed=args.seed + round_no * 100 + client_no,
                )
                local_states.append(state); counts.append(count); losses.append(loss)
            if method == "drift_weighted":
                aggregate, weights = drift_weighted_average(reference, local_states, args.temperature)
                drifts = [l2_parameter_drift(reference, state) for state in local_states]
            else:
                aggregate = sample_weighted_average(local_states, counts)
                weights = [count / sum(counts) for count in counts]
                drifts = [l2_parameter_drift(reference, state) for state in local_states]
            global_model.load_state_dict(aggregate)
            round_results[method] = {
                "client_sample_counts": counts,
                "client_mean_losses": losses,
                "client_l2_drifts": drifts,
                "aggregation_weights": weights,
                "heldout_metrics": evaluate(global_model, X_test, y_test),
            }
        history.append({"round": round_no, "methods": round_results})
        for method, values in round_results.items():
            m = values["heldout_metrics"]
            print(f"round={round_no} method={method} accuracy={m['accuracy']:.4f} macro_f1={m['f1_macro']:.4f}")
    report = {
        "project": "FRFL",
        "warning": "Metrics are from this exact run and split. They are not prefilled claims; inspect dataset, identity split, and sample counts before reporting.",
        "configuration": vars(args),
        "dataset": {"samples": int(len(X)), "embedding_dim": int(X.shape[1]), "classes": int(len(encoder.classes_)),
                    "train_samples": int(len(X_train)), "test_samples": int(len(X_test))},
        "history": history,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report: {out}")

if __name__ == "__main__":
    main()
