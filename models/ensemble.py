"""
Ensemble aggregator for the sharded models.

Based on the SISA framework (Bourtoule et al., 2019):
Each isolated shard model outputs class probabilities on a query sample.
The ensemble averages these probabilities to make the final prediction.

Because each shard is trained on disjoint data, withdrawing a record
only requires retraining the single shard containing that record. The other
shards remain untouched and valid.
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


class ShardedEnsemble:
    def __init__(self, models=None):
        """
        models: dict of {shard_id: sklearn_model}
        """
        self.models = models if models else {}

    def set_shard_model(self, shard_id, model):
        """Swap in a retrained model for one shard."""
        self.models[shard_id] = model

    def predict_proba(self, X, **kwargs):
        """Average probabilities across all active shards."""
        if len(self.models) == 0:
            raise RuntimeError("No models loaded — train the shards first.")

        all_probas = []
        for sid in sorted(self.models):
            p = self.models[sid].predict_proba(X)
            all_probas.append(p)

        avg = np.mean(all_probas, axis=0)
        return avg

    def predict(self, X, **kwargs):
        """Class prediction by argmax over ensemble probabilities."""
        return np.argmax(self.predict_proba(X, **kwargs), axis=1)

    def evaluate(self, X, y, label="Ensemble"):
        """Quick evaluation — returns accuracy, AUC, and per-shard metrics."""
        preds = self.predict(X)
        proba = self.predict_proba(X)

        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, proba[:, 1])
        except ValueError:
            auc = float("nan")

        print(f"\n{label} — acc: {acc:.4f}, auc: {auc:.4f} ({len(self.models)} shards)")

        shard_accs = {}
        for sid in sorted(self.models):
            sp = self.models[sid].predict(X)
            shard_accs[sid] = accuracy_score(y, sp)
            print(f"  shard {sid}: {shard_accs[sid]:.4f}")

        return {"accuracy": acc, "auc": auc, "shard_accuracies": shard_accs}

    def per_sample_loss(self, X, y):
        """
        Cross-entropy loss per sample across the ensemble.
        Low loss = model is highly confident = potential memorization.
        This provides the metric for the membership inference attack.
        """
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        losses = -np.log(proba[np.arange(len(y)), y.astype(int)])
        return losses

    def shard_loss(self, shard_id, X, y):
        """Per-sample cross-entropy loss from one specific shard model."""
        model = self.models[shard_id]
        proba = model.predict_proba(X)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        return -np.log(proba[np.arange(len(y)), y.astype(int)])
