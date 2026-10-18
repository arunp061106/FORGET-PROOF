"""
Membership Inference Attack (MIA) — proof that forgetting works.

We use the simplest effective approach: the loss-threshold attack
(Yeom et al. 2018). The idea is dead simple:

  If the model's loss on a sample is below some threshold,
  we guess "this sample was in the training set."

Why this works: models tend to be more confident (lower loss) on data
they trained on vs data they haven't seen. After we forget a record
and retrain, the model should be LESS confident on that record — its
loss should go up, and the attack should fail.

We report:
- Attack success rate on training members (should be high)
- Attack success rate on non-members / test set (should be ~50%)
- The gap between these is the "privacy leakage"

Reference: Shokri et al. "Membership Inference Attacks Against ML Models"
           https://arxiv.org/abs/1610.05820
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_losses(ensemble, X, y):
    """Get per-sample cross-entropy loss from the ensemble."""
    return ensemble.per_sample_loss(X, y)


def threshold_attack(member_losses, nonmember_losses, threshold=None):
    """
    Run the loss-threshold membership inference attack.

    If loss < threshold → predict "member"
    If loss >= threshold → predict "non-member"

    If no threshold given, we pick the one that maximizes accuracy
    (this is the "best case" for the attacker).

    Returns dict with attack metrics.
    """
    all_losses = np.concatenate([member_losses, nonmember_losses])
    labels = np.concatenate([
        np.ones(len(member_losses)),   # 1 = member
        np.zeros(len(nonmember_losses))  # 0 = non-member
    ])

    if threshold is None:
        # try a bunch of thresholds and pick the best one
        # (this gives the attacker maximum advantage)
        percentiles = np.percentile(all_losses, np.arange(10, 91, 5))
        best_acc = 0
        best_t = np.median(all_losses)

        for t in percentiles:
            preds = (all_losses < t).astype(float)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_t = t

        threshold = best_t

    # make predictions
    predictions = (all_losses < threshold).astype(float)
    attack_acc = accuracy_score(labels, predictions)

    # per-group success rates
    member_correct = np.mean(member_losses < threshold)
    nonmember_correct = np.mean(nonmember_losses >= threshold)

    # AUC — using negative loss as the "score" (lower loss = more likely member)
    try:
        auc = roc_auc_score(labels, -all_losses)
    except ValueError:
        auc = 0.5

    return {
        "attack_accuracy": attack_acc,
        "member_recall": member_correct,      # TPR: how often do we correctly ID members
        "nonmember_recall": nonmember_correct,  # TNR
        "auc": auc,
        "threshold": threshold,
        "privacy_leakage": member_correct - (1 - nonmember_correct),  # gap above chance
    }


def run_mia(ensemble, X_members, y_members, X_nonmembers, y_nonmembers,
            label=""):
    """
    Full MIA pipeline: compute losses, run attack, print results.

    X_members / y_members: data that IS in the training set
    X_nonmembers / y_nonmembers: data that is NOT (e.g. test set)
    """
    member_losses = compute_losses(ensemble, X_members, y_members)
    nonmember_losses = compute_losses(ensemble, X_nonmembers, y_nonmembers)

    result = threshold_attack(member_losses, nonmember_losses)

    prefix = f"[MIA {label}] " if label else "[MIA] "
    print(f"\n{prefix}Results:")
    print(f"  Attack accuracy:  {result['attack_accuracy']:.4f} (chance = 0.50)")
    print(f"  Member recall:    {result['member_recall']:.4f}")
    print(f"  Non-member recall:{result['nonmember_recall']:.4f}")
    print(f"  AUC:              {result['auc']:.4f}")
    print(f"  Privacy leakage:  {result['privacy_leakage']:.4f}")
    print(f"  Threshold used:   {result['threshold']:.4f}")

    # store the raw losses too — useful for plotting
    result["member_losses"] = member_losses
    result["nonmember_losses"] = nonmember_losses

    return result


def compare_before_after(before_result, after_result):
    """
    Compare MIA results before and after forgetting.
    This is THE key proof that the model actually forgot.
    """
    print("\n" + "="*60)
    print("BEFORE vs AFTER forgetting — Membership Inference Attack")
    print("="*60)

    metrics = ["attack_accuracy", "member_recall", "auc", "privacy_leakage"]
    for m in metrics:
        b = before_result[m]
        a = after_result[m]
        delta = a - b
        direction = "↓" if delta < 0 else "↑" if delta > 0 else "—"
        print(f"  {m:20s}: {b:.4f} → {a:.4f} ({direction} {abs(delta):.4f})")

    # the headline number: did privacy leakage decrease?
    leak_before = before_result["privacy_leakage"]
    leak_after = after_result["privacy_leakage"]
    if leak_after < leak_before:
        print(f"\n  ✓ Privacy leakage DECREASED by {leak_before - leak_after:.4f}")
        print(f"    The model has less information about the forgotten record.")
    else:
        print(f"\n  ⚠ Privacy leakage did not decrease.")
        print(f"    This can happen with small samples — report honestly.")

    return {
        "before": before_result,
        "after": after_result,
        "leakage_reduction": leak_before - leak_after,
    }
