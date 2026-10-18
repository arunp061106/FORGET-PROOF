"""
ForgetProof — Master Experiment Runner

This script runs the entire pipeline end-to-end:
1. Download & preprocess UCI Diabetes data
2. Train baseline model (single RandomForest, full dataset)
3. Plant canary records & train sharded ensemble (SISA, N=8) in parallel
4. Run Membership Inference Attack BEFORE forgetting (measures memorization)
5. Execute forget requests (retrains only affected shards, signs HMAC receipts)
6. Run Membership Inference Attack AFTER forgetting (verifies empirical unlearning)
7. Measure retrain speedup (single shard vs full model)
8. Shard sweep (N=2, 4, 8, 16) measuring accuracy vs isolation tradeoff
9. Streaming withdrawal simulation (30 requests) tracking cumulative savings
10. Generate charts, lineage diagram, and compliance report

All figures and logs are saved to the `outputs/` directory.
"""

import sys
import os
import json
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from data.download import download_dataset
from data.preprocess import prepare_data
from models.shard_trainer import train_all_shards
from models.ensemble import ShardedEnsemble
from models.baseline import train_baseline, full_retrain
from forgetting.canary import create_canaries, inject_canaries, get_canary_data
from forgetting.forget_engine import forget
from forgetting.membership_inference import run_mia, compare_before_after
from audit.receipt import ReceiptChain
from audit.compliance import compute_compliance_score, generate_report
from visualization.plots import (
    plot_accuracy_vs_shards, plot_retrain_comparison, plot_mia_before_after,
    plot_cumulative_savings, plot_accuracy_drift, plot_shard_sizes,
)
from visualization.lineage_graph import draw_lineage

OUTPUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "outputs")


def run_everything():
    overall_start = time.time()
    all_results = {}

    # =========================================================
    # STEP 1: Data Download and Preprocessing
    # =========================================================
    print("\n" + "="*60)
    print("STEP 1: Load and preprocess dataset")
    print("="*60)

    raw_df = download_dataset()
    N_SHARDS = 8
    data = prepare_data(n_shards=N_SHARDS, raw_df=raw_df)

    # =========================================================
    # STEP 2: Baseline Model (Full Retrain Standard)
    # =========================================================
    print("\n" + "="*60)
    print("STEP 2: Train baseline (single RandomForest on all data)")
    print("="*60)

    baseline = train_baseline(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"]
    )
    all_results["baseline"] = {
        "accuracy": baseline["accuracy"],
        "auc": baseline["auc"],
        "train_time": baseline["train_time"],
    }

    # =========================================================
    # STEP 3: Sharded Ensemble + Canary Injection
    # =========================================================
    print("\n" + "="*60)
    print("STEP 3: Inject canary records and train SISA shards")
    print("="*60)

    canary_df, canary_ids = create_canaries(
        data["train_df"], data["feature_names"], n_canaries=20
    )
    train_df_with_canaries = inject_canaries(data["train_df"], canary_df)

    # Train all 8 shards in parallel
    shard_result = train_all_shards(
        train_df_with_canaries, data["feature_names"],
        n_shards=N_SHARDS, n_jobs=-1
    )

    ensemble = ShardedEnsemble(models=shard_result["models"])
    ensemble_metrics = ensemble.evaluate(
        data["X_test"], data["y_test"], label="Sharded Ensemble"
    )
    all_results["ensemble"] = ensemble_metrics

    # Chart shard sizes
    plot_shard_sizes(train_df_with_canaries, N_SHARDS, OUTPUT_DIR)

    # =========================================================
    # STEP 4: Membership Inference Attack (BEFORE Forgetting)
    # =========================================================
    print("\n" + "="*60)
    print("STEP 4: Membership inference attack BEFORE unlearning")
    print("="*60)

    X_canary, y_canary, present_canary_ids = get_canary_data(
        canary_ids, train_df_with_canaries, data["feature_names"]
    )

    # Non-members from held-out test data
    n_nonmember = min(len(X_canary) * 5, len(data["X_test"]))
    X_nonmember = data["X_test"][:n_nonmember]
    y_nonmember = data["y_test"][:n_nonmember]

    mia_before = run_mia(
        ensemble, X_canary, y_canary, X_nonmember, y_nonmember,
        label="BEFORE unlearning (canaries memorized)"
    )

    # =========================================================
    # STEP 5: Process Forget Requests & Build Signed Audit Trail
    # =========================================================
    print("\n" + "="*60)
    print("STEP 5: Execute unlearning on canary records & sign receipts")
    print("="*60)

    receipt_chain = ReceiptChain()
    train_df_current = train_df_with_canaries.copy()

    # Forget 10 canaries, keep 10 as control
    canaries_to_forget = canary_ids[:10]

    for cid in canaries_to_forget:
        result, train_df_current = forget(
            cid, train_df_current, ensemble, data["feature_names"]
        )

        if "error" not in result:
            acc_after = np.mean(ensemble.predict(data["X_test"]) == data["y_test"])

            receipt = receipt_chain.add(
                record_id=cid,
                shard_id=result["shard_id"],
                retrain_time=result["retrain_time"],
                accuracy_before=ensemble_metrics["accuracy"],
                accuracy_after=float(acc_after),
                mia_before=mia_before["auc"],
            )

    # =========================================================
    # STEP 6: Membership Inference Attack (AFTER Forgetting)
    # =========================================================
    print("\n" + "="*60)
    print("STEP 6: Membership inference attack AFTER unlearning")
    print("="*60)

    # Test the forgotten records against the updated model
    X_forgotten, y_forgotten = [], []
    for cid in canaries_to_forget:
        row = canary_df[canary_df["record_id"] == cid]
        if len(row) > 0:
            X_forgotten.append(row[data["feature_names"]].values[0])
            y_forgotten.append(row["target"].values[0])

    X_forgotten = np.array(X_forgotten)
    y_forgotten = np.array(y_forgotten)

    mia_after = run_mia(
        ensemble, X_forgotten, y_forgotten, X_nonmember, y_nonmember,
        label="AFTER unlearning (target records removed)"
    )

    comparison = compare_before_after(mia_before, mia_after)
    all_results["mia_comparison"] = {
        "before_auc": mia_before["auc"],
        "after_auc": mia_after["auc"],
        "leakage_reduction": comparison["leakage_reduction"],
    }

    # =========================================================
    # STEP 7: Retrain Time Comparison
    # =========================================================
    print("\n" + "="*60)
    print("STEP 7: Retrain latency comparison")
    print("="*60)

    _, full_time = full_retrain(data["X_train"], data["y_train"])
    avg_shard_time = np.mean(list(shard_result["times"].values()))

    plot_retrain_comparison(avg_shard_time, full_time, N_SHARDS, OUTPUT_DIR)
    all_results["retrain_times"] = {
        "full_retrain": full_time,
        "avg_shard_retrain": avg_shard_time,
        "speedup": full_time / avg_shard_time if avg_shard_time > 0 else 0,
    }

    # =========================================================
    # STEP 8: Shard Sweep (N=2, 4, 8, 16)
    # =========================================================
    print("\n" + "="*60)
    print("STEP 8: Shard count sweep")
    print("="*60)

    from experiments.shard_sweep import run_sweep
    sweep_results = run_sweep(raw_df=raw_df, shard_counts=[2, 4, 8, 16])
    plot_accuracy_vs_shards(sweep_results, OUTPUT_DIR)
    all_results["sweep"] = sweep_results

    # =========================================================
    # STEP 9: Streaming Consent Withdrawal Simulation
    # =========================================================
    print("\n" + "="*60)
    print("STEP 9: Streaming consent revocation simulation")
    print("="*60)

    from experiments.streaming_sim import run_simulation
    sim_data = prepare_data(n_shards=N_SHARDS, raw_df=raw_df)
    sim_log = run_simulation(sim_data, n_requests=30)
    plot_cumulative_savings(sim_log, OUTPUT_DIR)
    plot_accuracy_drift(sim_log, OUTPUT_DIR)
    all_results["simulation"] = {
        "n_requests": len(sim_log),
        "total_sisa_time": sim_log[-1]["cumulative_sisa_time"],
        "total_full_time": sim_log[-1]["cumulative_full_time"],
        "compute_saved_pct": sim_log[-1]["compute_saved_pct"],
        "final_accuracy": sim_log[-1]["accuracy"],
    }

    # =========================================================
    # STEP 10: Charts & Lineage Graph
    # =========================================================
    plot_mia_before_after(mia_before, mia_after, OUTPUT_DIR)
    draw_lineage(N_SHARDS, forget_shard=canaries_to_forget[0] % N_SHARDS,
                 output_dir=OUTPUT_DIR)

    # =========================================================
    # STEP 11: Audit Trail Verification & Compliance Report
    # =========================================================
    print("\n" + "="*60)
    print("STEP 11: Audit trail verification & regulatory report")
    print("="*60)

    receipt_chain.summary()

    final_acc = float(np.mean(ensemble.predict(data["X_test"]) == data["y_test"]))
    compliance = compute_compliance_score(receipt_chain, final_acc, mia_after)
    generate_report(
        compliance, receipt_chain,
        output_path=os.path.join(OUTPUT_DIR, "compliance_report.html")
    )
    receipt_chain.to_json(os.path.join(OUTPUT_DIR, "receipt_chain.json"))

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    total_time = time.time() - overall_start
    print("\n" + "="*60)
    print("FORGETPROOF — EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nMeasured Results:")
    print(f"  Baseline Accuracy (full data):      {all_results['baseline']['accuracy']:.4f}")
    print(f"  Sharded Ensemble Accuracy:          {all_results['ensemble']['accuracy']:.4f}")
    print(f"  Accuracy Difference:                {all_results['baseline']['accuracy'] - all_results['ensemble']['accuracy']:.4f}")
    print(f"  MIA AUC Before Unlearning:          {all_results['mia_comparison']['before_auc']:.4f}")
    print(f"  MIA AUC After Unlearning:           {all_results['mia_comparison']['after_auc']:.4f}")
    print(f"  Full Retrain Latency:               {all_results['retrain_times']['full_retrain']:.2f}s")
    print(f"  Single Shard Retrain Latency:       {all_results['retrain_times']['avg_shard_retrain']:.2f}s")
    print(f"  Retrain Speedup (Measured):         {all_results['retrain_times']['speedup']:.1f}×")
    print(f"  Streaming Simulation Compute Saved: {all_results['simulation']['compute_saved_pct']:.1f}%")
    print(f"  Audit Chain Integrity:              {'Verified (HMAC-SHA256)' if receipt_chain.verify_chain() else 'FAIL'}")
    print(f"  Compliance Score:                   {compliance['score']}/100")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"Results JSON saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_everything()
