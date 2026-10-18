"""
Compliance score and regulatory report generator.

Turns the raw audit data into something a hospital CTO or
regulator would actually want to see. Computes a 0-100 score
based on how well the system is handling forget requests,
and generates an HTML report.
"""

import os
import json
from datetime import datetime, timezone

TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_compliance_score(receipt_chain, ensemble_accuracy, mia_results=None):
    """
    Score from 0-100 based on:
    - Request fulfillment rate (are all requests processed?)    — 30 pts
    - Average retrain speed (faster = better)                   — 25 pts
    - Privacy leakage reduction (MIA gap closed?)               — 25 pts
    - Chain integrity (no tampering)                            — 20 pts
    """
    score = 0
    breakdown = {}

    receipts = receipt_chain.receipts

    # 1. Fulfillment rate (30 pts)
    # For now all requests succeed, so this is always 30
    # In production this would track failed/pending requests
    n_fulfilled = len([r for r in receipts if "error" not in r])
    n_total = len(receipts) if receipts else 1
    fulfillment = n_fulfilled / n_total
    breakdown["fulfillment"] = round(fulfillment * 30, 1)
    score += breakdown["fulfillment"]

    # 2. Retrain speed (25 pts)
    # < 1s per shard retrain = perfect, > 30s = 0
    if receipts:
        avg_time = sum(r["retrain_time_seconds"] for r in receipts) / len(receipts)
        speed_score = max(0, 25 * (1 - avg_time / 30))
    else:
        avg_time = 0
        speed_score = 25
    breakdown["speed"] = round(speed_score, 1)
    score += speed_score

    # 3. Privacy (25 pts)
    # Based on MIA AUC — closer to 0.5 (random) = better privacy
    if mia_results and "auc" in mia_results:
        mia_auc = mia_results["auc"]
        # AUC of 0.5 = perfect privacy, AUC of 1.0 = total leakage
        privacy_score = 25 * max(0, 1 - 2 * abs(mia_auc - 0.5))
    else:
        privacy_score = 12.5  # unknown, give half credit
    breakdown["privacy"] = round(privacy_score, 1)
    score += privacy_score

    # 4. Chain integrity (20 pts)
    chain_valid = receipt_chain.verify_chain()
    breakdown["integrity"] = 20.0 if chain_valid else 0.0
    score += breakdown["integrity"]

    score = round(score, 1)

    print(f"\nCompliance Score: {score}/100")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    return {
        "score": score,
        "breakdown": breakdown,
        "chain_valid": chain_valid,
        "n_receipts": len(receipts),
        "ensemble_accuracy": ensemble_accuracy,
    }


def generate_report(compliance, receipt_chain, output_path=None):
    """Generate an HTML compliance report."""

    receipts_html = ""
    for i, r in enumerate(receipt_chain.receipts):
        receipts_html += f"""
        <tr>
            <td>{i+1}</td>
            <td>{r['request_id'][:8]}...</td>
            <td>{r['record_id']}</td>
            <td>{r['shard_id']}</td>
            <td>{r['retrain_time_seconds']:.3f}s</td>
            <td><code>{r['hash'][:16]}...</code></td>
        </tr>"""

    score = compliance["score"]
    # color based on score
    if score >= 80:
        score_color = "#22c55e"  # green
        grade = "A"
    elif score >= 60:
        score_color = "#eab308"  # yellow
        grade = "B"
    else:
        score_color = "#ef4444"  # red
        grade = "C"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ForgetProof Compliance Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0f172a; color: #e2e8f0;
            padding: 2rem; line-height: 1.6;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
        .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
        .score-card {{
            background: #1e293b; border-radius: 12px;
            padding: 2rem; text-align: center; margin-bottom: 2rem;
            border: 1px solid #334155;
        }}
        .score-number {{
            font-size: 4rem; font-weight: bold;
            color: {score_color};
        }}
        .score-label {{ color: #94a3b8; font-size: 1.2rem; }}
        .grade {{ font-size: 1.5rem; color: {score_color}; }}
        .breakdown {{
            display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 1rem; margin-bottom: 2rem;
        }}
        .metric {{
            background: #1e293b; padding: 1rem;
            border-radius: 8px; text-align: center;
            border: 1px solid #334155;
        }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; color: #38bdf8; }}
        .metric-label {{ color: #94a3b8; font-size: 0.85rem; }}
        table {{
            width: 100%; border-collapse: collapse;
            background: #1e293b; border-radius: 8px;
            overflow: hidden;
        }}
        th {{ background: #334155; padding: 0.75rem; text-align: left; color: #38bdf8; }}
        td {{ padding: 0.75rem; border-bottom: 1px solid #334155; }}
        tr:last-child td {{ border-bottom: none; }}
        code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
        .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.85rem; text-align: center; }}
        .chain-status {{
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            background: {"#22c55e20" if compliance["chain_valid"] else "#ef444420"};
            color: {"#22c55e" if compliance["chain_valid"] else "#ef4444"};
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 ForgetProof Compliance Report</h1>
        <p class="subtitle">
            Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
            &nbsp;|&nbsp; Chain integrity:
            <span class="chain-status">
                {"✓ Verified" if compliance["chain_valid"] else "✗ BROKEN"}
            </span>
        </p>

        <div class="score-card">
            <div class="score-number">{score}</div>
            <div class="score-label">Compliance Score</div>
            <div class="grade">Grade: {grade}</div>
        </div>

        <div class="breakdown">
            <div class="metric">
                <div class="metric-value">{compliance["breakdown"]["fulfillment"]}/30</div>
                <div class="metric-label">Fulfillment</div>
            </div>
            <div class="metric">
                <div class="metric-value">{compliance["breakdown"]["speed"]}/25</div>
                <div class="metric-label">Speed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{compliance["breakdown"]["privacy"]}/25</div>
                <div class="metric-label">Privacy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{compliance["breakdown"]["integrity"]}/20</div>
                <div class="metric-label">Integrity</div>
            </div>
        </div>

        <h2 style="color: #38bdf8; margin-bottom: 1rem;">Forget Request Log</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Request ID</th><th>Record</th>
                    <th>Shard</th><th>Retrain Time</th><th>Hash</th>
                </tr>
            </thead>
            <tbody>
                {receipts_html if receipts_html else "<tr><td colspan='6' style='text-align:center'>No requests processed yet</td></tr>"}
            </tbody>
        </table>

        <div class="footer">
            <p>ForgetProof — SISA-based Machine Unlearning for Healthcare</p>
            <p>Model accuracy: {compliance["ensemble_accuracy"]:.4f} | {compliance["n_receipts"]} forget requests processed</p>
        </div>
    </div>
</body>
</html>"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved to {output_path}")

    return html
