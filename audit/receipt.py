"""
Forget receipts — tamper-evident, HMAC-signed audit log of every forget request.

Each receipt contains:
- request_id (UUID)
- timestamp
- record_id and shard_id
- retrain time
- accuracy before/after
- MIA result before/after
- prev_hash (SHA-256 chain to previous receipt)
- hash (SHA-256 digest of receipt contents)
- signature (HMAC-SHA256 signature authenticating the receipt)

The SHA-256 chain guarantees that earlier entries cannot be altered
or deleted retroactively without invalidating all downstream hashes.
The HMAC signature ensures receipts originate from the authorized
unlearning controller and cannot be forged.

Note: The receipt proves that the unlearning request was executed and
logged without tampering. The empirical proof that unlearning succeeded
is the membership inference attack result.
"""

import json
import hashlib
import hmac
import uuid
from datetime import datetime, timezone


DEFAULT_HMAC_KEY = b"forgetproof-audit-key-linuxone-2026"


class ReceiptChain:
    def __init__(self, secret_key=DEFAULT_HMAC_KEY):
        self.receipts = []
        self.secret_key = secret_key

    def _hash(self, data_str):
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _sign(self, hash_str):
        return hmac.new(self.secret_key, hash_str.encode("utf-8"), hashlib.sha256).hexdigest()

    def add(self, record_id, shard_id, retrain_time,
            accuracy_before=None, accuracy_after=None,
            mia_before=None, mia_after=None, extra=None):
        """
        Create a new forget receipt, chain it to the previous one, and sign it.
        """
        prev_hash = self.receipts[-1]["hash"] if self.receipts else "GENESIS"

        receipt = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record_id": int(record_id),
            "shard_id": int(shard_id),
            "retrain_time_seconds": round(retrain_time, 4),
            "accuracy_before": round(accuracy_before, 4) if accuracy_before is not None else None,
            "accuracy_after": round(accuracy_after, 4) if accuracy_after is not None else None,
            "mia_auc_before": round(mia_before, 4) if mia_before is not None else None,
            "mia_auc_after": round(mia_after, 4) if mia_after is not None else None,
            "prev_hash": prev_hash,
        }

        if extra:
            receipt["extra"] = extra

        # hash the receipt content (without hash or signature fields)
        content = json.dumps(receipt, sort_keys=True)
        digest = self._hash(content)
        receipt["hash"] = digest
        receipt["signature"] = self._sign(digest)

        self.receipts.append(receipt)
        print(f"Receipt #{len(self.receipts)}: record {record_id}, "
              f"shard {shard_id}, hash {receipt['hash'][:12]}..., sig {receipt['signature'][:12]}...")

        return receipt

    def verify_chain(self):
        """
        Verify the integrity of the entire receipt chain and all HMAC signatures.
        Returns True if everything checks out, False if any entry was tampered with.
        """
        if not self.receipts:
            return True

        for i, receipt in enumerate(self.receipts):
            # check prev_hash links
            if i == 0:
                if receipt["prev_hash"] != "GENESIS":
                    print(f"Chain broken at receipt 0: expected GENESIS")
                    return False
            else:
                if receipt["prev_hash"] != self.receipts[i-1]["hash"]:
                    print(f"Chain broken at receipt {i}: prev_hash mismatch")
                    return False

            # verify this receipt's content hash
            stored_hash = receipt.get("hash")
            check_receipt = {k: v for k, v in receipt.items() if k not in ("hash", "signature")}
            content = json.dumps(check_receipt, sort_keys=True)
            computed_hash = self._hash(content)

            if stored_hash != computed_hash:
                print(f"Chain broken at receipt {i}: content hash mismatch")
                return False

            # verify HMAC signature
            stored_sig = receipt.get("signature")
            computed_sig = self._sign(stored_hash)
            if not stored_sig or not hmac.compare_digest(stored_sig, computed_sig):
                print(f"Chain broken at receipt {i}: invalid HMAC signature")
                return False

        print(f"Chain verified: {len(self.receipts)} receipts, all hashes & signatures valid ✓")
        return True

    def to_json(self, path=None):
        """Export the full chain as JSON."""
        data = {
            "chain_length": len(self.receipts),
            "verified": self.verify_chain(),
            "receipts": self.receipts,
        }

        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Chain exported to {path}")

        return data

    def summary(self):
        """Quick summary of the chain."""
        if not self.receipts:
            print("No receipts logged.")
            return

        print(f"\n{'='*50}")
        print(f"Forget Receipt Chain — {len(self.receipts)} entries")
        print(f"{'='*50}")

        total_time = sum(r["retrain_time_seconds"] for r in self.receipts)
        shards_affected = set(r["shard_id"] for r in self.receipts)

        print(f"Total retrain time: {total_time:.2f}s")
        print(f"Shards affected: {sorted(shards_affected)}")
        print(f"Chain integrity: {'✓ Valid (Hashes + HMAC signed)' if self.verify_chain() else '✗ BROKEN'}")

        for i, r in enumerate(self.receipts):
            print(f"  #{i+1} | record {r['record_id']} | shard {r['shard_id']} | "
                  f"{r['retrain_time_seconds']:.3f}s | hash {r['hash'][:10]}... | sig {r['signature'][:10]}...")
