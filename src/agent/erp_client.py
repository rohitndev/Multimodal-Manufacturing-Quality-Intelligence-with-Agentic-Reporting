"""Mock ERP client.

Posts WIP status updates to a configured ERP webhook (``ERP_WEBHOOK_URL``).
When no URL is configured, writes the payload to ``data/erp_outbox.jsonl`` so
the audit trail still exists end-to-end.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class ERPClient:
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        outbox: str = "data/erp_outbox.jsonl",
        timeout: int = 5,
    ):
        self.webhook_url = webhook_url or os.getenv("ERP_WEBHOOK_URL")
        self.outbox = Path(outbox)
        self.timeout = timeout
        self.outbox.parent.mkdir(parents=True, exist_ok=True)

    def update_wip(self, report_id: str, inspection_id: str, decision: str, payload: dict) -> dict:
        body = {
            "report_id": report_id,
            "inspection_id": inspection_id,
            "status": self._decision_to_status(decision),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }
        if self.webhook_url:
            try:
                resp = requests.post(self.webhook_url, json=body, timeout=self.timeout)
                logger.info("ERP webhook posted (%s)", resp.status_code)
                return {"transport": "http", "status_code": resp.status_code, "body": body}
            except Exception as exc:  # noqa: BLE001
                logger.warning("ERP webhook failed (%s) — outbox fallback", exc)
        with self.outbox.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(body) + "\n")
        return {"transport": "outbox", "path": str(self.outbox), "body": body}

    @staticmethod
    def _decision_to_status(decision: str) -> str:
        return {
            "PASS": "ACCEPTED",
            "QUARANTINE": "QUARANTINED",
            "REWORK": "REWORK_QUEUED",
            "FAIL": "SCRAPPED",
        }.get(decision.upper(), "REVIEW")
