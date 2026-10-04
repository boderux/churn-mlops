"""Telegram notifications (used by Airflow DAGs and CI/CD)."""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)
API = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram(text: str, level: str = "info") -> bool:
    """Send an HTML message to the configured chat. Never raises.

    Returns True when delivered. Without TELEGRAM_BOT_TOKEN/CHAT_ID it only logs,
    so local runs and unit tests need no secrets.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    icon = {"info": "ℹ️", "ok": "✅", "warning": "⚠️", "critical": "🚨"}.get(level, "ℹ️")
    message = f"{icon} <b>[churn-mlops]</b>\n{text}"
    if not token or not chat_id:
        log.warning("Telegram not configured; message skipped:\n%s", message)
        return False
    try:
        resp = requests.post(
            API.format(token=token),
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        log.error("Telegram send failed: %s", exc)
        return False
