"""
E-mail delivery of Verge PDF reports (server side only).

Configuration comes from environment variables (or a local .env file, see
.env.example); credentials never reach the browser.

  SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM
  SMTP_SECURITY      starttls (default) | ssl | none
  VERGE_EMAIL_OUTBOX 1 = test mode: the complete message (with the PDF
                     attached) is written to outputs/outbox/*.eml instead of
                     being sent, so the MIME can be checked without a server.

A report is only ever reported as "sent" after the SMTP server accepted it.
"""
from __future__ import annotations

import os
import re
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from . import config

OUTBOX_DIR = config.OUTPUT_DIR / "outbox"
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

NOT_CONFIGURED = "Email delivery is not configured. Add SMTP credentials to enable direct sending."
SEND_FAILED = "Unable to send the report. Please check the email configuration and try again."


class MailError(Exception):
    """User-facing message only (no credentials, no server details)."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def status() -> dict:
    """What the UI may know: the delivery mode and the sender, never secrets."""
    if _env("VERGE_EMAIL_OUTBOX") in ("1", "true", "yes"):
        return {"configured": True, "mode": "outbox", "from": _env("SMTP_FROM", "verge@localhost"),
                "message": "Test mode: reports are saved to outputs/outbox as .eml files, not sent."}
    if _env("SMTP_HOST") and _env("SMTP_FROM"):
        return {"configured": True, "mode": "smtp", "from": _env("SMTP_FROM")}
    return {"configured": False, "mode": "off", "message": NOT_CONFIGURED}


def clean_subject(subject: str) -> str:
    s = re.sub(r"[\r\n\t]+", " ", subject or "").strip()
    return s[:200] or "Verge Road Safety Report"


def valid_recipient(addr: str) -> str:
    a = (addr or "").strip()
    if len(a) > 254 or not _EMAIL_RE.match(a):
        raise MailError("Please enter one valid recipient email address.", status=400)
    return a


def build_message(to: str, subject: str, body: str, filename: str, pdf: bytes) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = _env("SMTP_FROM", "verge@localhost")
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="verge.local")
    msg.set_content(body)
    msg.add_attachment(pdf, maintype="application", subtype="pdf", filename=filename)
    return msg


def send_report(to: str, subject: str, body: str, filename: str, pdf: bytes) -> dict:
    to = valid_recipient(to)
    subject = clean_subject(subject)
    st = status()
    if not st["configured"]:
        raise MailError(NOT_CONFIGURED, status=503)
    msg = build_message(to, subject, body, filename, pdf)

    if st["mode"] == "outbox":
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTBOX_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{filename.rsplit('.', 1)[0]}.eml"
        path.write_bytes(bytes(msg))
        return {"status": "saved_to_outbox", "to": to, "attachment": filename,
                "path": str(path.relative_to(config.ROOT_DIR)).replace("\\", "/")}

    host, port = _env("SMTP_HOST"), int(_env("SMTP_PORT", "587") or 587)
    security = _env("SMTP_SECURITY", "starttls").lower()
    user, password = _env("SMTP_USERNAME"), os.environ.get("SMTP_PASSWORD", "")
    try:
        if security == "ssl":
            server = smtplib.SMTP_SSL(host, port, timeout=20, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(host, port, timeout=20)
        with server:
            if security == "starttls":
                server.starttls(context=ssl.create_default_context())
            if user:
                server.login(user, password)
            refused = server.send_message(msg)
        if refused:
            raise MailError(SEND_FAILED)
    except MailError:
        raise
    except (smtplib.SMTPException, OSError) as exc:
        raise MailError(SEND_FAILED) from exc
    return {"status": "sent", "to": to, "attachment": filename}
