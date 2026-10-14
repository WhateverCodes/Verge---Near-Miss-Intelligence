"""Road Safety Assistant (grounded answers), site PDF brief and e-mail delivery."""
import email
from email import policy

import pytest
from fastapi.testclient import TestClient

from backend.app import assistant, hotspot_insight, mailer, site_evidence
from backend.app.main import app

client = TestClient(app)


def _text(r):
    out = []
    for b in r["blocks"]:
        if b["type"] == "text":
            out.append(b["text"])
        elif b["type"] == "bullets":
            out += b["items"]
        else:
            out += [" ".join(row) for row in b["rows"]]
    return " ".join(out)


def _site(name_part):
    return next(h for h in site_evidence.current_hotspots() if name_part in h["name"])


def test_hotspot_answer_uses_the_exact_record():
    h = _site("Swargate")
    r = assistant.answer("Why is this hotspot critical?", site_id=h["site_id"])
    t = _text(r)
    assert f"{h['risk_score']:.2f}" in t and f"{h['accidents_count']} crashes" in t
    assert f"{h['fatalities']} fatalities" in t and h["risk_tier"] in t
    assert r["source"] == assistant.SRC_HOTSPOTS
    assert h["summary"] in t  # one source of truth with View Analysis / PDF


def test_comparison_table_comes_from_current_data():
    a, b = _site("Swargate"), _site("Karve")
    r = assistant.answer("Compare Swargate and Karve Road.")
    table = next(x for x in r["blocks"] if x["type"] == "table")
    assert [row[0] for row in table["rows"]] == [f"#{a['rank']} {a['name']}", f"#{b['rank']} {b['name']}"]
    assert table["rows"][0][2] == str(a["accidents_count"]) and table["rows"][1][3] == str(b["fatalities"])


def test_unavailable_information_is_not_invented():
    t = _text(assistant.answer("Which is the safest road in Pune?"))
    assert "monitored locations" in t and "can't" in t
    t = _text(assistant.answer("What was the weather during the crash?"))
    assert "don't have that evidence" in t
    t = _text(assistant.answer("Why is this hotspot high risk?"))  # no hotspot selected
    assert "Select a monitored hotspot" in t


def test_positioning_answers():
    assert "does not predict" in _text(assistant.answer("Does Verge predict accidents?"))
    assert "does not assign fault" in _text(assistant.answer("Does Verge assign fault?"))


def test_assistant_endpoint():
    r = client.post("/api/assistant", json={"question": "How is the risk score calculated?"})
    assert r.status_code == 200 and "0.35" in _text(r.json())


def test_site_pdf_download():
    h = _site("Swargate")
    r = client.get(f"/api/hotspots/{h['site_id']}/report.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert hotspot_insight.site_report_filename(h) in r.headers["content-disposition"]


def test_email_not_configured_is_reported_honestly(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_FROM", "VERGE_EMAIL_OUTBOX"):
        monkeypatch.delenv(k, raising=False)
    h = _site("Swargate")
    r = client.post(f"/api/hotspots/{h['site_id']}/email", json={"to": "reviewer@example.com"})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


def test_email_rejects_bad_recipient(monkeypatch):
    monkeypatch.setenv("VERGE_EMAIL_OUTBOX", "1")
    h = _site("Swargate")
    r = client.post(f"/api/hotspots/{h['site_id']}/email", json={"to": "not-an-address"})
    assert r.status_code == 400


def test_outbox_mode_attaches_the_same_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv("VERGE_EMAIL_OUTBOX", "1")
    monkeypatch.setattr(mailer, "OUTBOX_DIR", tmp_path)
    monkeypatch.setattr(mailer.config, "ROOT_DIR", tmp_path.parent)
    h = _site("Swargate")
    r = client.post(f"/api/hotspots/{h['site_id']}/email",
                    json={"to": "reviewer@example.com", "subject": "Test\r\nBcc: x@y.z"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "saved_to_outbox"  # never claims "sent"
    eml = list(tmp_path.glob("*.eml"))
    assert len(eml) == 1
    msg = email.message_from_bytes(eml[0].read_bytes(), policy=policy.default)
    assert "\n" not in msg["Subject"] and msg["Bcc"] is None  # header injection stripped
    att = next(msg.iter_attachments())
    assert att.get_filename() == hotspot_insight.site_report_filename(h)
    pdf = client.get(f"/api/hotspots/{h['site_id']}/report.pdf").content
    # same generator, same data: identical apart from the timestamp line
    assert att.get_content().startswith(b"%PDF") and abs(len(att.get_content()) - len(pdf)) < 64


class _FakeSMTP:
    """Stands in for smtplib.SMTP: records the calls, optionally fails."""
    calls = []
    fail = False

    def __init__(self, host, port, timeout=None):
        _FakeSMTP.calls = [("connect", host, port)]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        _FakeSMTP.calls.append(("starttls",))

    def login(self, user, password):
        _FakeSMTP.calls.append(("login", user))

    def send_message(self, msg):
        if _FakeSMTP.fail:
            raise mailer.smtplib.SMTPAuthenticationError(535, b"bad credentials")
        _FakeSMTP.calls.append(("send", msg["To"], next(msg.iter_attachments()).get_filename()))
        return {}


def _smtp_env(monkeypatch):
    monkeypatch.delenv("VERGE_EMAIL_OUTBOX", raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "verge@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret-password")
    monkeypatch.setenv("SMTP_FROM", "verge@example.com")
    monkeypatch.setattr(mailer.smtplib, "SMTP", _FakeSMTP)


def test_smtp_send_reports_sent_only_after_the_server_accepts(monkeypatch):
    _smtp_env(monkeypatch)
    _FakeSMTP.fail = False
    h = _site("Swargate")
    r = client.post(f"/api/hotspots/{h['site_id']}/email", json={"to": "reviewer@example.com"})
    assert r.status_code == 200 and r.json()["status"] == "sent"
    assert [c[0] for c in _FakeSMTP.calls] == ["connect", "starttls", "login", "send"]
    assert _FakeSMTP.calls[-1][2] == hotspot_insight.site_report_filename(h)


def test_smtp_failure_gives_a_clean_message(monkeypatch):
    _smtp_env(monkeypatch)
    _FakeSMTP.fail = True
    h = _site("Swargate")
    r = client.post(f"/api/hotspots/{h['site_id']}/email", json={"to": "reviewer@example.com"})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail == mailer.SEND_FAILED and "secret" not in detail and "535" not in detail
    _FakeSMTP.fail = False
    assert mailer.status() == {"configured": True, "mode": "smtp", "from": "verge@example.com"}  # no secrets exposed


def test_unmonitored_place_is_never_invented():
    for q in ("wt about naval bridge", "What about Naval Bridge?", "tell me abt kothrud depot"):
        t = _text(assistant.answer(q))
        assert "don't currently have" in t and "monitored hotspot data" in t


def test_shorthand_and_typos_find_the_right_site():
    h = _site("Swargate")
    assert h["summary"] in _text(assistant.answer("wt about swargte"))


def test_dataset_rankings_from_the_records():
    sites = [h for h in site_evidence.current_hotspots() if h.get("has_historical_record", True)]
    top_c = max(sites, key=lambda h: h["accidents_count"])
    top_f = max(sites, key=lambda h: h["fatalities"])
    t = _text(assistant.answer("Which hotspot has the most crashes?"))
    assert top_c["name"] in t and f": {top_c['accidents_count']}." in t
    t = _text(assistant.answer("Which hotspot has the highest fatality count?"))
    assert top_f["name"] in t and f": {top_f['fatalities']}." in t


def test_out_of_scope_even_with_a_selected_hotspot():
    h = _site("Swargate")
    t = _text(assistant.answer("Tell me a joke", site_id=h["site_id"]))
    assert t.startswith("I can answer questions about Verge's monitored hotspots")
