"""End-to-end API flow: location-linked analysis -> evidence -> site update -> PDF."""
import time

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client(isolated_outputs):
    return TestClient(app)


def _wait(client, analysis_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/analyses/{analysis_id}/status").json()
        if job["state"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("analysis did not finish")


def test_health_hotspots_report(client):
    assert client.get("/api/health").json()["status"] == "ok"
    feats = client.get("/api/hotspots").json()["features"]
    assert feats and "brief" in feats[0]["properties"]
    assert "Ranked locations" in client.get("/api/report").text


def test_location_linked_analysis_updates_site_and_builds_pdf(client, demo_clip, isolated_outputs):
    with open(demo_clip, "rb") as fh:
        r = client.post("/api/analyses", files={"file": ("cam.mp4", fh, "video/mp4")}, data={"site_id": "SITE_008"})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["analysis_id"])
    assert job["state"] == "done", job.get("error")
    assert [s["state"] for s in job["stages"]] == ["done"] * 5

    res = job["result"]
    assert res["near_miss_detected"] is True
    e = res["events"][0]
    assert e["evidence"]["frame_urls"] and e["evidence"]["snapshot_url"]
    assert e["min_ttc_s"] is not None and e["what_happened"] and e["why_flagged"]

    upd = res["site_update"]
    assert upd["site_id"] == "SITE_008"
    assert upd["after"]["risk_score"] > upd["before"]["risk_score"]

    # the uploaded raw video is not retained
    assert list((isolated_outputs / "uploads").iterdir()) == []

    pdf = client.get(e["report_pdf_url"])
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF") and pdf.content.rstrip().endswith(b"%%EOF")
    assert "attachment" in pdf.headers["content-disposition"]

    hs = {f["properties"]["site_id"]: f["properties"] for f in client.get("/api/hotspots").json()["features"]}
    assert hs["SITE_008"]["near_miss_count"] >= 1

    assert client.delete("/api/observations").json()["observations_removed"] == 1


def test_sync_endpoint_without_location_keeps_old_contract(client, demo_clip):
    with open(demo_clip, "rb") as fh:
        r = client.post("/api/analyze-video", files={"file": ("cam.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("site_id", "frames_processed", "duration_s", "vehicles_tracked", "pedestrians_tracked",
                "near_miss_detected", "event_count", "events", "verdict", "annotated_video_url"):
        assert key in body
    assert body["site_update"] is None


def test_bad_inputs_give_clean_errors(client):
    assert client.post("/api/analyses", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/api/analyze-video", files={"file": ("x.mp4", b"not a video", "video/mp4")}).status_code == 400
    assert client.post("/api/analyses", files={"file": ("x.mp4", b"", "video/mp4")}).status_code == 400
    assert client.post("/api/analyses", files={"file": ("x.mp4", b"abc", "video/mp4")},
                       data={"site_id": "NOPE"}).status_code == 400
    assert client.get("/api/analyses/does-not-exist/report.pdf").status_code == 404
