import hashlib

from backend.app import config, report, risk_scoring, site_evidence


def _csv_hash():
    return hashlib.sha256(config.SAMPLE_ACCIDENTS_CSV.read_bytes()).hexdigest()


def test_default_scoring_unchanged_when_no_subset_given():
    sites = risk_scoring.load_accident_data(config.SAMPLE_ACCIDENTS_CSV)
    a = risk_scoring.compute_hotspots(dict(sites), {})
    b = risk_scoring.compute_hotspots(dict(sites), {}, historical_site_ids=list(sites))
    assert [(h["site_id"], h["risk_score"]) for h in a] == [(h["site_id"], h["risk_score"]) for h in b]


def test_components_add_up_to_score(isolated_outputs):
    for h in site_evidence.current_hotspots():
        assert abs(h["historical_component"] + h["observed_component"] - h["risk_score"]) < 0.002


def test_observation_raises_site_priority_without_touching_history(isolated_outputs):
    csv_before = _csv_hash()
    before = site_evidence.current_hotspots()
    low = max(before, key=lambda h: h["rank"])  # lowest-ranked site
    event = {"event_index": 0, "kind": "vehicle_pedestrian", "severity": 0.5, "peak_severity": 0.7,
             "peak_t": 3.0, "min_ttc_s": 0.4, "min_distance_m": 4.0, "pet_s": None}
    site_evidence.record_observation("test-1", low["site_id"], [event], meta={})
    after = site_evidence.current_hotspots()
    upd = site_evidence.site_update(before, after, low["site_id"], [event])

    assert upd["after"]["near_miss_count"] == upd["before"]["near_miss_count"] + 1
    assert upd["after"]["risk_score"] > upd["before"]["risk_score"]
    assert upd["after"]["historical_component"] == upd["before"]["historical_component"]
    assert upd["after"]["rank"] <= upd["before"]["rank"]
    assert _csv_hash() == csv_before

    site_evidence.clear_observations()
    reset = {h["site_id"]: h["risk_score"] for h in site_evidence.current_hotspots()}
    assert reset == {h["site_id"]: h["risk_score"] for h in before}


def test_map_pin_has_no_history_and_does_not_shift_others(isolated_outputs):
    before = {h["site_id"]: h["historical_component"] for h in site_evidence.current_hotspots()}
    pin = site_evidence.add_custom_site("Test crossing", 18.53, 73.85)
    hs = {h["site_id"]: h for h in site_evidence.current_hotspots()}
    assert hs[pin["site_id"]]["has_historical_record"] is False
    assert hs[pin["site_id"]]["historical_component"] == 0
    for sid, hist in before.items():
        assert hs[sid]["historical_component"] == hist
    # same spot again -> same pin, not a duplicate
    assert site_evidence.add_custom_site("", 18.53, 73.85)["site_id"] == pin["site_id"]


def test_report_is_clean_utf8_and_structured(isolated_outputs):
    hotspots = site_evidence.current_hotspots()
    text = report.generate_report(hotspots)
    for bad in ("�", "â€", "Ã"):  # replacement char / common mojibake
        assert bad not in text
    assert "**Evidence**" in text and "**Recommended human review**" in text
    b = report.site_brief(hotspots[0])
    assert b["evidence"] and b["key_factors"] and 1 <= len(b["review"]) <= 3
