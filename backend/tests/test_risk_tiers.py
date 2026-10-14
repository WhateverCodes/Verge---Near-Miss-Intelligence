"""Review-priority bands (config.RISK_TIERS) on the unchanged weighted score."""
from backend.app import config, risk_scoring


def test_tier_boundaries():
    assert config.risk_tier(0.0) == "Low"
    assert config.risk_tier(0.19) == "Low"
    assert config.risk_tier(0.2) == "Moderate"
    assert config.risk_tier(0.4) == "High"
    assert config.risk_tier(0.6) == "Critical"
    assert config.risk_tier(1.0) == "Critical"


def test_worst_crash_history_is_critical_without_video_evidence():
    # Crash history alone can contribute at most W_ACCIDENTS + W_FATALITIES.
    # The site with the worst record on both must be able to reach Critical
    # even before any video has been analysed there.
    ceiling = config.W_ACCIDENTS + config.W_FATALITIES
    assert config.risk_tier(ceiling) == "Critical"

    sites = risk_scoring.load_accident_data(config.SAMPLE_ACCIDENTS_CSV)
    hotspots = risk_scoring.compute_hotspots(sites, {})
    worst = max(sites.values(), key=lambda s: (s["accidents_count"], s["fatalities"]))
    h = next(x for x in hotspots if x["site_id"] == worst["site_id"])
    assert h["risk_tier"] == "Critical"
    assert h["rank"] == 1


def test_tiers_follow_the_score_order():
    sites = risk_scoring.load_accident_data(config.SAMPLE_ACCIDENTS_CSV)
    hotspots = risk_scoring.compute_hotspots(sites, {})
    order = ["Low", "Moderate", "High", "Critical"]
    levels = [order.index(h["risk_tier"]) for h in hotspots]  # hotspots are sorted by score, high first
    assert levels == sorted(levels, reverse=True)
    assert {h["risk_tier"] for h in hotspots} == set(order)  # the sample data spans all four bands
