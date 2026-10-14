import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app import config  # noqa: E402


def make_track(track_id: int, cls: str, positions, w: int = 20, h: int = 20) -> list[dict]:
    """positions: iterable of (frame, x, y) -> flat track rows (as tracking.build_tracks)."""
    return [
        {"id": track_id, "cls": cls, "frame": f, "t": f / config.FRAME_RATE_FPS, "x": x, "y": y, "w": w, "h": h}
        for f, x, y in positions
    ]


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    """Point every generated-output path at a temp dir. The historical CSV is
    left pointing at the real (read-only) file."""
    out = tmp_path / "outputs"
    out.mkdir()
    for name, value in {
        "OUTPUT_DIR": out,
        "UPLOADS_DIR": out / "uploads",
        "ANALYSES_DIR": out / "analyses",
        "OBSERVATIONS_JSON": out / "observations.json",
        "CUSTOM_SITES_JSON": out / "custom_sites.json",
        "NEAR_MISS_JSON": out / "near_miss_events.json",
        "TRACKS_JSON": out / "tracks.json",
        "HOTSPOTS_GEOJSON": out / "hotspots.geojson",
        "REPORT_MD": out / "report.md",
    }.items():
        monkeypatch.setattr(config, name, value)
    (out / "uploads").mkdir()
    (out / "analyses").mkdir()
    return out


@pytest.fixture(scope="session")
def demo_clip(tmp_path_factory):
    """A short synthetic near-miss clip (same generator as the demo clips)."""
    import cv2

    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_demo_clips as g

    path = tmp_path_factory.mktemp("clips") / "crossing.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), g.FPS, (g.W, g.H))
    bg = g.background()
    for _ in range(g.LEAD_IN_FRAMES):
        wr.write(bg)
    for f in g.crossing_near_miss():
        wr.write(f)
    wr.release()
    return path
