"""Unit tests for detect.py and utils.py logic (no GPU / model weights required)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_ultralytics_stub() -> types.ModuleType:
    """Return a minimal stub for the `ultralytics` package."""
    stub = types.ModuleType("ultralytics")
    stub.YOLO = MagicMock()
    return stub


@pytest.fixture()
def detect_module():
    """Import detect with a stubbed ultralytics; restore original state after test."""
    original = sys.modules.pop("detect", None)
    sys.modules["ultralytics"] = _make_ultralytics_stub()
    import detect
    importlib.reload(detect)
    yield detect
    # Teardown: remove the stub so other tests get a clean slate.
    sys.modules.pop("detect", None)
    if original is not None:
        sys.modules["detect"] = original


# ---------------------------------------------------------------------------
# detect.py – argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self, detect_module):
        with patch("sys.argv", ["detect.py"]):
            args = detect_module.parse_args()
        assert args.weights == "best.pt"
        assert args.source == "0"
        assert args.conf == pytest.approx(0.5)
        assert args.imgsz == 640
        assert args.show is False
        assert args.save is False
        assert args.project == "outputs"
        assert args.name == "predict"

    def test_custom_flags(self, detect_module):
        argv = [
            "detect.py",
            "--weights", "model.pt",
            "--source", "video.mp4",
            "--conf", "0.3",
            "--imgsz", "1280",
            "--show",
            "--save",
            "--project", "results",
            "--name", "run1",
        ]
        with patch("sys.argv", argv):
            args = detect_module.parse_args()
        assert args.weights == "model.pt"
        assert args.source == "video.mp4"
        assert args.conf == pytest.approx(0.3)
        assert args.imgsz == 1280
        assert args.show is True
        assert args.save is True
        assert args.project == "results"
        assert args.name == "run1"


# ---------------------------------------------------------------------------
# detect.py – normalize_source
# ---------------------------------------------------------------------------

class TestNormalizeSource:
    def test_digit_string_returns_int(self, detect_module):
        assert detect_module.normalize_source("0") == 0
        assert detect_module.normalize_source("2") == 2

    def test_path_string_returned_unchanged(self, detect_module):
        assert detect_module.normalize_source("video.mp4") == "video.mp4"
        assert detect_module.normalize_source("/tmp/image.jpg") == "/tmp/image.jpg"


# ---------------------------------------------------------------------------
# detect.py – main raises FileNotFoundError for missing weights
# ---------------------------------------------------------------------------

class TestMainMissingWeights:
    def test_raises_when_weights_missing(self, detect_module, tmp_path):
        missing = str(tmp_path / "no_such_model.pt")
        with patch("sys.argv", ["detect.py", "--weights", missing]):
            with pytest.raises(FileNotFoundError, match="Model not found"):
                detect_module.main()


# ---------------------------------------------------------------------------
# utils.py – load_class_names
# ---------------------------------------------------------------------------

class TestLoadClassNames:
    def test_loads_names(self, tmp_path):
        from utils import load_class_names

        names_file = tmp_path / "classes.txt"
        names_file.write_text("cat\ndog\nbird\n", encoding="utf-8")
        names = load_class_names(names_file)
        assert names == ["cat", "dog", "bird"]

    def test_skips_blank_lines(self, tmp_path):
        from utils import load_class_names

        names_file = tmp_path / "classes.txt"
        names_file.write_text("cat\n\ndog\n  \nbird", encoding="utf-8")
        names = load_class_names(names_file)
        assert names == ["cat", "dog", "bird"]

    def test_raises_for_missing_file(self):
        from utils import load_class_names

        with pytest.raises(FileNotFoundError):
            load_class_names("/nonexistent/classes.txt")


# ---------------------------------------------------------------------------
# utils.py – draw_detections
# ---------------------------------------------------------------------------

class TestDrawDetections:
    def _blank_image(self, h: int = 480, w: int = 640) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_returns_same_image_object(self):
        from utils import draw_detections

        img = self._blank_image()
        result = draw_detections(img, [[10, 10, 100, 100]], [0.9], [0])
        assert result is img

    def test_draws_rectangle(self):
        from utils import draw_detections

        img = self._blank_image()
        draw_detections(img, [[10, 10, 100, 100]], [0.9], [0])
        # The green channel should have non-zero pixels from the box.
        assert img[:, :, 1].max() > 0

    def test_no_detections_leaves_image_black(self):
        from utils import draw_detections

        img = self._blank_image()
        draw_detections(img, [], [], [])
        assert img.sum() == 0

    def test_with_class_names(self):
        from utils import draw_detections

        img = self._blank_image()
        # Should not raise when valid class names are supplied.
        draw_detections(img, [[50, 50, 200, 200]], [0.85], [0], class_names=["person"])

    def test_out_of_range_class_id_no_crash(self):
        from utils import draw_detections

        img = self._blank_image()
        draw_detections(img, [[50, 50, 200, 200]], [0.5], [99], class_names=["cat"])

    def test_custom_text_color(self):
        from utils import draw_detections

        img = self._blank_image()
        # Pass white text color; should not raise.
        draw_detections(
            img, [[10, 10, 100, 100]], [0.9], [0], text_color=(255, 255, 255)
        )

