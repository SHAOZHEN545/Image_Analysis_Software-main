import csv
import json
import os

import pytest

pytest.importorskip("wx")

import wx

from ImageUI import ImageUI
from fitting_window import FittingWindow


def _strip_trailing(path):
    return path.rstrip("/\\")


def test_deduce_session_base_with_known_suffix():
    assert (
        FittingWindow._deduce_session_base(
            "/tmp/run123-fitting-settings.json"
        )
        == "run123"
    )
    assert (
        FittingWindow._deduce_session_base("data/session-heatmap.csv")
        == "session-heatmap"
    )
    assert (
        FittingWindow._deduce_session_base("C:/exports/run42.json") == "run42"
    )
    assert (
        FittingWindow._deduce_session_base("C:/exports/run42-session.json")
        == "run42"
    )


def test_extract_image_file_from_csv(tmp_path):
    csv_path = tmp_path / "session.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Value", "Image File", "Other"])
        writer.writerow(["1", "", "note"])
        writer.writerow(["2", "C:/data/2024/run/image001.fits", "info"])

    result = FittingWindow._extract_image_file_from_csv(str(csv_path))
    assert result == "C:/data/2024/run/image001.fits"


def test_find_session_csv_handles_new_bundle_suffix(tmp_path):
    base_name = "bundle"
    legacy = tmp_path / f"{base_name}.csv"
    legacy.write_text("legacy", encoding="utf-8")
    assert FittingWindow._find_session_csv(base_name, str(tmp_path)) == str(legacy)

    legacy.unlink()
    session_variant = tmp_path / f"{base_name}-session.csv"
    session_variant.write_text("session", encoding="utf-8")
    assert FittingWindow._find_session_csv(base_name, str(tmp_path)) == str(
        session_variant
    )


def test_normalise_image_folder_handles_absolute_and_relative(tmp_path):
    export_folder = str(tmp_path)
    abs_image = tmp_path / "run" / "image002.fits"
    abs_image.parent.mkdir()
    abs_image.write_text("dummy")

    absolute_result = FittingWindow._normalise_image_folder(
        str(abs_image), export_folder
    )
    assert absolute_result.endswith((os.sep, "\\"))
    assert _strip_trailing(absolute_result) == _strip_trailing(str(abs_image.parent))

    rel_image = os.path.join("nested", "image003.fits")
    relative_result = FittingWindow._normalise_image_folder(rel_image, export_folder)
    expected_relative = os.path.join(export_folder, "nested")
    assert relative_result.endswith((os.sep, "\\"))
    assert _strip_trailing(relative_result) == _strip_trailing(expected_relative)


def test_strip_sequence_header_handles_header_and_plain_values():
    header_text = "Parameter (unit)\n1\n2\n"
    assert FittingWindow._strip_sequence_header(header_text) == "1\n2"

    plain_text = "1\n2\n3"
    assert FittingWindow._strip_sequence_header(plain_text) == "1\n2\n3"


def test_serialise_sequence_from_text_normalises_whitespace():
    text = " 1  2 \r\n3\r\n\r\n"
    assert FittingWindow._serialise_sequence_from_text(text) == ["1  2 ", "3"]

    assert FittingWindow._serialise_sequence_from_text("") == []


def test_apply_user_settings_updates_start_file():
    existing_app = wx.App.Get()
    app = existing_app or wx.App(False)
    created = existing_app is None
    window = FittingWindow(None)
    try:
        window.startFileCtrl.ChangeValue("initial_start")
        window._apply_user_settings({"start_file": "restored_file.fits"})
        assert window.startFileCtrl.GetValue() == "restored_file.fits"
    finally:
        window.Destroy()
        if created:
            app.Destroy()


def test_imageui_collect_and_apply_aoi(tmp_path):
    existing_app = wx.App.Get()
    app = existing_app or wx.App(False)
    created = existing_app is None

    image_ui = ImageUI(None)
    image_ui.Hide()
    image_ui._settings_path = str(tmp_path / "image_settings.json")
    image_ui._save_settings = lambda: None
    image_ui.updateImageListBox = lambda: None
    image_ui.updateAOIImageAndProfiles = lambda: None
    try:
        # Seed distinct AOI values and capture the snapshot.
        image_ui.AOI1.ChangeValue("10")
        image_ui.AOI2.ChangeValue("20")
        image_ui.AOI3.ChangeValue("30")
        image_ui.AOI4.ChangeValue("40")
        snapshot = image_ui._collect_settings()
        assert snapshot["aoi"] == {
            "x_left": 10,
            "y_top": 20,
            "x_right": 30,
            "y_bottom": 40,
        }

        # Alter the controls, then ensure the snapshot restores them.
        image_ui.AOI1.ChangeValue("1")
        image_ui.AOI2.ChangeValue("2")
        image_ui.AOI3.ChangeValue("3")
        image_ui.AOI4.ChangeValue("4")

        snapshot["aoi"] = {
            "x_left": 11,
            "y_top": 21,
            "x_right": 31,
            "y_bottom": 41,
        }
        image_ui.apply_settings_snapshot(snapshot)

        assert image_ui.AOI1.GetValue() == "11"
        assert image_ui.AOI2.GetValue() == "21"
        assert image_ui.AOI3.GetValue() == "31"
        assert image_ui.AOI4.GetValue() == "41"
        assert image_ui.AOI == [[11, 21], [31, 41]]
    finally:
        image_ui.Destroy()
        if created:
            app.Destroy()


def test_session_bundle_includes_exclusions(tmp_path, monkeypatch):
    existing_app = wx.App.Get()
    app = existing_app or wx.App(False)
    created = existing_app is None

    window = FittingWindow(None)
    window.Hide()
    window._settings_path = str(tmp_path / "fit_settings.json")
    window._save_user_settings = lambda: None

    class DummySheet:
        def __init__(self, title="Run Data"):
            self.title = title

        def append(self, _row):
            pass

    class DummyWorkbook:
        def __init__(self):
            self.active = DummySheet()

        def create_sheet(self, title):
            return DummySheet(title)

        def save(self, _path):
            pass

    monkeypatch.setattr("fitting_window.Workbook", DummyWorkbook)

    window.results = {"image_file": ["img1", "img2"]}
    window.var_values = [1.0, 2.0]
    window.param_values = [0.1, 0.2]
    window.param2_values = [0.3, 0.4]
    window.plot_data = {1: {"excluded": {1}}}
    window.run_point_exclusions = {0: {0, 2}}
    window.subrun_results = [
        {"results": {"image_file": ["img1"]}, "var_values": [1.0]},
        {"results": {"image_file": ["img2"]}, "var_values": [2.0]},
    ]

    try:
        window._write_session_bundle("bundle", str(tmp_path), {1})
        bundle_path = tmp_path / "bundle.json"
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert data["fitting"]["excluded_indices"] == [1]
        assert data["fitting"]["run_exclusions"] == {"0": [0, 2]}
    finally:
        window.Destroy()
        if created:
            app.Destroy()
