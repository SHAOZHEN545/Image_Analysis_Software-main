"""UI window for capturing images from ThorCam cameras.

This module provides a wxPython-based interface that can connect to one or
more ThorLab cameras. Each connected camera runs its acquisition loop in a
separate thread so that image capture does not block the main application. For
hardware-triggered operation a three–image stack is written to a FITS file and
an absorption image preview is shown to the user.
"""

import math
import os
import json
import queue
import threading
import time
from datetime import datetime, date
from pathlib import PureWindowsPath
from typing import Optional

try:
    # Ensure ThorLabs DLLs are discoverable before importing the SDK
    from windows_setup import configure_path
    configure_path()
except Exception:
    pass

import numpy as np
from astropy.io import fits
import wx
import matplotlib
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib import cm
from collections import deque

from localPath import LOCAL_PATH
from imgFunc_v7 import createAbsorbImg
from svd_basis import SVDBasis, SVDBasisError, SVDLoadResult


SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "Camera Settings.txt")


DEFAULT_ATOM_SUBFOLDER = "Cs"
NETWORK_FALLBACK_BASE = PureWindowsPath("c:/Users/pdelf/Desktop/Image Network Folder")


PREVIEW_COLORMAP_OPTIONS = [
    ("Grey Scale", "gray_r"),
    ("Jet", "jet"),
]


def load_settings_file():
    try:
        with open(SETTINGS_FILE, "r") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_settings_file(data):
    with open(SETTINGS_FILE, "w") as fh:
        json.dump(data, fh, indent=2)

try:  # The ThorCam SDK may not be installed in all environments
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
    from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE, TRIGGER_POLARITY
except Exception:  # pragma: no cover
    TLCameraSDK = None
    OPERATION_MODE = None
    TRIGGER_POLARITY = None


class ThorCamWindow(wx.Frame):
    """Top-level window that hosts multiple camera panels."""

    def __init__(self, parent):
        super().__init__(parent, title="ThorCam Capture")
        self.sdk = None
        self.camera_panels = []

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Number of cameras (only 1 or 2 supported for now)
        settings = load_settings_file()
        count_sizer = wx.BoxSizer(wx.HORIZONTAL)
        count_sizer.Add(wx.StaticText(self, label="Cameras:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.camCountChoice = wx.Choice(self, choices=["1", "2"])
        init_count = settings.get("camera_count", 1)
        if init_count not in (1, 2):
            init_count = 1
        self.camCountChoice.SetStringSelection(str(init_count))
        count_sizer.Add(self.camCountChoice, 0)
        main_sizer.Add(count_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Container for per-camera panels
        self.camPanelContainer = wx.Panel(self)
        self.camGrid = wx.GridSizer(1, 1, 5, 5)
        self.camPanelContainer.SetSizer(self.camGrid)
        main_sizer.Add(self.camPanelContainer, 1, wx.ALL | wx.EXPAND, 5)

        self.SetSizerAndFit(main_sizer)

        self.camCountChoice.Bind(wx.EVT_CHOICE, self.on_cam_count)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Initialize SDK (if available)
        if TLCameraSDK is not None:
            try:
                self.sdk = TLCameraSDK()
            except Exception as exc:
                wx.MessageBox(
                    "Failed to load ThorLabs SDK: "
                    f"{exc}\nEnsure the 'SDK' and 'dlls' folders are present "
                    "or install the official ThorCam software.",
                    "ThorLabs SDK Error",
                    wx.OK | wx.ICON_ERROR,
                )

        # Build the requested number of camera panels up front so the
        # top-level frame is sized to the stored canvas dimensions.
        self.create_camera_panels(init_count)

        # Ensure the chosen camera count is persisted
        settings["camera_count"] = init_count
        save_settings_file(settings)

    # ------------------------------------------------------------------
    def on_cam_count(self, event):
        count = int(self.camCountChoice.GetStringSelection())
        self.create_camera_panels(count)
        data = load_settings_file()
        data["camera_count"] = count
        save_settings_file(data)

    def create_camera_panels(self, count):
        """Create the requested number of camera panels."""
        self.camGrid.Clear(True)
        for p in self.camera_panels:
            p.shutdown()
        self.camera_panels = []
        if count == 1:
            rows, cols = 1, 1
        else:
            rows, cols = 1, 2
        self.camGrid.SetRows(rows)
        self.camGrid.SetCols(cols)
        for i in range(count):
            panel = CameraPanel(self.camPanelContainer, self.sdk, i + 1)
            self.camera_panels.append(panel)
            self.camGrid.Add(panel, 1, wx.EXPAND)
            panel.on_refresh()
        self.camPanelContainer.Layout()
        self.Layout()
        self.Fit()

    # ------------------------------------------------------------------
    def on_close(self, event):
        for panel in self.camera_panels:
            panel.shutdown()
        if self.sdk is not None:
            try:
                self.sdk.dispose()
            except Exception:
                pass
        event.Skip()


class ImageCanvas(wx.Panel):
    """Panel that draws a bitmap with an outline and pixel axes."""

    def __init__(self, parent, display_scale=1.0):
        super().__init__(parent)
        self._display_scale = display_scale
        self._display_width = 0
        self._display_height = 0
        self._axis_width = 0
        self._axis_height = 0
        self._bitmap = None
        self._bitmap_buffer = None
        self._margin_left = 48
        self._margin_right = 18
        self._margin_top = 18
        self._margin_bottom = 32
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, self._on_size)

    # ------------------------------------------------------------------
    def set_display_scale(self, scale):
        self._display_scale = scale
        self.Refresh(False)

    def set_display_size(self, width, height):
        self._display_width = max(1, int(width))
        self._display_height = max(1, int(height))
        self._update_min_size()
        self.Refresh(False)

    def set_axis_dimensions(self, width, height):
        self._axis_width = max(0, int(width))
        self._axis_height = max(0, int(height))
        self.Refresh(False)

    def update_bitmap(self, bitmap, buffer, axis_size=None):
        self._bitmap = bitmap
        self._bitmap_buffer = buffer
        if bitmap is not None and bitmap.IsOk():
            self._display_width = bitmap.GetWidth()
            self._display_height = bitmap.GetHeight()
        if axis_size is not None:
            self.set_axis_dimensions(*axis_size)
        self._update_min_size()
        self.Refresh(False)

    # ------------------------------------------------------------------
    def _on_size(self, event):
        self.Refresh(False)
        event.Skip()

    def _update_min_size(self):
        total_w = self._margin_left + self._display_width + self._margin_right
        total_h = self._margin_top + self._display_height + self._margin_bottom
        self.SetMinSize((total_w, total_h))

    def _on_paint(self, _event):
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()

        client_w, client_h = self.GetClientSize()
        content_w = self._margin_left + self._display_width + self._margin_right
        content_h = self._margin_top + self._display_height + self._margin_bottom
        offset_x = max(0, (client_w - content_w) // 2)
        offset_y = max(0, (client_h - content_h) // 2)

        origin_x = self._margin_left + offset_x
        origin_y = self._margin_top + offset_y
        rect = wx.Rect(origin_x, origin_y, self._display_width, self._display_height)

        if self._bitmap is not None and self._bitmap.IsOk():
            dc.DrawBitmap(self._bitmap, rect.x, rect.y, True)

        axis_colour = wx.Colour(200, 200, 200)
        dc.SetPen(wx.Pen(axis_colour, width=1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(rect)

        self._draw_axes(dc, rect, axis_colour)

    def _draw_axes(self, dc, rect, colour):
        if self._axis_width <= 0 or self._axis_height <= 0:
            return

        scale_x = rect.width / float(self._axis_width)
        scale_y = rect.height / float(self._axis_height)
        tick_len = 5
        dc.SetTextForeground(colour)

        # X-axis ticks along the bottom of the image
        base_y = rect.y + rect.height
        x_ticks = self._generate_ticks(self._axis_width)
        for value in x_ticks:
            px = rect.x + int(round(value * scale_x))
            dc.DrawLine(px, base_y, px, base_y + tick_len)
            label = str(value)
            tw, th = dc.GetTextExtent(label)
            dc.DrawText(label, px - tw // 2, base_y + tick_len + 2)

        # Y-axis ticks along the left side of the image (origin at top)
        base_x = rect.x
        y_ticks = self._generate_ticks(self._axis_height)
        for value in y_ticks:
            py = rect.y + int(round(value * scale_y))
            dc.DrawLine(base_x - tick_len, py, base_x, py)
            label = str(value)
            tw, th = dc.GetTextExtent(label)
            dc.DrawText(label, base_x - tick_len - tw - 2, py - th // 2)

    def _generate_ticks(self, length):
        if length <= 0:
            return []
        step = self._nice_tick_step(length)
        ticks = list(range(0, length, step))
        if ticks[-1] != length:
            ticks.append(length)
        return ticks

    @staticmethod
    def _nice_tick_step(length):
        if length <= 0:
            return 1
        desired = max(1, length // 5)
        magnitude = 10 ** int(math.floor(math.log10(desired))) if desired > 0 else 1
        for multiplier in (1, 2, 5, 10):
            step = multiplier * magnitude
            if step >= desired:
                return max(1, step)
        return max(1, 10 * magnitude)


class SVDGenerationDialog(wx.Dialog):
    """Dialog that gathers parameters and tracks progress while building an SVD basis."""

    def __init__(self, parent, initial_dir: str, default_refs: int, default_k: int):
        super().__init__(
            parent,
            title="Generate SVD Basis",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        self._start_handler = None
        self._cancel_handler = None
        self._running = False
        self.cancel_event = threading.Event()
        self._progress_range = 1

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        folder_row = wx.BoxSizer(wx.HORIZONTAL)
        folder_row.Add(wx.StaticText(panel, label="Reference folder:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.folder_ctrl = wx.TextCtrl(panel, style=wx.TE_READONLY)
        folder_row.Add(self.folder_ctrl, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.browse_btn = wx.Button(panel, label="Browse…")
        folder_row.Add(self.browse_btn, 0)
        main_sizer.Add(folder_row, 0, wx.EXPAND | wx.ALL, 5)

        grid = wx.FlexGridSizer(0, 2, 5, 5)
        grid.AddGrowableCol(1)
        grid.Add(wx.StaticText(panel, label="# reference FITS to use:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.refs_ctrl = wx.SpinCtrl(panel, min=5, max=10_000, initial=max(5, default_refs))
        grid.Add(self.refs_ctrl, 0, wx.EXPAND)
        grid.Add(wx.StaticText(panel, label="# SVD modes k:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.modes_ctrl = wx.SpinCtrl(panel, min=1, max=500, initial=max(1, default_k))
        grid.Add(self.modes_ctrl, 0, wx.EXPAND)
        main_sizer.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.status_label = wx.StaticText(panel, label="Select a folder and press Start.")
        main_sizer.Add(self.status_label, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.progress = wx.Gauge(panel, range=1, size=(200, -1))
        self.progress.SetValue(0)
        main_sizer.Add(self.progress, 0, wx.EXPAND | wx.ALL, 5)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(panel, label="Start")
        self.start_btn.SetDefault()
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL)
        btn_sizer.Add(self.start_btn, 0, wx.RIGHT, 8)
        btn_sizer.Add(self.cancel_btn, 0)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        panel.SetSizer(main_sizer)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)
        self.SetMinSize((420, self.GetBestSize().height))

        self._initial_dir = initial_dir
        if initial_dir:
            self.folder_ctrl.SetValue(initial_dir)
        self._update_start_enabled()

        self.browse_btn.Bind(wx.EVT_BUTTON, self._on_browse)
        self.start_btn.Bind(wx.EVT_BUTTON, self._on_start)
        self.cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.Bind(wx.EVT_CLOSE, self._on_close)

    # ------------------------------------------------------------------
    def set_start_handler(self, handler):
        self._start_handler = handler

    def set_cancel_handler(self, handler):
        self._cancel_handler = handler

    def _update_start_enabled(self):
        enable = bool(self.folder_ctrl.GetValue()) and not self._running
        self.start_btn.Enable(enable)

    def _on_browse(self, event):
        initial_dir = self.folder_ctrl.GetValue() or self._initial_dir or ""
        dlg = wx.DirDialog(self, "Select SVD Reference Folder", initial_dir)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.folder_ctrl.SetValue(path)
            self._initial_dir = path
        dlg.Destroy()
        self._update_start_enabled()

    def _on_start(self, event):
        if self._running:
            return
        folder = self.folder_ctrl.GetValue()
        if not folder:
            wx.MessageBox(
                "Select a reference folder before starting.",
                "Single SVD",
                wx.OK | wx.ICON_WARNING,
                parent=self,
            )
            return
        n_refs = self.refs_ctrl.GetValue()
        k = self.modes_ctrl.GetValue()
        started = False
        if self._start_handler is not None:
            started = bool(self._start_handler(folder, n_refs, k))
        if started:
            self.cancel_event.clear()
            self._running = True
            self.status_label.SetLabel("Preparing…")
            self.progress.SetRange(1)
            self.progress.SetValue(0)
            self._update_start_enabled()
            self.cancel_btn.Enable(True)
            self.refs_ctrl.Enable(False)
            self.modes_ctrl.Enable(False)
            self.browse_btn.Enable(False)
        else:
            self.cancel_event.clear()

    def _on_cancel(self, event):
        if self._running:
            self.status_label.SetLabel("Cancelling…")
            self.cancel_btn.Enable(False)
            self.cancel_event.set()
            if self._cancel_handler is not None:
                self._cancel_handler()
        else:
            self.EndModal(wx.ID_CANCEL)

    def _on_close(self, event):
        if self._running:
            self._on_cancel(event)
        else:
            event.Skip()

    # ------------------------------------------------------------------
    def on_generation_started(self, maximum):
        self._progress_range = max(1, int(maximum))
        self.progress.SetRange(self._progress_range)
        self.progress.SetValue(0)

    def update_progress(self, processed, target, path):
        target = max(1, int(target))
        if target != self._progress_range:
            self._progress_range = target
            self.progress.SetRange(self._progress_range)
        value = max(0, min(int(processed), self._progress_range))
        self.progress.SetValue(value)
        label = f"Processed {value} of {self._progress_range}"
        if path:
            label += f"\n{os.path.basename(path)}"
        self.status_label.SetLabel(label)

    def report_success(self, basis_path):
        if basis_path:
            message = f"Saved basis: {os.path.basename(basis_path)}"
        else:
            message = "SVD basis generated."
        self.status_label.SetLabel(message)
        self.progress.SetValue(self._progress_range)
        self.cancel_btn.Enable(False)
        self._running = False
        self.refs_ctrl.Enable(True)
        self.modes_ctrl.Enable(True)
        self.browse_btn.Enable(True)
        self._update_start_enabled()
        self.EndModal(wx.ID_OK)

    def report_cancelled(self):
        self.status_label.SetLabel("Generation cancelled.")
        self._running = False
        self.refs_ctrl.Enable(True)
        self.modes_ctrl.Enable(True)
        self.browse_btn.Enable(True)
        self._update_start_enabled()
        self.EndModal(wx.ID_CANCEL)

    def report_error(self, message):
        wx.MessageBox(message, "Single SVD", wx.OK | wx.ICON_ERROR, parent=self)
        self.status_label.SetLabel("Error occurred. Adjust settings and try again.")
        self.cancel_btn.Enable(True)
        self.cancel_event.clear()
        self._running = False
        self.refs_ctrl.Enable(True)
        self.modes_ctrl.Enable(True)
        self.browse_btn.Enable(True)
        self._update_start_enabled()


class CameraPanel(wx.Panel):
    """UI panel and capture thread for a single camera."""

    def __init__(self, parent, sdk, idx):
        super().__init__(parent)
        self.sdk = sdk
        self.index = idx
        self.running = False
        self.worker = None
        self.camera = None
        self.cam_info = None
        self.available_cams = []
        self.run_mode = "Continuous"
        self.exposure_us = 1000
        self.gain = 0
        self.black_level = 0
        self.bin_x = 1
        self.bin_y = 1
        self.intensity_window = 10
        self.intensity_scale = 1.0
        self._latest_frame = None
        self._latest_fps = None
        self._latest_label = None
        self._current_overlay_text = ""
        self._t0 = None
        self._int_line = None
        self._frame_queue = queue.Queue(maxsize=4)
        self._od_range = (0.0, 5.0)
        self._display_scale = 0.6
        self._display_width = int(512 * self._display_scale)
        self._display_height = int(512 * self._display_scale)
        # None = no acquisition throttle; GUI already coalesces frames in _on_paint_tick
        self.max_continuous_fps = None
        # Optional: UI-only display cap (None for unlimited). Drawing can be expensive; 30 is smooth.
        self.display_fps_cap = 30.0
        self._last_display_push = 0.0
        self._overlay_font = wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self._raw_bit_depth = 12
        self._raw_max_value = float((1 << self._raw_bit_depth) - 1)
        self._raw_scale = 255.0 / self._raw_max_value
        self._colormap_options = PREVIEW_COLORMAP_OPTIONS
        self._selected_colormap_name = "gray_r"
        self._rebuild_colormap_lut()
        self._paint_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_paint_tick, self._paint_timer)
        self._paint_timer.Start(16)
        # Continuous mode does not require a save directory. The folder will be created if the user later enables hardware triggering.
        self.save_folder = ""
        self.today = date.today()
        self.popout_frame = None
        self._hardware_stack_queue = queue.Queue(maxsize=6)
        self._postproc_thread = threading.Thread(target=self._hardware_postproc_loop, daemon=True)
        self._postproc_thread.start()
        self.capture_mode = "3 Shot Absorption"
        self.svd_basis: Optional[SVDBasis] = None
        self.svd_reference_folder = ""
        self.svd_basis_path = ""
        self.svd_default_refs = 100
        self.svd_default_k = 10
        self._svd_loading = False
        self._svd_loader_thread: Optional[threading.Thread] = None
        self._svd_cancel_event: Optional[threading.Event] = None
        self._svd_generation_dialog: Optional[SVDGenerationDialog] = None
        self._svd_previous_status = "Not loaded"

        box = wx.StaticBoxSizer(wx.VERTICAL, self, f"Camera {idx}")

        # Camera selection row
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(self, label="Camera:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.camChoice = wx.Choice(self)
        row.Add(self.camChoice, 1, wx.RIGHT, 5)
        self.refreshBtn = wx.Button(self, label="Refresh")
        row.Add(self.refreshBtn, 0, wx.RIGHT, 5)
        self.toggleBtn = wx.ToggleButton(self, label="Connect")
        row.Add(self.toggleBtn, 0)
        box.Add(row, 0, wx.ALL | wx.EXPAND, 5)

        # Run mode and mode selections aligned horizontally (run mode first)
        mode_run_row = wx.BoxSizer(wx.HORIZONTAL)

        run_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Run Mode")
        self.runModeBox = wx.RadioBox(
            self,
            choices=["Continuous", "Hardware Trigger"],
            majorDimension=1,
            style=wx.RA_SPECIFY_ROWS,
        )
        self.runModeBox.SetSelection(0)
        run_box.Add(self.runModeBox, 0, wx.ALL, 5)
        mode_run_row.Add(run_box, 1, wx.EXPAND | wx.RIGHT, 5)

        mode_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Mode")
        self.modeBox = wx.RadioBox(
            self,
            choices=["3 Shot Absorption", "Single SVD"],
            majorDimension=1,
            style=wx.RA_SPECIFY_ROWS,
        )
        mode_box.Add(self.modeBox, 0, wx.ALL, 5)
        mode_run_row.Add(mode_box, 1, wx.EXPAND)

        box.Add(mode_run_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.svd_panel = wx.Panel(self)
        svd_box = wx.StaticBoxSizer(wx.StaticBox(self.svd_panel, label="Single SVD Options"), wx.VERTICAL)

        svd_row = wx.BoxSizer(wx.HORIZONTAL)
        self.loadSVDBtn = wx.Button(self.svd_panel, label="Load SVD Reference…")
        svd_row.Add(self.loadSVDBtn, 0, wx.RIGHT, 5)
        self.generateSVDBtn = wx.Button(self.svd_panel, label="Generate SVD Basis…")
        svd_row.Add(self.generateSVDBtn, 0, wx.RIGHT, 5)
        self.svdStatusLabel = wx.StaticText(self.svd_panel, label="Not loaded")
        svd_row.Add(self.svdStatusLabel, 1, wx.ALIGN_CENTER_VERTICAL)
        svd_box.Add(svd_row, 0, wx.EXPAND | wx.ALL, 5)

        basis_row = wx.BoxSizer(wx.HORIZONTAL)
        basis_row.Add(
            wx.StaticText(self.svd_panel, label="Basis file:"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            5,
        )
        self.svdBasisCtrl = wx.TextCtrl(self.svd_panel, style=wx.TE_READONLY)
        basis_row.Add(self.svdBasisCtrl, 1, wx.EXPAND)
        svd_box.Add(basis_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.svd_panel.SetSizer(svd_box)
        box.Add(self.svd_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.svd_panel.Hide()

        # Camera settings split into acquisition and intensity sub-panels
        settings_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Camera Settings")
        settings_row = wx.BoxSizer(wx.HORIZONTAL)

        acq_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Acquisition")
        acq_grid = wx.FlexGridSizer(0, 2, 5, 5)
        acq_grid.Add(wx.StaticText(self, label="Exposure (us):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.exposureCtrl = wx.SpinCtrl(self, min=1, max=1_000_000, initial=1000)
        acq_grid.Add(self.exposureCtrl, 0)
        acq_grid.Add(wx.StaticText(self, label="Gain:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.gainCtrl = wx.SpinCtrl(self, min=0, max=100, initial=0)
        acq_grid.Add(self.gainCtrl, 0)
        acq_grid.Add(wx.StaticText(self, label="Black Level:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.blackCtrl = wx.SpinCtrl(self, min=0, max=255, initial=0)
        acq_grid.Add(self.blackCtrl, 0)
        acq_grid.Add(wx.StaticText(self, label="Binning:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.binChoice = wx.Choice(self, choices=["1x1", "2x2", "4x4"])
        self.binChoice.SetSelection(0)
        acq_grid.Add(self.binChoice, 0)
        acq_grid.Add(wx.StaticText(self, label="Colormap:"), 0, wx.ALIGN_CENTER_VERTICAL)
        colormap_labels = [label for label, _ in self._colormap_options]
        text_widths = [self.GetTextExtent(label)[0] for label in colormap_labels]
        choice_width = max(text_widths, default=60) + 16  # Keep dropdown compact similar to previous checkbox width
        self.colormapChoice = wx.Choice(self, choices=colormap_labels, size=(choice_width, -1))
        self.colormapChoice.SetToolTip("Select the color map for camera previews")
        default_idx = next(
            (idx for idx, (_, cmap) in enumerate(self._colormap_options) if cmap == "gray_r"),
            0,
        )
        self.colormapChoice.SetSelection(default_idx)
        self._selected_colormap_name = self._colormap_options[default_idx][1]
        acq_grid.Add(self.colormapChoice, 0)
        acq_box.Add(acq_grid, 0, wx.ALL, 5)

        intensity_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Intensity")
        int_grid = wx.FlexGridSizer(0, 2, 5, 5)
        int_grid.Add(wx.StaticText(self, label="Intensity Window (s):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.windowCtrl = wx.SpinCtrl(
            self,
            min=1,
            max=86_400,
            initial=self.intensity_window,
        )
        self.windowCtrl.SetToolTip("Set the intensity history window (up to 24 hours)")
        int_grid.Add(self.windowCtrl, 0)
        int_grid.Add(wx.StaticText(self, label="Intensity Max:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.scaleCtrl = wx.TextCtrl(self, value=str(self.intensity_scale))
        int_grid.Add(self.scaleCtrl, 0)
        intensity_box.Add(int_grid, 0, wx.ALL, 5)

        settings_row.Add(acq_box, 0, wx.EXPAND | wx.RIGHT, 5)
        settings_row.Add(intensity_box, 0, wx.EXPAND)
        settings_box.Add(settings_row, 0, wx.EXPAND)
        box.Add(settings_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Display bitmap for the main image and matplotlib figure for the
        # intensity trace beneath it.
        self._img_width = 512
        self._img_height = 512
        self.image_canvas = ImageCanvas(self, self._display_scale)
        self.image_canvas.set_display_size(self._display_width, self._display_height)
        self.image_canvas.set_axis_dimensions(self._img_width, self._img_height)
        box.Add(self.image_canvas, 1, wx.ALL | wx.EXPAND, 5)

        self.preview_titles = ["Probe + Atoms", "Probe Only", "Dark Field"]
        self.preview_axes = []
        self.preview_images = []
        self._preview_limits = []
        self._last_three_shot_frames = None
        self._showing_three_layout = None
        self._base_canvas_height = int(round(150 * 0.75))
        self._intensity_height_ratio = self._base_canvas_height / 512.0
        self._three_shot_height_ratio = self._intensity_height_ratio
        fig = Figure(figsize=(5, self._base_canvas_height / 100.0))
        self.canvas = FigureCanvas(self, -1, fig)
        self.canvas.SetMinSize((512, self._base_canvas_height))
        box.Add(self.canvas, 0, wx.ALL | wx.EXPAND, 5)
        self.intensity_ax = None
        self._configure_plot_layout(show_three=False)
        # Container for intensity data (time, total intensity)
        self.intensity_history = deque()
        self._intensity_time = 0.0
        self.intensity_paused = False
        self.last_raw_image = None
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        toolbar_row = wx.BoxSizer(wx.HORIZONTAL)
        toolbar_row.Add(self.toolbar, 1, wx.EXPAND)
        self.pauseBtn = wx.ToggleButton(self, label="Pause Graph")
        toolbar_row.Add(self.pauseBtn, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        self.popoutCheck = wx.CheckBox(self, label="Pop Out")
        toolbar_row.Add(self.popoutCheck, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        self.flipHCheck = wx.CheckBox(self, label="H Flip")
        toolbar_row.Add(self.flipHCheck, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        self.flipVCheck = wx.CheckBox(self, label="V Flip")
        self.flipVCheck.SetValue(True)
        toolbar_row.Add(self.flipVCheck, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        box.Add(toolbar_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.canvas.draw()

        # Log output
        self.logCtrl = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.logCtrl.SetMinSize((-1, 80))
        box.Add(self.logCtrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Save folder controls
        path_row = wx.BoxSizer(wx.HORIZONTAL)
        self.folderCtrl = wx.TextCtrl(self, value=self.save_folder)
        self.browseBtn = wx.Button(self, label="Choose...")
        path_row.Add(self.folderCtrl, 1, wx.RIGHT, 5)
        path_row.Add(self.browseBtn, 0)
        box.Add(path_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.SetSizerAndFit(box)

        self.refreshBtn.Bind(wx.EVT_BUTTON, self.on_refresh)
        self.toggleBtn.Bind(wx.EVT_TOGGLEBUTTON, self.on_toggle)
        self.browseBtn.Bind(wx.EVT_BUTTON, self.on_browse)
        self.camChoice.Bind(wx.EVT_CHOICE, self.on_cam_select)
        self.modeBox.Bind(wx.EVT_RADIOBOX, self.on_mode_change)
        self.runModeBox.Bind(wx.EVT_RADIOBOX, self.on_run_mode_change)
        self.exposureCtrl.Bind(wx.EVT_SPINCTRL, self.on_exposure_change)
        self.gainCtrl.Bind(wx.EVT_SPINCTRL, self.on_gain_change)
        self.blackCtrl.Bind(wx.EVT_SPINCTRL, self.on_black_level_change)
        self.binChoice.Bind(wx.EVT_CHOICE, self.on_bin_change)
        self.windowCtrl.Bind(wx.EVT_SPINCTRL, self.on_window_change)
        self.scaleCtrl.Bind(wx.EVT_TEXT, self.on_scale_change)
        self.folderCtrl.Bind(wx.EVT_TEXT, self.on_folder_change)
        self.popoutCheck.Bind(wx.EVT_CHECKBOX, self.on_popout)
        self.pauseBtn.Bind(wx.EVT_TOGGLEBUTTON, self.on_pause_graph)
        self.flipHCheck.Bind(wx.EVT_CHECKBOX, self.on_flip_change)
        self.flipVCheck.Bind(wx.EVT_CHECKBOX, self.on_flip_change)
        self.colormapChoice.Bind(wx.EVT_CHOICE, self.on_colormap_changed)
        self.loadSVDBtn.Bind(wx.EVT_BUTTON, self.on_load_svd_reference)
        self.generateSVDBtn.Bind(wx.EVT_BUTTON, self.on_generate_svd_basis)

        self.last_serial = self.load_last_serial()
        self.update_mode_availability()
        self._update_single_svd_ui()

    # ------------------------------------------------------------------
    def log(self, msg):
        self.logCtrl.AppendText(msg + "\n")

    def on_refresh(self, event=None):
        if self.sdk is None:
            self.camChoice.SetItems([])
            self.log("ThorLabs SDK not available")
            return
        serials = self.sdk.discover_available_cameras()
        self.available_cams = serials
        self.camChoice.SetItems(serials)
        if serials:
            if self.last_serial in serials:
                sel = serials.index(self.last_serial)
            else:
                sel = 0
                self.last_serial = serials[0]
                self.save_last_serial(self.last_serial)
            self.camChoice.SetSelection(sel)
            self.apply_saved_settings(self.last_serial)
            self.refresh_preview_visibility()
        else:
            self.log("No cameras detected")

    def on_toggle(self, event):
        if self.toggleBtn.GetValue():
            self.start()
        else:
            self.stop()

    def on_browse(self, event):
        dlg = wx.DirDialog(self, "Select Save Folder", self.save_folder)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.folderCtrl.SetValue(path)
            self.save_folder = path
        dlg.Destroy()

    def on_cam_select(self, event):
        serial = self.camChoice.GetStringSelection()
        if serial:
            self.last_serial = serial
            self.save_last_serial(serial)
            self.apply_saved_settings(serial)
            self.refresh_preview_visibility()

    def on_mode_change(self, event):
        self.capture_mode = self.modeBox.GetStringSelection()
        self.refresh_preview_visibility()
        self._update_single_svd_ui()
        if self.cam_info:
            self.save_current_settings()

    def on_run_mode_change(self, event):
        self.run_mode = self.runModeBox.GetStringSelection()
        self._last_display_push = 0.0
        self.update_mode_availability()
        self._update_start_button_state()

    def on_exposure_change(self, event):
        self.exposure_us = self.exposureCtrl.GetValue()

    def on_gain_change(self, event):
        self.gain = self.gainCtrl.GetValue()

    def on_black_level_change(self, event):
        self.black_level = self.blackCtrl.GetValue()

    def on_bin_change(self, event):
        sel = self.binChoice.GetStringSelection()
        try:
            self.bin_x, self.bin_y = map(int, sel.split("x"))
        except Exception:
            self.bin_x = self.bin_y = 1

    def on_window_change(self, event):
        self.intensity_window = self.windowCtrl.GetValue()
        while self.intensity_history and self.intensity_history[0][0] < self._intensity_time - self.intensity_window:
            self.intensity_history.popleft()
        if self.intensity_ax is not None:
            self.intensity_ax.set_xlim(-self.intensity_window, 0)
            self.canvas.draw_idle()
        if self.cam_info:
            self.save_current_settings()

    def on_scale_change(self, event):
        try:
            val = float(self.scaleCtrl.GetValue())
            if val <= 0:
                return
            self.intensity_scale = val
            if self.intensity_ax is not None:
                self.intensity_ax.set_ylim(0, self.intensity_scale)
                self.canvas.draw_idle()
            if self.cam_info:
                self.save_current_settings()
        except ValueError:
            pass

    def _is_svd_basis_ready(self):
        return self.svd_basis is not None and getattr(self.svd_basis, "loaded", False)

    def _set_svd_status(self, text):
        self.svdStatusLabel.SetLabel(text)

    def _update_single_svd_ui(self):
        is_single = self.modeBox.GetStringSelection() == "Single SVD"
        self.svd_panel.Show(is_single)
        self.svd_panel.Enable(is_single)
        self.Layout()
        self._update_start_button_state()

    def _update_start_button_state(self):
        if self.running:
            self.toggleBtn.Enable(True)
            return
        need_basis = (
            self.runModeBox.GetStringSelection() == "Hardware Trigger"
            and self.modeBox.GetStringSelection() == "Single SVD"
        )
        enabled = True
        if need_basis and not self._is_svd_basis_ready():
            enabled = False
        self.toggleBtn.Enable(enabled)

    def _update_preview_titles(self):
        mode = self.modeBox.GetStringSelection()
        if mode == "Single SVD":
            titles = ["Atom", "SVD Background", "Dark"]
        else:
            titles = ["Probe + Atoms", "Probe Only", "Dark Field"]
        if getattr(self, "preview_titles", None) != titles:
            self.preview_titles = titles

    def on_load_svd_reference(self, event):
        if self._svd_loading:
            return
        wildcard = "SVD basis files (*.npz)|*.npz|All files|*.*"
        current = self.svdBasisCtrl.GetValue()
        if current:
            initial_dir = os.path.dirname(current)
        elif self.svd_basis_path:
            initial_dir = os.path.dirname(self.svd_basis_path)
        else:
            initial_dir = self.svd_reference_folder or self.save_folder or ""
        dlg = wx.FileDialog(
            self,
            "Select SVD Basis File",
            initial_dir,
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        path = dlg.GetPath()
        dlg.Destroy()

        basis = SVDBasis()
        try:
            result = basis.load_from_file(path)
        except SVDBasisError as exc:
            self.log(f"SVD basis file load failed: {exc}")
            wx.MessageBox(
                f"Failed to load SVD basis file\n{exc}",
                "Single SVD",
                wx.OK | wx.ICON_ERROR,
            )
            return

        self._on_svd_load_success(basis, result)
    def on_generate_svd_basis(self, event):
        if self._svd_loading:
            return
        initial_dir = self.svd_reference_folder or self.save_folder or ""
        dialog = SVDGenerationDialog(self, initial_dir, self.svd_default_refs, self.svd_default_k)
        dialog.set_start_handler(lambda folder, n_refs, k: self._begin_svd_generation(folder, n_refs, k, dialog))
        dialog.set_cancel_handler(self._on_svd_generation_cancel_requested)
        self._svd_generation_dialog = dialog
        dialog.ShowModal()
        dialog.Destroy()
        self._svd_generation_dialog = None

    def _begin_svd_generation(self, folder, n_refs, k, dialog):
        if self._svd_loading:
            return False
        if not os.path.isdir(folder):
            dialog.report_error("Selected folder does not exist.")
            return False
        fits_count = sum(
            1 for name in os.listdir(folder) if name.lower().endswith((".fits", ".fit"))
        )
        if fits_count == 0:
            dialog.report_error("No FITS files were found in the selected folder.")
            return False

        self.svd_reference_folder = folder
        self.svd_default_refs = max(5, min(int(n_refs), 10_000))
        self.svd_default_k = max(1, min(int(k), 500))
        self._svd_previous_status = self.svdStatusLabel.GetLabel()
        self._svd_cancel_event = dialog.cancel_event
        self._svd_loading = True
        self.loadSVDBtn.Enable(False)
        self.generateSVDBtn.Enable(False)
        self._set_svd_status("Generating…")

        maximum = max(1, min(self.svd_default_refs, fits_count))
        dialog.on_generation_started(maximum)

        basis = SVDBasis()

        def progress(processed, target, path):
            wx.CallAfter(dialog.update_progress, processed, target or maximum, path)

        def worker():
            try:
                result = basis.load_from_folder(
                    folder,
                    self.svd_default_refs,
                    self.svd_default_k,
                    cancel_event=self._svd_cancel_event,
                    progress_callback=progress,
                )
            except SVDBasisError as exc:
                cancelled = "cancelled" in str(exc).lower()
                wx.CallAfter(self._on_svd_load_error, str(exc), cancelled)
            except Exception as exc:  # pragma: no cover - defensive
                wx.CallAfter(self._on_svd_load_error, str(exc), False)
            else:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_svd.npz"
                    save_path = os.path.join(folder, filename)
                    counter = 1
                    while os.path.exists(save_path):
                        filename = f"{timestamp}_svd_{counter}.npz"
                        save_path = os.path.join(folder, filename)
                        counter += 1
                    basis.save_to_file(save_path)
                    result.basis_path = save_path
                except SVDBasisError as exc:
                    wx.CallAfter(
                        self._on_svd_load_error,
                        f"Failed to save SVD basis: {exc}",
                        False,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    wx.CallAfter(
                        self._on_svd_load_error,
                        f"Failed to save SVD basis: {exc}",
                        False,
                    )
                else:
                    wx.CallAfter(self._on_svd_load_success, basis, result)
            finally:
                wx.CallAfter(self._finish_svd_load)

        thread = threading.Thread(target=worker, daemon=True)
        self._svd_loader_thread = thread
        thread.start()
        return True

    def _on_svd_generation_cancel_requested(self):
        if self._svd_cancel_event is not None:
            self._svd_cancel_event.set()
        self.log("SVD basis generation cancellation requested")

    def _finish_svd_load(self):
        self._svd_cancel_event = None
        self._svd_loader_thread = None
        self._svd_loading = False
        self.loadSVDBtn.Enable(True)
        self.generateSVDBtn.Enable(True)
        self._update_start_button_state()

    def _on_svd_load_success(self, basis, result):
        self.svd_basis = basis
        self.svd_reference_folder = result.folder
        self.svd_basis_path = result.basis_path or ""
        if self.svd_basis_path:
            self.svdBasisCtrl.SetValue(self.svd_basis_path)
        else:
            self.svdBasisCtrl.SetValue("")
        if self.svd_basis_path:
            basis.basis_path = self.svd_basis_path
        status = f"Loaded ({result.n_refs} refs, k={result.k})"
        self._set_svd_status(status)
        self._svd_previous_status = status
        self._update_start_button_state()
        self.log(
            f"SVD basis loaded from {result.folder} | refs={result.n_refs} | k={result.k}"
        )
        if self.svd_basis_path:
            self.log(f"SVD basis file: {self.svd_basis_path}")
        if result.singular_values:
            formatted = ", ".join(f"{sv:.3g}" for sv in result.singular_values[:5])
            self.log(f"Leading singular values: {formatted}")
        if result.normalization_factors:
            arr = np.asarray(result.normalization_factors, dtype=float)
            self.log(
                (
                    "Normalization α median="
                    f"{np.median(arr):.3g}, range={arr.min():.3g}–{arr.max():.3g}"
                )
            )
        if self._svd_generation_dialog is not None:
            self._svd_generation_dialog.report_success(self.svd_basis_path)

    def _on_svd_load_error(self, message, cancelled):
        dialog = self._svd_generation_dialog
        if cancelled:
            self._set_svd_status(self._svd_previous_status)
            self.log("SVD basis loading cancelled")
            if dialog is not None:
                dialog.report_cancelled()
            return
        self.svd_basis = None
        self.svd_reference_folder = ""
        self.svd_basis_path = ""
        self.svdBasisCtrl.SetValue("")
        self._set_svd_status("Not loaded")
        self._svd_previous_status = "Not loaded"
        self.log(f"SVD basis load failed: {message}")
        text = f"Failed to build SVD basis:\n{message}"
        if dialog is not None:
            dialog.report_error(text)
        else:
            wx.MessageBox(
                text,
                "Single SVD",
                wx.OK | wx.ICON_ERROR,
            )
    def _configure_plot_layout(self, show_three, force=False):
        """Rebuild the matplotlib layout for the current mode."""

        titles_key = tuple(self.preview_titles)
        if (
            not force
            and self._showing_three_layout == show_three
            and getattr(self, "_preview_titles_key", None) == titles_key
        ):
            return

        fig = self.canvas.figure
        fig.clf()
        self.preview_axes = []
        self.preview_images = []
        self._preview_limits = []

        self._int_line = None
        if show_three:
            gs = fig.add_gridspec(1, len(self.preview_titles), wspace=0.3)
            self.intensity_ax = None
            for idx, title in enumerate(self.preview_titles):
                ax = fig.add_subplot(gs[0, idx])
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                try:
                    ax.set_box_aspect(1)
                except AttributeError:
                    ax.set_aspect("equal", adjustable="box")
                self.preview_axes.append(ax)
            self.preview_images = [None] * len(self.preview_axes)
            self._preview_limits = [None] * len(self.preview_axes)
            fig.subplots_adjust(left=0.03, right=0.97, bottom=0.12, top=0.88, wspace=0.25)
        else:
            gs = fig.add_gridspec(1, 1)
            self.intensity_ax = fig.add_subplot(gs[0, 0])
            self.intensity_ax.set_xlabel("Time (s)")
            self.intensity_ax.set_ylabel("Normalized Intensity")
            self.intensity_ax.set_xlim(-self.intensity_window, 0)
            self.intensity_ax.set_ylim(0, self.intensity_scale)
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.92)
        self._showing_three_layout = show_three
        self._preview_titles_key = titles_key
        self.canvas.draw_idle()

    def _render_three_shot_previews(self, frames):
        if not frames or not self.preview_axes:
            return
        count = min(len(frames), len(self.preview_axes))
        for idx in range(count):
            ax = self.preview_axes[idx]
            frame = frames[idx]
            data = self._apply_flips(np.asarray(frame))
            if data.size:
                vmin = float(np.min(data))
                vmax = float(np.max(data))
                if not np.isfinite(vmin):
                    vmin = 0.0
                if not np.isfinite(vmax):
                    vmax = vmin + 1.0
                if vmax <= vmin:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
            ax.set_visible(True)
            if self.preview_images[idx] is None:
                self.preview_images[idx] = ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
            else:
                self.preview_images[idx].set_data(data)
            self.preview_images[idx].set_clim(vmin, vmax)
            self._preview_limits[idx] = (vmin, vmax)
        for idx in range(count, len(self.preview_axes)):
            if self.preview_images[idx] is not None:
                try:
                    self.preview_images[idx].remove()
                except ValueError:
                    pass
                self.preview_images[idx] = None
        self.canvas.draw_idle()

    def refresh_preview_visibility(self):
        cont = self.runModeBox.GetStringSelection() == "Continuous"
        mode = self.modeBox.GetStringSelection()
        is_three_shot = mode == "3 Shot Absorption"
        is_single_svd = mode == "Single SVD"
        show_three = not cont and (is_three_shot or is_single_svd)
        show_canvas = cont or show_three
        self.canvas.Show(show_canvas)
        self.toolbar.Show(show_canvas)
        prev_titles = getattr(self, "_preview_titles_key", tuple(self.preview_titles))
        self._update_preview_titles()
        force_layout = tuple(self.preview_titles) != prev_titles
        self._configure_plot_layout(show_three, force=force_layout)
        if not show_three and self.intensity_ax is not None:
            self.intensity_ax.set_xlim(-self.intensity_window, 0)
            self.intensity_ax.set_ylim(0, self.intensity_scale)
            self.canvas.draw_idle()
        if show_three and self._last_three_shot_frames:
            self._render_three_shot_previews(self._last_three_shot_frames)
        if not show_three:
            self.clear_three_shot_previews()
        self.colormapChoice.Enable(cont)
        self.canvas.draw_idle()
        self.Layout()

    def clear_three_shot_previews(self, drop_cache=False):
        for idx, artist in enumerate(self.preview_images):
            if artist is not None:
                try:
                    artist.remove()
                except ValueError:
                    pass
                self.preview_images[idx] = None
        for ax in self.preview_axes:
            ax.set_visible(False)
        if drop_cache:
            self._last_three_shot_frames = None
        if self._preview_limits:
            self._preview_limits = [None] * len(self.preview_axes)

    def update_three_shot_previews(self, frames):
        if not frames:
            return
        cont = self.runModeBox.GetStringSelection() == "Continuous"
        mode = getattr(self, "capture_mode", self.modeBox.GetStringSelection())
        if cont or mode not in ("3 Shot Absorption", "Single SVD"):
            return
        limited = []
        for fr in frames:
            if fr is None:
                continue
            limited.append(np.asarray(fr).copy())
            if len(limited) == len(self.preview_titles):
                break
        if len(limited) != len(self.preview_titles):
            return
        self._last_three_shot_frames = limited
        if self._showing_three_layout:
            self._render_three_shot_previews(self._last_three_shot_frames)

    def queue_hardware_stack(self, frames, *, error=False):
        if not frames:
            return
        payload = (tuple(np.asarray(fr) for fr in frames), bool(error))
        if not self._postproc_thread or not self._postproc_thread.is_alive():
            self._process_hardware_stack(payload)
            return
        try:
            self._hardware_stack_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._hardware_stack_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._hardware_stack_queue.put_nowait(payload)
            except queue.Full:
                wx.CallAfter(self.log, "Post-processing queue saturated; dropping stack")

    def _hardware_postproc_loop(self):
        while True:
            item = self._hardware_stack_queue.get()
            if item is None:
                break
            self._process_hardware_stack(item)

    def _prepare_hardware_images(self, frames, error):
        mode = getattr(self, "capture_mode", "3 Shot Absorption")
        if mode == "Single SVD":
            return self._prepare_single_svd_images(frames, error)

        images = [np.asarray(fr, dtype=np.float32) for fr in frames]
        if not error or len(images) != 3:
            return images

        # When a frame is duplicated to fill a missing trigger, synthesize a
        # neutral stack whose absorption image evaluates near zero.  Preserve
        # the overall level of the last captured frame so the saved FITS still
        # reflects the scene brightness.
        baseline = images[-1].astype(np.float32, copy=True)
        synthetic_dark = baseline.copy()
        offset = np.float32(1.0)
        synthetic_probe = baseline.copy()
        synthetic_atom = baseline.copy()
        synthetic_probe += offset
        synthetic_atom += offset
        return [synthetic_atom, synthetic_probe, synthetic_dark]

    def _prepare_single_svd_images(self, frames, error):
        arrays = [np.asarray(fr, dtype=np.float32) for fr in frames]
        if len(arrays) < 2:
            wx.CallAfter(
                self.log,
                "Single SVD: insufficient frames captured; using fallback stack",
            )
            return self._make_single_svd_fallback(arrays)
        atom = arrays[0]
        dark = arrays[1]
        basis = self.svd_basis
        if basis is None or not getattr(basis, "loaded", False):
            wx.CallAfter(self.log, "Single SVD: basis unavailable; using fallback stack")
            return self._make_single_svd_fallback(arrays)
        try:
            background = basis.synthesize_background(atom, dark)
        except SVDBasisError as exc:
            wx.CallAfter(self.log, f"Single SVD synthesis failed: {exc}")
            return self._make_single_svd_fallback(arrays)
        except Exception as exc:  # pragma: no cover - defensive
            wx.CallAfter(self.log, f"Single SVD unexpected error: {exc}")
            return self._make_single_svd_fallback(arrays)

        stats = getattr(basis, "last_synthesis_stats", None)
        if stats:
            alpha = stats.get("alpha")
            resid = stats.get("residual_rms")
            mask_frac = stats.get("mask_fraction")
            try:
                if mask_frac is not None:
                    msg = (
                        f"Single SVD: α={float(alpha):.3g}, RMS={float(resid):.3g}, "
                        f"mask={float(mask_frac):.2%}"
                    )
                else:
                    msg = f"Single SVD: α={float(alpha):.3g}, RMS={float(resid):.3g}"
            except (TypeError, ValueError):
                msg = "Single SVD: background synthesized"
            wx.CallAfter(self.log, msg)

        background = np.asarray(background, dtype=np.float32)
        atom = np.asarray(atom, dtype=np.float32)
        dark = np.asarray(dark, dtype=np.float32)
        return [atom, background, dark]

    def _make_single_svd_fallback(self, arrays):
        if not arrays:
            return []
        atom = np.asarray(arrays[0], dtype=np.float32)
        if len(arrays) > 1:
            dark = np.asarray(arrays[1], dtype=np.float32)
        else:
            dark = atom
        return [atom.copy(), atom.copy(), dark.copy()]

    def _process_hardware_stack(self, payload):
        try:
            frames, error = payload
            images = self._prepare_hardware_images(frames, error)
            path = self.save_fits(images, error=error)
            previews = [img.copy() for img in images]
            trans = createAbsorbImg(images)
            with np.errstate(divide="ignore", invalid="ignore"):
                od = -np.log(np.clip(trans, 1e-6, None))
            od[~np.isfinite(od)] = 0.0
            label = os.path.basename(path) if path else None
        except Exception as exc:
            wx.CallAfter(self.log, f"Post-processing error: {exc}")
            return
        wx.CallAfter(self.update_three_shot_previews, previews)
        wx.CallAfter(self._push_frame, od.astype(np.float32), None, label)

    def update_mode_availability(self):
        mode = self.runModeBox.GetStringSelection()
        cont = mode == "Continuous"
        self.run_mode = mode
        if cont:
            self._last_display_push = 0.0
        self.modeBox.Enable(not cont)
        self.windowCtrl.Enable(cont)
        self.scaleCtrl.Enable(cont)
        self.pauseBtn.Enable(cont)
        self.folderCtrl.Enable(not cont)
        self.browseBtn.Enable(not cont)
        if not cont:
            self.intensity_history.clear()
            self._intensity_time = 0.0
            self._t0 = None
            if self._int_line is not None:
                self._int_line.remove()
                self._int_line = None
            self.canvas.draw_idle()
            self.pauseBtn.SetValue(False)
            self.intensity_paused = False
            if not self.save_folder:
                self.init_save_folder()
        else:
            self._t0 = None
            self.intensity_history.clear()
            self._intensity_time = 0.0
            if self._int_line is not None:
                self._int_line.remove()
                self._int_line = None
            self.canvas.draw_idle()
        self.refresh_preview_visibility()
        self._update_single_svd_ui()

    def _parse_dated_folder(self, folder):
        """Return components of a dated save folder if it matches the expected pattern."""

        if not folder:
            return None
        cleaned = folder.rstrip("\\/")
        if not cleaned:
            return None
        try:
            path = PureWindowsPath(cleaned)
        except TypeError:
            return None
        parts = path.parts
        if len(parts) < 4:
            return None
        try:
            year = int(parts[-4])
            month = int(parts[-3])
            day = int(parts[-2])
        except ValueError:
            return None
        base_parts = parts[:-4]
        if not base_parts:
            return None
        base = PureWindowsPath(*base_parts)
        atom = parts[-1] or DEFAULT_ATOM_SUBFOLDER
        return {
            "base": base,
            "atom": atom,
            "date": date(year, month, day),
            "trailing_sep": folder.endswith(("\\", "/")),
        }

    def _resolve_dated_folder(self):
        """Determine base components for constructing a dated folder."""

        parsed = self._parse_dated_folder(self.save_folder)
        if parsed:
            return parsed
        if not self.save_folder:
            return {
                "base": PureWindowsPath(LOCAL_PATH.rstrip("\\/")),
                "atom": DEFAULT_ATOM_SUBFOLDER,
                "date": None,
                "trailing_sep": True,
            }
        return None

    def _format_dated_folder(self, base, target_date, atom, trailing_sep):
        """Build a dated folder path string from the supplied components."""

        base_path = PureWindowsPath(base)
        path = base_path.joinpath(
            str(target_date.year),
            str(target_date.month),
            str(target_date.day),
            atom or DEFAULT_ATOM_SUBFOLDER,
        )
        folder = str(path)
        if trailing_sep:
            folder += "\\"
        return folder

    def _ensure_save_folder_for_today(self):
        """Ensure ``self.save_folder`` points to today's dated directory."""

        today = date.today()
        components = self._resolve_dated_folder()
        if components is None:
            self.today = today
            return

        previous = self.save_folder
        folder = self._format_dated_folder(
            components["base"], today, components["atom"], components["trailing_sep"]
        )
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except Exception:
                fallback_folder = self._format_dated_folder(
                    NETWORK_FALLBACK_BASE,
                    today,
                    components["atom"],
                    components["trailing_sep"],
                )
                folder = fallback_folder
                if not os.path.exists(folder):
                    try:
                        os.makedirs(folder)
                    except Exception:
                        pass

        self.today = today
        self.save_folder = folder
        if hasattr(self, "folderCtrl"):
            def sync_folder(path=folder, prev=previous):
                if self.folderCtrl.GetValue() != path or prev != path:
                    self.folderCtrl.SetValue(path)

            wx.CallAfter(sync_folder)

    def init_save_folder(self):
        """Determine a default folder for saving triggered image stacks."""
        self._ensure_save_folder_for_today()

    def on_folder_change(self, event):
        self.save_folder = self.folderCtrl.GetValue()
        parsed = self._parse_dated_folder(self.save_folder)
        if parsed:
            self.today = parsed["date"]
        else:
            self.today = date.today()

    def update_save_folder(self):
        """Update the save folder if a new day has started."""
        current = date.today()
        if current == getattr(self, "today", current):
            return
        self._ensure_save_folder_for_today()

    def on_popout(self, event):
        if self.popoutCheck.GetValue():
            if self.popout_frame is None:
                self.popout_frame = PopoutFrame(self)
                self.popout_frame.Show()
                self.popout_frame.set_canvas_size(self._img_width, self._img_height)
                if self.last_raw_image is not None:
                    self._render_bitmap(self.last_raw_image)
        else:
            if self.popout_frame is not None:
                self.popout_frame.Destroy()
                self.popout_frame = None
                if self.last_raw_image is not None:
                    self._render_bitmap(self.last_raw_image)

    def on_pause_graph(self, event):
        self.intensity_paused = self.pauseBtn.GetValue()

    def on_flip_change(self, event):
        if self.cam_info:
            self.save_current_settings()
        if self.last_raw_image is not None:
            self._render_bitmap(self.last_raw_image)
        if self._last_three_shot_frames:
            self.update_three_shot_previews([np.asarray(fr) for fr in self._last_three_shot_frames])

    def on_colormap_changed(self, event):
        cmap_name = None
        if getattr(self, "colormapChoice", None):
            idx = self.colormapChoice.GetSelection()
            if idx != wx.NOT_FOUND and getattr(self, "_colormap_options", None):
                try:
                    cmap_name = self._colormap_options[idx][1]
                except IndexError:
                    cmap_name = None
        changed = self._apply_preview_colormap(cmap_name)
        if self.cam_info:
            self.save_current_settings()
        if self.last_raw_image is not None and (changed or event is not None):
            self._render_bitmap(self.last_raw_image)
        if event is not None:
            event.Skip()

    def set_canvas_size(self, width, height):
        width = int(width)
        height = int(height)
        self._img_width = width
        self._img_height = height
        display_w = max(1, int(width * self._display_scale))
        display_h = max(1, int(height * self._display_scale))
        self._display_width = display_w
        self._display_height = display_h
        self.image_canvas.set_display_scale(self._display_scale)
        self.image_canvas.set_display_size(display_w, display_h)
        self.image_canvas.set_axis_dimensions(width, height)
        dpi = self.canvas.figure.get_dpi()
        ratio = self._three_shot_height_ratio if self._showing_three_layout else self._intensity_height_ratio
        target_height = int(round(height * ratio))
        canvas_height = max(self._base_canvas_height, target_height)
        min_width = max(display_w, 512)
        tiles = max(len(self.preview_titles), 1)
        min_square = int(math.ceil(min_width / tiles))
        # Keep the normalized intensity plot the same vertical scale that the
        # three-shot hardware thumbnails require so both layouts occupy
        # comparable vertical space.
        canvas_height = max(canvas_height, min_square)
        self.canvas.SetMinSize((min_width, canvas_height))
        self.canvas.SetSize((-1, canvas_height))
        fig = self.canvas.figure
        fig.set_size_inches(min_width / dpi, canvas_height / dpi)
        if self.intensity_ax is not None:
            self.intensity_ax.set_xlim(-self.intensity_window, 0)
            self.intensity_ax.set_ylim(0, self.intensity_scale)
        self.canvas.draw_idle()
        self.toolbar.update()
        if self.popout_frame is not None:
            self.popout_frame.set_canvas_size(width, height)
        if self.last_raw_image is not None:
            self._render_bitmap(self.last_raw_image)
        self.Layout()
        self.SetMinSize(self.GetSizer().CalcMin())
        parent = self.GetParent()
        if parent:
            parent.Layout()
            parent.SetMinSize(parent.GetSizer().CalcMin())
        top = self.GetTopLevelParent()
        if top:
            top.Layout()
            top.SendSizeEvent()

    def _apply_preview_colormap(self, cmap_name):
        if not getattr(self, "_colormap_options", None):
            self._colormap_options = PREVIEW_COLORMAP_OPTIONS
        cmap_name = (cmap_name or "").strip().lower()
        fallback_idx = next(
            (idx for idx, (_, cmap) in enumerate(self._colormap_options) if cmap == "gray_r"),
            0,
        )
        idx = fallback_idx
        for opt_idx, (_, cmap) in enumerate(self._colormap_options):
            if cmap.lower() == cmap_name:
                idx = opt_idx
                break
        selected = self._colormap_options[idx][1]
        changed = selected != getattr(self, "_selected_colormap_name", None)
        self._selected_colormap_name = selected
        if getattr(self, "colormapChoice", None) and self.colormapChoice.GetSelection() != idx:
            self.colormapChoice.SetSelection(idx)
        if changed:
            self._rebuild_colormap_lut()
        return changed

    def _get_selected_colormap(self):
        return getattr(self, "_selected_colormap_name", "gray_r")

    def apply_saved_settings(self, serial):
        cfg = load_settings_file().get(serial)
        if not cfg:
            return
        self.runModeBox.SetStringSelection(cfg.get("run_mode", "Continuous"))
        self.update_mode_availability()
        self.modeBox.SetStringSelection(cfg.get("mode", "3 Shot Absorption"))
        self.capture_mode = self.modeBox.GetStringSelection()
        self.exposureCtrl.SetValue(cfg.get("exposure_us", 1000))
        self.gainCtrl.SetValue(cfg.get("gain", 0))
        self.blackCtrl.SetValue(cfg.get("black_level", 0))
        bin_sel = f"{cfg.get('bin_x', 1)}x{cfg.get('bin_y', 1)}"
        if bin_sel in self.binChoice.GetItems():
            self.binChoice.SetStringSelection(bin_sel)
        self.windowCtrl.SetValue(cfg.get("intensity_window", self.intensity_window))
        self.intensity_window = self.windowCtrl.GetValue()
        self.scaleCtrl.SetValue(str(cfg.get("intensity_scale", self.intensity_scale)))
        try:
            self.intensity_scale = float(self.scaleCtrl.GetValue())
        except ValueError:
            self.intensity_scale = 1.0
        self.flipHCheck.SetValue(cfg.get("flip_horizontal", False))
        self.flipVCheck.SetValue(cfg.get("flip_vertical", True))
        cmap_name = cfg.get("preview_colormap")
        if not isinstance(cmap_name, str):
            use_jet = cfg.get("jet_preview")
            if isinstance(use_jet, bool):
                cmap_name = "jet" if use_jet else "gray_r"
        changed = self._apply_preview_colormap(cmap_name)
        if (changed or not isinstance(cmap_name, str)) and self.last_raw_image is not None:
            self._render_bitmap(self.last_raw_image)
        if self.intensity_ax is not None:
            self.intensity_ax.set_xlim(-self.intensity_window, 0)
            self.intensity_ax.set_ylim(0, self.intensity_scale)
            self.canvas.draw_idle()
        folder = cfg.get("save_folder")
        if folder:
            self.save_folder = folder
            self.folderCtrl.SetValue(folder)
        else:
            self.save_folder = ""
            self.folderCtrl.SetValue(self.save_folder)
        self._ensure_save_folder_for_today()
        svd_refs = cfg.get("svd_n_refs")
        if isinstance(svd_refs, int):
            self.svd_default_refs = max(5, min(svd_refs, 10_000))
        svd_k = cfg.get("svd_k")
        if isinstance(svd_k, int):
            self.svd_default_k = max(1, min(svd_k, 500))
        svd_folder = cfg.get("svd_folder")
        if isinstance(svd_folder, str):
            self.svd_reference_folder = svd_folder
        else:
            self.svd_reference_folder = ""
        self.svd_basis = None
        self.svd_basis_path = ""
        self.svdBasisCtrl.SetValue("")
        self._set_svd_status("Not loaded")
        self._svd_previous_status = "Not loaded"
        self._update_single_svd_ui()
        self._update_start_button_state()

    def load_last_serial(self):
        return load_settings_file().get(f"panel{self.index}_serial")

    def save_last_serial(self, serial):
        data = load_settings_file()
        data[f"panel{self.index}_serial"] = serial
        save_settings_file(data)

    def save_current_settings(self):
        if self.cam_info is None:
            return
        data = load_settings_file()
        cfg = data.get(self.cam_info, {})
        # Remove any old dimension entries so image size isn't persisted
        cfg.pop("width", None)
        cfg.pop("height", None)
        cfg.update(
            {
                "run_mode": self.run_mode,
                "mode": self.modeBox.GetStringSelection(),
                "exposure_us": self.exposure_us,
                "gain": self.gain,
                "black_level": self.black_level,
                "bin_x": self.bin_x,
                "bin_y": self.bin_y,
                "intensity_window": self.intensity_window,
                "intensity_scale": self.intensity_scale,
                "flip_horizontal": self.flipHCheck.GetValue(),
                "flip_vertical": self.flipVCheck.GetValue(),
                "preview_colormap": self._get_selected_colormap(),
                "jet_preview": self._get_selected_colormap() == "jet",
                "save_folder": self.save_folder,
                "svd_folder": self.svd_reference_folder,
                "svd_n_refs": self.svd_default_refs,
                "svd_k": self.svd_default_k,
            }
        )
        data[self.cam_info] = cfg
        data[f"panel{self.index}_serial"] = self.cam_info
        save_settings_file(data)

    def start(self):
        if self.running:
            return
        idx = self.camChoice.GetSelection()
        if idx == wx.NOT_FOUND:
            self.log("Select a camera first")
            self.toggleBtn.SetValue(False)
            return
        selected_mode = self.modeBox.GetStringSelection()
        selected_run_mode = self.runModeBox.GetStringSelection()
        if (
            selected_run_mode == "Hardware Trigger"
            and selected_mode == "Single SVD"
            and not self._is_svd_basis_ready()
        ):
            wx.MessageBox(
                "Load an SVD basis before starting Single SVD capture.",
                "Single SVD",
                wx.OK | wx.ICON_WARNING,
            )
            self.log("Single SVD capture blocked: load SVD basis first")
            self.toggleBtn.SetValue(False)
            return
        self.cam_info = self.available_cams[idx]
        self.last_serial = self.cam_info
        self.camChoice.Enable(False)
        self.refreshBtn.Enable(False)
        self.modeBox.Enable(False)
        self.runModeBox.Enable(False)
        self.exposureCtrl.Enable(False)
        self.gainCtrl.Enable(False)
        self.blackCtrl.Enable(False)
        self.binChoice.Enable(False)
        self.run_mode = selected_run_mode
        self.capture_mode = selected_mode
        self.exposure_us = self.exposureCtrl.GetValue()
        self.gain = self.gainCtrl.GetValue()
        self.black_level = self.blackCtrl.GetValue()
        try:
            self.bin_x, self.bin_y = map(int, self.binChoice.GetStringSelection().split("x"))
        except Exception:
            self.bin_x = self.bin_y = 1
        self.running = True
        self.worker = CameraWorker(self)
        self.worker.start()
        self.toggleBtn.SetLabel("Disconnect")
        self.log(f"Connecting to {self.cam_info}")
        self.save_current_settings()

    def stop(self):
        """Stop acquisition and ensure the camera disconnects."""
        self.running = False
        if self.worker:
            # Wait for the worker thread to fully exit so the camera is
            # disarmed and disposed before closing the application.
            self.worker.join()
            self.worker = None
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._hardware_stack_queue.get_nowait()
            except queue.Empty:
                break
        self.clear_three_shot_previews(drop_cache=True)
        # If, for any reason, the worker failed to release the camera, do so
        # here to avoid leaving it connected when the window closes.
        if self.camera is not None:
            try:
                self.camera.disarm()
                self.camera.dispose()
            except Exception:
                pass
            self.camera = None
        # Reset UI elements in case the frame is still visible while shutting
        # down or when the user explicitly disconnects.
        self.toggleBtn.SetValue(False)
        self.toggleBtn.SetLabel("Connect")
        self.camChoice.Enable(True)
        self.refreshBtn.Enable(True)
        self.runModeBox.Enable(True)
        self.update_mode_availability()
        self.exposureCtrl.Enable(True)
        self.gainCtrl.Enable(True)
        self.blackCtrl.Enable(True)
        self.binChoice.Enable(True)
        if self.cam_info:
            self.log(f"Disconnected {self.cam_info}")

    def shutdown(self):
        if self.running:
            self.stop()
        if self.popout_frame is not None:
            self.popout_frame.Destroy()
            self.popout_frame = None
        if self._paint_timer.IsRunning():
            self._paint_timer.Stop()
        if self._postproc_thread and self._postproc_thread.is_alive():
            self._hardware_stack_queue.put(None)
            self._postproc_thread.join()
            self._postproc_thread = None

    # ------------------------------------------------------------------
    def _set_raw_bit_depth(self, bit_depth):
        try:
            bits = int(bit_depth)
        except (TypeError, ValueError):
            return
        bits = max(8, min(16, bits))
        if bits == self._raw_bit_depth:
            return
        self._raw_bit_depth = bits
        self._raw_max_value = float((1 << bits) - 1)
        self._raw_scale = 255.0 / self._raw_max_value
        self._rebuild_colormap_lut()

    def _rebuild_colormap_lut(self):
        levels = 1 << self._raw_bit_depth
        getters = []
        registry = getattr(matplotlib, "colormaps", None)
        if registry is not None:
            method = getattr(registry, "get_cmap", None)
            if callable(method):
                getters.append(method)

        cm_getter = getattr(cm, "get_cmap", None)
        if callable(cm_getter):
            getters.append(cm_getter)

        def request_colormap(name):
            for getter in getters:
                value_error = False
                try:
                    cmap = getter(name, lut=levels)
                except TypeError:
                    pass
                except ValueError:
                    value_error = True
                else:
                    if cmap is not None:
                        return cmap

                if value_error:
                    continue

                try:
                    cmap = getter(name, levels)
                except TypeError:
                    pass
                except ValueError:
                    continue
                else:
                    if cmap is not None:
                        return cmap

            return None

        cmap = request_colormap(self._selected_colormap_name)
        if cmap is None:
            cmap = request_colormap("gray_r")

        if cmap is not None:
            lut = cmap(np.linspace(0, 1, levels))[:, :3]
        else:
            ramp = np.linspace(0, 1, levels, dtype=np.float32)
            lut = np.repeat(ramp[:, None], 3, axis=1)
            if hasattr(self, "logCtrl"):
                self.log("Falling back to grayscale colormap; Matplotlib colormap lookup failed.")

        self._colormap_lut = (lut * 255).astype(np.uint8)
        lo, hi = self._od_range
        denom = max(hi - lo, 1e-6)
        self._od_scale = (levels - 1) / denom

    def _raw_to_u8(self, arr):
        return np.clip(arr.astype(np.float32) * self._raw_scale, 0, 255).astype(np.uint8)

    def _raw_to_colormap_indices(self, arr):
        levels = len(self._colormap_lut)
        if levels <= 1:
            return np.zeros_like(arr, dtype=np.int32)
        clipped = np.clip(arr.astype(np.int64), 0, levels - 1)
        return clipped.astype(np.int32)

    def _od_to_lut_indices(self, arr):
        lo, hi = self._od_range
        levels = len(self._colormap_lut)
        if levels <= 1:
            return np.zeros_like(arr, dtype=np.int32)
        cleaned = np.nan_to_num(arr.astype(np.float32), nan=lo, posinf=hi, neginf=lo)
        clipped = np.clip(cleaned, lo, hi)
        scaled = (clipped - lo) * self._od_scale
        return np.clip(np.rint(scaled), 0, levels - 1).astype(np.int32)

    def _apply_flips(self, img):
        if self.flipHCheck.GetValue():
            img = np.fliplr(img)
        if self.flipVCheck.GetValue():
            img = np.flipud(img)
        return img

    def _gray_to_rgb(self, g8):
        if g8.ndim == 2:
            rgb = np.empty((g8.shape[0], g8.shape[1], 3), dtype=np.uint8)
            rgb[..., 0] = g8
            rgb[..., 1] = g8
            rgb[..., 2] = g8
            return rgb
        return g8

    def _render_bitmap(self, frame, overlay_text=None):
        if frame is None:
            return
        if overlay_text is None:
            overlay_text = self._current_overlay_text or None
        img = self._apply_flips(frame)
        axis_h, axis_w = img.shape[:2]
        step = 1
        if self._display_scale < 1.0:
            inv = int(round(1.0 / self._display_scale))
            step = max(1, inv)
        if step > 1:
            img = img[::step, ::step]
        cmap_name = self._get_selected_colormap()
        if frame.dtype == np.uint16:
            use_color = (
                cmap_name != "gray_r"
                and self.runModeBox.GetStringSelection() == "Continuous"
            )
            if use_color:
                indices = self._raw_to_colormap_indices(img)
                rgb = self._colormap_lut[indices]
            else:
                gray8 = self._raw_to_u8(img)
                if cmap_name == "gray_r":
                    rgb = self._gray_to_rgb(gray8)
                else:
                    # Fall back to grayscale when color LUTs are disabled in this mode.
                    rgb = self._gray_to_rgb(gray8)
        else:
            indices = self._od_to_lut_indices(img)
            rgb = self._colormap_lut[indices]
        h, w = rgb.shape[:2]
        data = rgb.tobytes()
        bmp = wx.Bitmap.FromBuffer(w, h, data)
        if overlay_text:
            dc = wx.MemoryDC(bmp)
            dc.SetFont(self._overlay_font)
            dc.SetTextForeground(wx.Colour(255, 255, 255))
            padding = 4
            tw, th = dc.GetTextExtent(overlay_text)
            box_w = tw + padding * 2
            box_h = th + padding * 2
            x = max(0, w - box_w - padding)
            y = padding
            dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0)))
            dc.SetPen(wx.Pen(wx.Colour(0, 0, 0)))
            dc.DrawRectangle(x, y, box_w, box_h)
            dc.DrawText(overlay_text, x + padding, y + padding)
            dc.SelectObject(wx.NullBitmap)
        if self.popout_frame is not None:
            target = self.popout_frame.image_canvas
        else:
            target = self.image_canvas
        target.update_bitmap(bmp, data, axis_size=(axis_w, axis_h))

    def _push_frame(self, frame, fps, label):
        # Do not throttle acquisition; only throttle display pushes if desired.
        if self.run_mode == "Continuous" and self.display_fps_cap:
            now = time.monotonic()
            if now - self._last_display_push < 1.0 / self.display_fps_cap:
                return
            self._last_display_push = now
        payload = (frame, fps, label)
        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(payload)
            except queue.Full:
                pass

    def _on_paint_tick(self, _evt):
        payload = None
        while True:
            try:
                payload = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if payload is None:
            return
        frame, fps, label = payload
        self._latest_frame = frame
        self._latest_fps = fps
        self._latest_label = label
        self.last_raw_image = frame

        text = ""
        if fps is not None:
            text = f"{fps:.1f} FPS"
        elif label:
            text = label
        self._current_overlay_text = text

        update_intensity = (
            self.run_mode == "Continuous" and not self.intensity_paused and self.intensity_ax is not None
        )
        if update_intensity and frame.dtype == np.uint16:
            samp = frame[::2, ::2]
            total = float(samp.sum()) / (max(samp.size, 1) * self._raw_max_value)
            now = time.monotonic()
            if self._t0 is None:
                self._t0 = now
            t = now - self._t0
            self._intensity_time = t
            self.intensity_history.append((t, total))
            while self.intensity_history and self.intensity_history[0][0] < t - self.intensity_window:
                self.intensity_history.popleft()
            xs = [ti - t for ti, _ in self.intensity_history]
            ys = [vi for _, vi in self.intensity_history]
            if self._int_line is None:
                (self._int_line,) = self.intensity_ax.plot(xs, ys, color="tab:blue")
            else:
                self._int_line.set_data(xs, ys)
            self.intensity_ax.set_xlim(-self.intensity_window, 0)
            self.canvas.draw_idle()

        if self.run_mode != "Continuous" or frame.dtype != np.uint16:
            self._t0 = None

        self._render_bitmap(frame, text if text else None)

    def save_fits(self, images, *, error=False):
        """Save a three-frame stack as a FITS file.

        If ``error`` is true, the filename is suffixed with ``-error`` to
        indicate a duplicated frame in the stack.
        """
        try:
            self.update_save_folder()
            folder = self.save_folder
            os.makedirs(folder, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "-error" if error else ""
            path = os.path.join(folder, f"{ts}{suffix}.fits")
            if self.run_mode == "Hardware Trigger" and self.capture_mode == "Single SVD":
                if len(images) < 3:
                    raise ValueError("Single SVD capture did not produce three frames")

                def to_float32(arr):
                    data = np.asarray(arr, dtype=np.float64)
                    data = np.clip(data, 0.0, 65535.0)
                    return data.astype(np.float32)

                stack = np.stack(
                    [to_float32(images[0]), to_float32(images[1]), to_float32(images[2])],
                    axis=0,
                )
                primary = fits.PrimaryHDU(stack)
                hdr = primary.header
                hdr["IMGMODE"] = ("SingleSVD", "Capture mode")
                hdr["DATE-OBS"] = datetime.now().isoformat(timespec="seconds")
                hdr["EXPOSURE"] = (int(self.exposure_us), "Exposure time (us)")
                hdr["GAIN"] = (int(self.gain), "Camera gain")
                hdr["BINNING"] = (f"{self.bin_x}x{self.bin_y}", "Camera binning")
                basis = self.svd_basis if self._is_svd_basis_ready() else None
                if basis is not None:
                    hdr["SVDK"] = (int(basis.k), "Number of SVD modes")
                    hdr["SVDNREF"] = (int(basis.n_refs), "Reference FITS used")
                    hdr["SVDNORM"] = ("median_ratio_to_mean", "Scalar normalization method")
                    if basis.reference_folder:
                        hdr["REFPATH"] = (basis.reference_folder, "SVD reference folder")
                    if getattr(basis, "basis_path", ""):
                        hdr["SVDBFILE"] = (basis.basis_path, "Serialized SVD basis file")
                    stats = getattr(basis, "last_synthesis_stats", None)
                    if stats:
                        alpha = stats.get("alpha")
                        resid = stats.get("residual_rms")
                        mask_frac = stats.get("mask_fraction")
                        if alpha is not None:
                            hdr["SVDALPHA"] = (float(alpha), "Median normalization factor")
                        if resid is not None:
                            hdr["SVDRMS"] = (float(resid), "LS residual RMS")
                        if mask_frac is not None:
                            hdr["SVDMASK"] = (
                                float(mask_frac),
                                "Fraction of pixels masked from LS fit",
                            )
                primary.writeto(path, overwrite=True)
            else:
                stacked = np.stack(images, axis=0).astype(np.float32)
                fits.PrimaryHDU(stacked).writeto(path, overwrite=True)
            self.log(f"FITS saved: {path}")
            return path
        except Exception as exc:
            self.log(f"Failed to save FITS: {exc}")
            return None


class PopoutFrame(wx.Frame):
    """Separate window showing a camera image."""

    def __init__(self, panel):
        super().__init__(panel, title=f"Camera {panel.index} Image")
        self.panel = panel
        w, h = panel._display_width, panel._display_height
        self.image_canvas = ImageCanvas(self, panel._display_scale)
        self.image_canvas.set_display_size(w, h)
        self.image_canvas.set_axis_dimensions(panel._img_width, panel._img_height)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.image_canvas, 1, wx.ALL | wx.EXPAND, 5)
        self.SetSizerAndFit(sizer)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def set_canvas_size(self, width, height):
        display_w = max(1, int(width * self.panel._display_scale))
        display_h = max(1, int(height * self.panel._display_scale))
        self.image_canvas.set_display_scale(self.panel._display_scale)
        self.image_canvas.set_display_size(display_w, display_h)
        self.image_canvas.set_axis_dimensions(width, height)
        self.Layout()
        self.Fit()

    def on_close(self, event):
        self.panel.popout_frame = None
        self.panel.popoutCheck.SetValue(False)
        event.Skip()


class CameraWorker(threading.Thread):
    """Background thread that handles image acquisition for a camera."""

    def __init__(self, panel):
        super().__init__(daemon=True)
        self.panel = panel

    def run(self):
        p = self.panel
        try:
            with p.sdk.open_camera(p.cam_info) as cam:
                p.camera = cam
                wx.CallAfter(p.set_canvas_size, cam.image_width_pixels, cam.image_height_pixels)
                # Apply user-selected settings once before starting capture.
                cam.exposure_time_us = p.exposure_us
                if hasattr(cam, "gain"):
                    cam.gain = p.gain
                if hasattr(cam, "black_level"):
                    cam.black_level = p.black_level
                if hasattr(cam, "bin_x") and hasattr(cam, "bin_y"):
                    cam.bin_x = p.bin_x
                    cam.bin_y = p.bin_y
                bit_depth = getattr(cam, "bit_depth", None)
                if bit_depth is None:
                    bit_depth = getattr(cam, "bits_per_pixel", None)
                if bit_depth:
                    wx.CallAfter(p._set_raw_bit_depth, bit_depth)

                bit_depth_report = getattr(cam, "bit_depth", getattr(cam, "bits_per_pixel", "?"))
                wx.CallAfter(
                    p.log,
                    (
                        f"Connected {p.cam_info} | {cam.image_width_pixels}x{cam.image_height_pixels} "
                        f"| bit_depth={bit_depth_report}"
                    ),
                )

                if p.run_mode == "Continuous":
                    cam.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
                    cam.frames_per_trigger_zero_for_unlimited = 1

                    # One frame per issued software trigger
                    arm_depth = 16

                    # Adaptive timeout: at least 0.75 s, else ~10x exposure
                    timeout_s = max(0.75, 10.0 * p.exposure_us / 1_000_000.0)

                    def prime(depth):
                        issued = 0
                        for _ in range(depth):
                            if not p.running:
                                break
                            try:
                                cam.issue_software_trigger()
                            except Exception as exc:
                                wx.CallAfter(p.log, f"Continuous: trigger error {exc}")
                                break
                            else:
                                issued += 1
                        return issued

                    cam.arm(arm_depth)
                    outstanding = prime(arm_depth)
                    last_frame_ts = time.monotonic()
                    last = None

                    # Non-blocking acquisition loop
                    while p.running:
                        frame = cam.get_pending_frame_or_null()
                        if frame is None:
                            if not p.running:
                                break
                            # Re-arm if we haven't seen a frame for a while (USB hiccup, UI stall, etc.)
                            if time.monotonic() - last_frame_ts > timeout_s:
                                try:
                                    cam.disarm()
                                except Exception:
                                    pass
                                cam.arm(arm_depth)
                                outstanding = prime(arm_depth)
                                last = None
                                last_frame_ts = time.monotonic()
                                wx.CallAfter(p.log, "Continuous: re-armed after timeout")
                            else:
                                time.sleep(0.0005)
                            continue

                        now = time.monotonic()
                        last_frame_ts = now
                        if outstanding > 0:
                            outstanding -= 1

                        fps = None
                        if last is not None:
                            dt = now - last
                            fps = 1.0 / dt if dt > 0 else None
                        last = now

                        img = np.copy(frame.image_buffer).reshape(
                            cam.image_height_pixels, cam.image_width_pixels
                        )

                        # Push image to UI (UI may be throttled separately)
                        p._push_frame(img, fps, None)

                        # Keep the pipeline primed
                        if p.running and outstanding < arm_depth:
                            try:
                                cam.issue_software_trigger()
                            except Exception as exc:
                                wx.CallAfter(p.log, f"Continuous: trigger error {exc}")
                            else:
                                outstanding += 1

                    try:
                        cam.disarm()
                    except Exception:
                        pass
                else:
                    cam.operation_mode = OPERATION_MODE.HARDWARE_TRIGGERED
                    cam.trigger_polarity = TRIGGER_POLARITY.ACTIVE_HIGH
                    try:
                        cam.frames_per_trigger_zero_for_unlimited = 1  # critical: 1 frame per edge
                    except Exception:
                        pass

                    # Give ourselves a generous queue to avoid overflows if UI stalls briefly
                    cam.arm(16)

                    # Three external triggers (Atom, Probe, Dark), one frame per edge:
                    # - frames_per_trigger_zero_for_unlimited = 1
                    # - arm depth >= 16 to survive brief UI stalls
                    # - Before each 3-shot: flush stale frames
                    # - Per-frame and per-stack timeouts to resync if a trigger is missed
                    # - Acquisition thread never blocks on plotting or disk I/O

                    def flush_pending(max_frames=64):
                        drained = 0
                        while drained < max_frames:
                            f = cam.get_pending_frame_or_null()
                            if f is None:
                                break
                            drained += 1
                            dispose = getattr(f, "dispose", None)
                            if callable(dispose):
                                try:
                                    dispose()
                                except Exception:
                                    pass
                        if drained:
                            wx.CallAfter(p.log, f"Hardware: flushed {drained} stale frame(s)")

                    # Optional: safety timeout per frame (missed trigger) and per 3-shot
                    per_frame_timeout_s = 0.5
                    per_stack_timeout_s = 2.0

                    need_flush = True

                    frames_per_stack = 3 if p.capture_mode != "Single SVD" else 2
                    success_label = (
                        "Hardware: 3-shot captured (A/P/D)"
                        if frames_per_stack == 3
                        else "Hardware: single-SVD captured (Atom/Dark)"
                    )

                    while p.running:
                        if need_flush:
                            flush_pending()
                            need_flush = False

                        images = []

                        # Wait indefinitely for the first trigger in the stack so we stay armed.
                        first_frame = None
                        while first_frame is None and p.running:
                            first_frame = cam.get_pending_frame_or_null()
                            if first_frame is None:
                                time.sleep(0.0005)

                        if not p.running or first_frame is None:
                            break

                        t_stack_start = time.monotonic()

                        img = np.copy(first_frame.image_buffer).reshape(
                            cam.image_height_pixels, cam.image_width_pixels
                        )
                        images.append(img)
                        dispose = getattr(first_frame, "dispose", None)
                        if callable(dispose):
                            try:
                                dispose()
                            except Exception:
                                pass

                        stack_error = False

                        for i in range(1, frames_per_stack):
                            frame = None
                            t0 = time.monotonic()
                            # Only enforce timeouts after the first frame has arrived.
                            while frame is None and p.running:
                                frame = cam.get_pending_frame_or_null()
                                if frame is None:
                                    now = time.monotonic()
                                    if now - t0 > per_frame_timeout_s:
                                        wx.CallAfter(
                                            p.log,
                                            (
                                                f"Hardware: frame #{i + 1} timed out; "
                                                "duplicating last frame"
                                            ),
                                        )
                                        stack_error = True
                                        need_flush = True
                                        break
                                    if now - t_stack_start > per_stack_timeout_s:
                                        wx.CallAfter(
                                            p.log,
                                            (
                                                "Hardware: 3-shot timeout; duplicating last "
                                                "frame(s)"
                                            ),
                                        )
                                        stack_error = True
                                        need_flush = True
                                        break
                                    time.sleep(0.0005)
                            if not p.running:
                                break
                            if frame is None:
                                break

                            img = np.copy(frame.image_buffer).reshape(
                                cam.image_height_pixels, cam.image_width_pixels
                            )
                            images.append(img)
                            dispose = getattr(frame, "dispose", None)
                            if callable(dispose):
                                try:
                                    dispose()
                                except Exception:
                                    pass

                        if not p.running:
                            break

                        if images:
                            while len(images) < frames_per_stack:
                                stack_error = True
                                images.append(images[-1].copy())

                            try:
                                p.queue_hardware_stack(images, error=stack_error)
                                if stack_error:
                                    wx.CallAfter(
                                        p.log,
                                        (
                                            "Hardware: 3-shot captured with duplicated frame(s)"
                                            if frames_per_stack == 3
                                            else "Hardware: single-SVD captured with duplicated frame(s)"
                                        ),
                                    )
                                else:
                                    wx.CallAfter(p.log, success_label)
                            except Exception as exc:
                                wx.CallAfter(p.log, f"Queue error: {exc}")

                    try:
                        cam.disarm()
                    except Exception:
                        pass
        except Exception as exc:  # pragma: no cover - requires hardware
            wx.CallAfter(p.log, f"Camera error: {exc}")
        finally:
            # The context manager automatically disposes the camera; just
            # clear the reference and reset the UI state.
            p.camera = None
            p.running = False
            wx.CallAfter(p.toggleBtn.SetValue, False)
            wx.CallAfter(p.toggleBtn.SetLabel, "Connect")
            wx.CallAfter(p.camChoice.Enable, True)
            wx.CallAfter(p.refreshBtn.Enable, True)
            wx.CallAfter(p.runModeBox.Enable, True)
            wx.CallAfter(p.update_mode_availability)
            wx.CallAfter(p.exposureCtrl.Enable, True)
            wx.CallAfter(p.gainCtrl.Enable, True)
            wx.CallAfter(p.blackCtrl.Enable, True)
            wx.CallAfter(p.binChoice.Enable, True)
            if p.cam_info:
                wx.CallAfter(p.log, f"Disconnected {p.cam_info}")

