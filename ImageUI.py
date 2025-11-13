'''Main GUI for analyzing atomic images.'''

import contextlib
import datetime
import gc
import glob
import json
import os
import shutil
import sys
import time

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import gridspec, rc
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import matplotlib
from matplotlib.figure import Figure
from PIL import Image
from astropy.io import fits
from scipy import linalg as LA
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.signal import medfilt, find_peaks
from scipy import stats
from sklearn.decomposition import PCA
import wx
import wx.lib.scrolledpanel
import csv

from fit_functions import FIT_FUNCTIONS

from canvasFrame import *
from canvasPanel import *
from degenerateFitter import *
from exp_params import *
from figurePanel import *
from fitTool import *
from imagePlot import *
from imgFunc_v7 import *
from localPath import *
from Monitor import *
from watchforchange import *
from constant_v6 import kB, hbar, massUnit
from fitting_window import FittingWindow
from average_preview import AvgPreviewFrame

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "imageui_settings.json")

ABSORPTION_COLORMAP_OPTIONS = [
    ("Grey Scale", "gray_r"),
    ("Jet", "jet"),
]


def _safe_get_cmap(name, fallback="gray_r", lut=None):
    """Return a Matplotlib colormap using the modern API when available."""

    def _with_lut(cmap_obj):
        if lut is None or cmap_obj is None:
            return cmap_obj
        try:
            return cmap_obj.resampled(lut)
        except AttributeError:
            # Older Matplotlib may not support ``resampled`` – fall back to cm.get_cmap.
            return cm.get_cmap(getattr(cmap_obj, "name", fallback), lut=lut)

    registry = getattr(matplotlib, "colormaps", None)
    target = (name or "").strip() or fallback
    if registry is not None:
        try:
            cmap = registry.get_cmap(target)
        except ValueError:
            cmap = None
        if cmap is None:
            try:
                cmap = registry.get_cmap(fallback)
            except Exception:
                cmap = None
        cmap = _with_lut(cmap)
        if cmap is not None:
            return cmap

    # Fallback for very old Matplotlib versions that lack ``matplotlib.colormaps``
    try:
        return cm.get_cmap(target, lut=lut)
    except Exception:
        return cm.get_cmap(fallback, lut=lut)

class RedirectText(object):
    """File-like object to redirect stdout/stderr to a wx.TextCtrl."""

    def __init__(self, text_ctrl):
        self.out = text_ctrl

    def write(self, string):
        wx.CallAfter(self.out.AppendText, string)

    def flush(self):
        pass


class AtomNumberDisplayFrame(wx.Frame):
    """Small window to show atom number in large font."""

    def __init__(self, parent):
        super().__init__(parent, title="Atom Number", size=(200, 150))
        self.parent = parent
        self.text = wx.StaticText(self, label="", style=wx.ALIGN_CENTER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.AddStretchSpacer()
        sizer.Add(self.text, 0, wx.ALIGN_CENTER)
        sizer.AddStretchSpacer()
        self.SetSizer(sizer)
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def on_resize(self, event):
        w, h = self.GetClientSize()
        ptsize = max(min(w, h) // 4, 5)
        font = self.text.GetFont()
        font.SetPointSize(ptsize)
        self.text.SetFont(font)
        if event:
            event.Skip()

    def set_number(self, value):
        self.text.SetLabel(str(value))
        self.on_resize(None)

    def on_close(self, event):
        if getattr(self.parent, "showAtomNumberCheck", None):
            self.parent.showAtomNumberCheck.SetValue(False)
        self.parent.atomNumberFrame = None
        event.Skip()


class PlotsFrame(wx.Frame):
    """Window that plots various fit values over time."""

    def __init__(self, parent, show_avg_default=False):
        super().__init__(parent, title="Trends Plot")
        self.parent = parent

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()

        choices = [
            "True_X_Width",
            "True_Y_Width",
            "X_Center",
            "Y_Center",
            "Atom_Number",
            "X_Atom_Number",
            "Y_Atom_Number",
        ]
        self.choiceLabel = wx.StaticText(self, label="Value:")
        self.varChoice = wx.Choice(self, choices=choices)
        self.varChoice.SetStringSelection("Atom_Number")

        self.showAvgCheck = wx.CheckBox(self, label="Show Avg")
        self.showAvgCheck.SetValue(show_avg_default)
        self.avgLabel = wx.StaticText(self, label="Avg len:")
        self.avgCtrl = wx.TextCtrl(
            self, value="10", size=(50, 22), style=wx.TE_PROCESS_ENTER
        )
        self.restartButton = wx.Button(self, label="Restart")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.EXPAND)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(self.choiceLabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        hsizer.Add(self.varChoice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        hsizer.Add(self.showAvgCheck, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        hsizer.Add(self.avgLabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        hsizer.Add(self.avgCtrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        hsizer.Add(self.restartButton, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(hsizer, 0, wx.ALIGN_CENTER_HORIZONTAL)
        self.SetSizerAndFit(sizer)

        self.avgCtrl.Bind(wx.EVT_TEXT_ENTER, self.on_avg_change)
        self.avgCtrl.Bind(wx.EVT_KILL_FOCUS, self.on_avg_change)
        self.restartButton.Bind(wx.EVT_BUTTON, self.on_restart)
        self.varChoice.Bind(wx.EVT_CHOICE, self.on_choice_change)
        self.showAvgCheck.Bind(wx.EVT_CHECKBOX, self.on_avg_toggle)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self._sync_avg_controls(show_avg_default)

    def on_choice_change(self, event):
        self.update_plot()
        event.Skip()

    def on_avg_toggle(self, event):
        enabled = self.showAvgCheck.GetValue()
        self._sync_avg_controls(enabled)
        if hasattr(self.parent, "set_live_plot_avg_enabled"):
            self.parent.set_live_plot_avg_enabled(enabled)
        self.update_plot()
        if event:
            event.Skip()

    def on_avg_change(self, event):
        self.update_plot()
        event.Skip()

    def on_restart(self, event):
        for key in self.parent.plot_data:
            self.parent.plot_data[key] = []
        for key in getattr(self.parent, "plot_errors", {}):
            self.parent.plot_errors[key] = []
        if hasattr(self.parent, "_last_plotted_file"):
            self.parent._last_plotted_file = None
        self.update_plot()
        event.Skip()

    def get_avg_len(self):
        try:
            return max(1, int(self.avgCtrl.GetValue()))
        except ValueError:
            return 1

    def update_plot(self):
        var = self.varChoice.GetStringSelection()
        data = self.parent.plot_data.get(var, [])
        if not len(data):
            self.axes.clear()
            self.canvas.draw()
            return

        arr = np.asarray(data, dtype=float)
        x = np.arange(len(arr))
        err_arr = np.asarray(self.parent.plot_errors.get(var, []), dtype=float)
        if len(err_arr) != len(arr):
            err_arr = None

        # Determine display units and labels
        if var == "Atom_Number":
            y = arr
            y_err = err_arr
            ylabel = "Atom # (millions)"
        elif var == "X_Atom_Number":
            y = arr
            y_err = err_arr
            ylabel = "X Atom # (millions)"
        elif var == "Y_Atom_Number":
            y = arr
            y_err = err_arr
            ylabel = "Y Atom # (millions)"
        elif var == "True_X_Width":
            y = arr * 1e6
            y_err = err_arr * 1e6 if err_arr is not None else None
            ylabel = "True X Width (µm)"
        elif var == "True_Y_Width":
            y = arr * 1e6
            y_err = err_arr * 1e6 if err_arr is not None else None
            ylabel = "True Y Width (µm)"
        elif var == "X_Center":
            y = arr
            y_err = err_arr
            ylabel = "X Center (px)"
        else:
            y = arr
            y_err = err_arr
            ylabel = "Y Center (px)"

        self.axes.clear()
        if y_err is not None and np.any(y_err):
            self.axes.errorbar(x, y, yerr=y_err, fmt="o", color="C0", label=var)
        else:
            self.axes.scatter(x, y, color="C0", label=var)

        if self.showAvgCheck.GetValue():
            avg_len = self.get_avg_len()
            if len(y) >= avg_len:
                kernel = np.ones(avg_len) / float(avg_len)
                run_avg = np.convolve(y, kernel, mode="valid")
                xavg = np.arange(avg_len - 1, len(y))
                if len(xavg) > 3:
                    x_smooth = np.linspace(xavg.min(), xavg.max(), len(xavg) * 10)
                    f = PchipInterpolator(xavg, run_avg)
                    y_smooth = f(x_smooth)
                    self.axes.plot(x_smooth, y_smooth, color="C1", label=f"Avg {avg_len}")
                else:
                    self.axes.plot(xavg, run_avg, color="C1", label=f"Avg {avg_len}")
                current = y[-1]
                current_avg = run_avg[-1]
                self.axes.set_title(
                    f"{var}={current:.3f}, Avg={current_avg:.3f}"
                )
            else:
                current = y[-1]
                self.axes.set_title(f"{var}={current:.3f}")
        else:
            current = y[-1]
            self.axes.set_title(f"{var}={current:.3f}")

        self.axes.set_xlabel("Shot #")
        self.axes.set_ylabel(ylabel)
        self.axes.legend()
        self.canvas.draw()

    def on_close(self, event):
        if hasattr(self.parent, "on_live_plot_closed"):
            self.parent.on_live_plot_closed(self.showAvgCheck.GetValue())
        event.Skip()

    def _sync_avg_controls(self, enabled):
        self.avgLabel.Enable(enabled)
        self.avgCtrl.Enable(enabled)

class ImageUI(wx.Frame):
    """Main application window for atom image analysis."""

    TAIL_FRACTION_DEFAULT = 0.5
    TAIL_FRACTION_MIN = 0.05
    TAIL_FRACTION_MAX = 0.95

    @staticmethod
    def _gaussian_atom_number_uncertainty(amplitude, sigma, amp_err, sigma_err, scale):
        """Propagate Gaussian fit uncertainties to atom number error."""

        try:
            amplitude = float(amplitude)
            sigma = float(sigma)
            scale = float(scale)
        except (TypeError, ValueError):
            return 0.0

        if not (np.isfinite(amplitude) and np.isfinite(sigma) and np.isfinite(scale)):
            return 0.0

        try:
            amp_err = float(amp_err)
        except (TypeError, ValueError):
            amp_err = 0.0
        try:
            sigma_err = float(sigma_err)
        except (TypeError, ValueError):
            sigma_err = 0.0

        if not np.isfinite(amp_err):
            amp_err = 0.0
        if not np.isfinite(sigma_err):
            sigma_err = 0.0

        variance = (sigma * amp_err) ** 2 + (amplitude * sigma_err) ** 2
        if variance <= 0.0 or not np.isfinite(variance):
            return 0.0

        return abs(scale) * np.sqrt(variance)

    def __init__(self, parent, title):
        # Start with a slightly taller window for better vertical space
        super(ImageUI, self).__init__(parent, title=title, size=(1000, 1300))

        self._settings_path = SETTINGS_FILE
        self._settings = {}
        self._startup_complete = False
        # Persist the preferred fit method so the first auto-run matches the
        # saved configuration (defaulting to Gaussian).
        self._preferred_fit_method = "Gaussian"

        self.atom = 'Cs'
        self.magnification = 0.33
        self.detuning_mhz = 0.0
        self.pixelSize = 3.45
        self.pixelToDistance = (self.pixelSize / self.magnification) * 1e-6
        self.crossSection = 1.0e-13
        self.mass = 132.905 * massUnit
        self.rawAtomNumber = 1
        self.atomNumberFrame = None
        self.plotsFrame = None
        self.live_plot_show_avg = False
        self.avgFrame = None
        self.avgPreviewActive = False
        self.avg_preview_count = 3
        self.avg_intensity_calibration = None
        self.avg_weight_provider = None
        self.avg_apply_saturation_correction = False
        self.avg_saturation_intensity = None
        self.avg_transition_gamma = None
        self.avg_probe_detuning = 0.0
        self.plot_data = {
            "True_X_Width": [],
            "True_Y_Width": [],
            "X_Center": [],
            "Y_Center": [],
            "Atom_Number": [],
            "X_Atom_Number": [],
            "Y_Atom_Number": [],
        }
        self.plot_errors = {
            "True_X_Width": [],
            "True_Y_Width": [],
            "X_Center": [],
            "Y_Center": [],
            "Atom_Number": [],
            "X_Atom_Number": [],
            "Y_Atom_Number": [],
        }
        self._last_plotted_file = None

        self.timeChanged = False
        self.chosenLayerNumber = 4
        self.expectedFileSize_mb = 18.2
        self.actualFileSize = 0
        self.monitor = Monitor("c:\\ ", self.autoRun, self.expectedFileSize_mb)
        self._watch_suspend_depth = 0
        self._pending_list_highlight = None

        self.gVals = None
        self.pVals = None
        self.fVals = None
        self.imageData = None
        self.atomImage = None
        self.AOIImage = None
        self._limited_aoi_image = None
        self._limited_atom_image = None
        self._limited_atom_mask = None
        self._limited_atom_od_mask = None
        self.x_summed = None
        self.y_summed = None
        self.currentXProfile = None
        self.currentYProfile = None
        self.currentXProfileFit = None
        self.currentYProfileFit = None
        self.x_fitted = None
        self.y_fitted = None
        self.x_r_squared = None
        self.y_r_squared = None

        self.isRotationNeeded = False
        self.autoExportEnabled = True
        self.prevImageAngle = 0.
        self.imageAngle = 0.
        self.imagePivotX = 1
        self.imagePivotY = 1

        self.atomNumFromFitX = -1
        self.atomNumFromFitY = -1

        self.atomNumFromGaussianX = -1
        self.atomNumFromGaussianY = -1

        self.atomNumFromGaussianX_std = 0.0
        self.atomNumFromGaussianY_std = 0.0
        self.atomNumFromFitX_std = 0.0
        self.atomNumFromFitY_std = 0.0

        self.atomNumFromDegenFitX = -1
        self.atomNumFromDegenFitY = -1

        self.isXFitSuccessful = False
        self.isYFitSuccessful = False

        self.x_center = 0.
        self.y_center = 0.
        self.x_width = 1.
        self.y_width = 1.

        self.x_width_std = 1.
        self.y_width_std = 1.

        self.x_center_std = 0.
        self.y_center_std = 0.

        self.x_offset = 0.
        self.y_offset = 0.
        self.x_peakHeight = 1.
        self.y_peakHeight = 1.
        self.x_slope = 0.
        self.y_slope = 0.

        self.true_x_width = 1.
        self.true_y_width = 1.

        self.true_x_width_std = 1.
        self.true_y_width_std = 1.

        self.true_x_width_list = []
        self.true_y_width_list = []
        self.true_x_center_list = []
        self.true_y_center_list = []
        self.atom_number_list = []
        self.tof_fit_result = None
        self.lifetime_fit_result = None
        self.mot_ringdown_fit_result = None
        self.avg_atom_number = 0
        self.tof_values = []
        self.tof_files_to_fit = []
        self.tof_fit_index = 0
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False
        self.tofOffset = 4.0
        self.selected_tof_fit = "Temperature, Density, and PSD"

        # Tracking information for TOF fitting sessions
        self.tof_run_id = None
        self.tof_run_records = []

        self.od = [0, 0]
        self._degeneracy_rows = []

        self.TOF = 1
        self.temperature = [0, 0]
        self.tempLongTime = [0, 0]
        self.xTrapFreq = 50
        self.yTrapFreq = 2000

        self.AOI = None

        self.fitOverlay = None
        self.quickFitBool = False

        self.xLeft = None
        self.xRight = None
        self.yBottom = None
        self.yTop = None

        self.rect =  None

        self.isFitSuccessful = False

        self.benchmark_startTime=0
        self.benchmark_endTime=0

        self.q = None
        self.gaussionParams = None
        self.fermionParams = None
        self.bosonParams = None

        self.path = None
        self.defringingRefPath = None
        self.filename = None
        self.fileIndex = 0
        self.Tmp = None
        self.data = None


        self.fileType = "fits"

        self.fileList = []

        self.InitUI()
        self.Centre()
        self.Show()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.redirect = RedirectText(self.consoleBox)
        sys.stdout = self.redirect
        sys.stderr = self.redirect

        self.timeString = None

        self.fitMethodGaussian.SetValue(True)
        self.layer4Button.SetValue(True)
        self.autoRunning = False
        self.modifiedFileName = None
        self.currentImg = None
        self.currentfitImage = None
        self.imageList = []
        self.imageListFlag = 0

        self.fitsFile.SetValue(True)

        self._load_settings()

        self.betterRef = None

        self.degenFitter = degenerateFitter()
        self.x_tOverTc = -1.
        self.x_thomasFermiRadius = 1.
        self.x_becPopulationRatio= 0.

        self.y_tOverTc = -1.
        self.y_thomasFermiRadius = 1.
        self.y_becPopulationRatio= 0.

        self.x_TOverTF = 1.
        self.x_fermiRadius = 1.

        self.y_TOverTF = 1.
        self.y_fermiRadius = 1.

        self.isMedianFilterOn = False
        self.isNormalizationOn = False
        # Delay enabling the directory watcher until the frame has finished
        # initialising so the first automatic fit uses the persisted
        # configuration.
        wx.CallAfter(self._finish_startup)


    def InitUI(self):
        # Use a scrolled panel that expands with the frame rather than a fixed size
        self.panel = wx.lib.scrolledpanel.ScrolledPanel(self)

        font1 = wx.Font(18, wx.DEFAULT, wx.NORMAL, wx.BOLD)

        # Main sizer for the top row which holds controls and image
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        vbox0 = wx.BoxSizer(wx.VERTICAL)

        settingBox = wx.StaticBox(self.panel, label = 'Settings')
        settingBoxSizer = wx.StaticBoxSizer(settingBox, wx.VERTICAL)

        fileTypeBox = wx.StaticBox(self.panel, label='File Type')
        fileTypeBoxSizer = wx.StaticBoxSizer(fileTypeBox, wx.HORIZONTAL)
                
        self.fitsFile = wx.RadioButton(self.panel, label="fits", style = wx.RB_GROUP )
        self.fitsFile.SetToolTip("Load .fits image files")
        self.aiaFile = wx.RadioButton(self.panel, label="aia")
        self.aiaFile.SetToolTip("Load .aia image files")
        self.tifFile = wx.RadioButton(self.panel, label="tif")
        self.tifFile.SetToolTip("Load .tif image files")

        fileTypeBoxSizer.Add(self.fitsFile, flag=wx.ALL, border=5)
        fileTypeBoxSizer.Add(self.aiaFile, flag=wx.ALL, border=5)
        fileTypeBoxSizer.Add(self.tifFile, flag=wx.ALL, border=5)

        self.Bind(wx.EVT_RADIOBUTTON, self.setFileType, self.aiaFile)
        self.Bind(wx.EVT_RADIOBUTTON, self.setFileType, self.tifFile)
        self.Bind(wx.EVT_RADIOBUTTON, self.setFileType, self.fitsFile)

        base = LOCAL_PATH.rstrip("\\")

        self.today = datetime.date.today()
        self.path = (
            os.path.join(
                base,
                str(self.today.year),
                str(self.today.month),
                str(self.today.day),
                "Cs",
            )
            + "\\"
        )
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except Exception:
                location = input("Where is the folder?: ")
                network_base = f"c:\\Users\{location}\\Desktop\\Image Network Folder".rstrip("\\")
                self.path = (
                    os.path.join(
                        network_base,
                        str(self.today.year),
                        str(self.today.month),
                        str(self.today.day),
                        "Cs",
                    )
                    + "\\"
                )

        self.imageFolderPath = wx.TextCtrl(self.panel, value=self.path)
        self.imageFolderPath.SetToolTip("Folder containing image files")
        self.choosePathButton = wx.Button(self.panel, label="Choose Path")
        self.choosePathButton.SetToolTip("Select the folder to load images from")
        self.choosePathButton.Bind(wx.EVT_BUTTON, self.choosePath)

        settingBoxSizer.Add(fileTypeBoxSizer, flag=wx.ALL, border=5)

        fermionOrBosonBox = wx.StaticBox(self.panel, label = 'Fit Type')
        fermionOrBosonBoxSizer = wx.StaticBoxSizer(fermionOrBosonBox, wx.HORIZONTAL)

        self.fitMethodGaussian = wx.RadioButton(self.panel, label="Gaussian", style = wx.RB_GROUP )
        self.fitMethodGaussian.SetToolTip("Fit using a Gaussian profile")
        self.fitMethodFermion = wx.RadioButton(self.panel, label="Fermion")
        self.fitMethodFermion.SetToolTip("Fit using a fermionic model")
        self.fitMethodBoson = wx.RadioButton(self.panel, label="Boson")
        self.fitMethodBoson.SetToolTip("Fit using a bosonic model")

        self.checkDisplayRadialAvg = wx.CheckBox(self.panel, label="Radially Averaged")
        self.checkDisplayRadialAvg.SetToolTip("Show radial average on profiles")
        self.Bind(wx.EVT_CHECKBOX, self.displayRadialAvg, id = self.checkDisplayRadialAvg.GetId())

        self.checkNormalization = wx.CheckBox(self.panel, label="Normalization")
        self.checkNormalization.SetToolTip("Normalize atom and reference images")
        self.Bind(wx.EVT_CHECKBOX, self.displayNormalization, id = self.checkNormalization.GetId())

        self.checkLimitOD = wx.CheckBox(self.panel, label="OD Limit")
        self.checkLimitOD.SetToolTip(
            "Exclude profile points outside the specified optical density range"
        )
        self.minODLabel = wx.StaticText(self.panel, label="Min:")
        self.minODCtrl = wx.TextCtrl(
            self.panel, value="", size=(25, -1), style=wx.TE_PROCESS_ENTER
        )
        self.minODCtrl.SetToolTip(
            "Minimum average OD per pixel retained in Gaussian fits"
        )
        if hasattr(self.minODCtrl, "SetHint"):
            self.minODCtrl.SetHint("Min")
        self.maxODLabel = wx.StaticText(self.panel, label="Max:")
        self.maxODCtrl = wx.TextCtrl(
            self.panel, value="3.0", size=(25, -1), style=wx.TE_PROCESS_ENTER
        )
        self.maxODCtrl.SetToolTip(
            "Maximum average OD per pixel used in Gaussian fits"
        )
        if hasattr(self.maxODCtrl, "SetHint"):
            self.maxODCtrl.SetHint("Max")
        odMaskSizer = wx.BoxSizer(wx.HORIZONTAL)
        odMaskSizer.Add(self.checkLimitOD, flag=wx.ALIGN_CENTER_VERTICAL)
        odMaskSizer.Add(self.minODLabel, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=4)
        odMaskSizer.Add(
            self.minODCtrl, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=2
        )
        odMaskSizer.Add(self.maxODLabel, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=4)
        odMaskSizer.Add(
            self.maxODCtrl, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=5
        )

        self.checkTailFit = wx.CheckBox(self.panel, label="Tail Fit")
        self.checkTailFit.SetToolTip(
            "Fit Gaussian width using only tail samples while plotting the full fit"
        )
        tail_fraction_column = wx.BoxSizer(wx.VERTICAL)
        tail_toggle_row = wx.BoxSizer(wx.HORIZONTAL)
        tail_toggle_row.Add(self.checkTailFit, 0, wx.ALIGN_CENTER_VERTICAL)
        tail_fraction_column.Add(tail_toggle_row, 0, wx.ALIGN_LEFT)

        tail_controls_row = wx.BoxSizer(wx.HORIZONTAL)
        tail_controls_row.AddSpacer(16)
        self.tailFractionCtrl = wx.TextCtrl(
            self.panel,
            value="0.50",
            size=(50, -1),
            style=wx.TE_PROCESS_ENTER,
        )
        self.tailFractionCtrl.SetToolTip(
            "Fraction (0.05–0.95) of the Gaussian amplitude retained when selecting tail samples"
        )
        tail_controls_row.Add(
            self.tailFractionCtrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5
        )

        self.tailLeftCheck = wx.CheckBox(self.panel, label="L")
        self.tailLeftCheck.SetValue(True)
        tail_controls_row.Add(self.tailLeftCheck, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.tailRightCheck = wx.CheckBox(self.panel, label="R")
        self.tailRightCheck.SetValue(True)
        tail_controls_row.Add(self.tailRightCheck, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 2)
        tail_controls_row.AddSpacer(8)
        self.tailTopCheck = wx.CheckBox(self.panel, label="T")
        self.tailTopCheck.SetValue(True)
        tail_controls_row.Add(self.tailTopCheck, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 2)
        self.tailBottomCheck = wx.CheckBox(self.panel, label="B")
        self.tailBottomCheck.SetValue(True)
        tail_controls_row.Add(self.tailBottomCheck, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 2)
        tail_fraction_column.Add(tail_controls_row, 0, wx.TOP, 4)
        self.tailControlsRow = tail_controls_row
        self.tailFractionRow = tail_fraction_column

        self._update_od_limit_visibility()
        self._update_tail_fit_visibility()

        self.Bind(wx.EVT_CHECKBOX, self.onLimitODToggle, id=self.checkLimitOD.GetId())
        self.Bind(wx.EVT_TEXT_ENTER, self.update1DProfilesAndFit, id=self.maxODCtrl.GetId())
        self.Bind(wx.EVT_TEXT_ENTER, self.update1DProfilesAndFit, id=self.minODCtrl.GetId())
        self.checkTailFit.Bind(wx.EVT_CHECKBOX, self.onTailFitToggle)
        self.tailFractionCtrl.Bind(wx.EVT_TEXT_ENTER, self.onTailFractionCommit)
        self.tailFractionCtrl.Bind(wx.EVT_KILL_FOCUS, self.onTailFractionCommit)
        self.tailLeftCheck.Bind(wx.EVT_CHECKBOX, self.update1DProfilesAndFit)
        self.tailRightCheck.Bind(wx.EVT_CHECKBOX, self.update1DProfilesAndFit)
        self.tailTopCheck.Bind(wx.EVT_CHECKBOX, self.update1DProfilesAndFit)
        self.tailBottomCheck.Bind(wx.EVT_CHECKBOX, self.update1DProfilesAndFit)

        self.Bind(wx.EVT_RADIOBUTTON, self.update1DProfilesAndFit, id = self.fitMethodFermion.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.update1DProfilesAndFit, id = self.fitMethodBoson.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.update1DProfilesAndFit, id = self.fitMethodGaussian.GetId())

        fermionOrBosonBoxSizer.Add(self.fitMethodGaussian, flag=wx.ALL, border=5)
        fermionOrBosonBoxSizer.Add(self.fitMethodFermion, flag=wx.ALL, border=5)
        fermionOrBosonBoxSizer.Add(self.fitMethodBoson, flag=wx.ALL, border=5)        

        # Subpanel for additional fit options
        fitOptionsBox = wx.StaticBox(self.panel, label='Fit Options')
        fitOptionsBoxSizer = wx.StaticBoxSizer(fitOptionsBox, wx.VERTICAL)
        fitOptionsBoxSizer.Add(self.checkNormalization, flag=wx.ALL, border=5)
        fitOptionsBoxSizer.Add(self.checkDisplayRadialAvg, flag=wx.ALL, border=5)
        fitOptionsBoxSizer.Add(odMaskSizer, flag=wx.ALL, border=5)
        fitOptionsBoxSizer.Add(self.tailFractionRow, flag=wx.ALL, border=5)

        fitOptionsBoxSizer.SetMinSize(fermionOrBosonBoxSizer.GetMinSize())

        settingBoxSizer.Add(fermionOrBosonBoxSizer, flag=wx.ALL | wx.EXPAND, border = 5)
        settingBoxSizer.Add(fitOptionsBoxSizer, flag=wx.ALL | wx.EXPAND, border = 5)

        # Ensure the settings panel expands to full column width
        vbox0.Add(settingBoxSizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        fittingBox = wx.StaticBox(self.panel, label='Auto Fitting')
        fittingBoxSizer = wx.StaticBoxSizer(fittingBox, wx.VERTICAL)

        # Image folder path controls
        imgFolderText = wx.StaticText(self.panel, label='Images')
        fittingBoxSizer.Add(imgFolderText, flag=wx.ALL, border=5)
        imgFolderSizer = wx.BoxSizer(wx.HORIZONTAL)
        imgFolderSizer.Add(self.imageFolderPath, 1, flag=wx.ALL | wx.EXPAND, border=5)
        imgFolderSizer.Add(self.choosePathButton, flag=wx.ALL, border=5)
        fittingBoxSizer.Add(imgFolderSizer, flag=wx.ALL | wx.EXPAND, border=0)

        # Data folder for fit results
        default_snippet_path = os.path.join(
            os.path.expanduser("~"),
            "Desktop",
            "Data",
            f"{self.today.year}",
            f"{self.today.month:02d}",
            f"{self.today.day:02d}",
        )
        os.makedirs(default_snippet_path, exist_ok=True)
        self.snippetPath = default_snippet_path
        dataText = wx.StaticText(self.panel, label='Data')
        self.snippetTextBox = wx.TextCtrl(self.panel, value=self.snippetPath)
        self.snippetTextBox.SetToolTip("Folder storing fit results")
        self.snippetTextBox.Bind(wx.EVT_TEXT, self.setSnippetPath)
        snippetBoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        snippetBoxSizer.Add(self.snippetTextBox, 1, flag=wx.ALL | wx.EXPAND, border=5)
        self.chooseSnippetButton = wx.Button(self.panel, label='Choose Path')
        self.chooseSnippetButton.SetToolTip("Select folder to store fit results")
        self.chooseSnippetButton.Bind(wx.EVT_BUTTON, self.chooseSnippetPath)
        snippetBoxSizer.Add(self.chooseSnippetButton, flag=wx.ALL, border=5)
        fittingBoxSizer.Add(dataText, flag=wx.ALL, border=5)
        fittingBoxSizer.Add(snippetBoxSizer, flag=wx.ALL | wx.EXPAND, border=0)

        # Controls for starting auto run and saving fits
        self.autoButton = wx.Button(self.panel, label='Start')
        self.autoButton.SetToolTip("Start continuous monitoring for new images")
        self.autoButton.Bind(wx.EVT_BUTTON, self.startAutoRun)
        fittingBoxSizer.Add(self.autoButton, flag=wx.ALL, border=5)

        self.checkAutoExport = wx.CheckBox(self.panel, label="Auto")
        self.checkAutoExport.SetValue(True)
        self.checkAutoExport.SetToolTip("Auto append new fit results to csv")
        self.checkAutoExport.Bind(wx.EVT_CHECKBOX, self.toggleAutoExport)
        self.saveSnippetButton = wx.Button(self.panel, label="Save Fit")
        self.saveSnippetButton.SetToolTip("Save current fit results to CSV")
        self.saveSnippetButton.Bind(wx.EVT_BUTTON, self.saveSnippetNow)
        self._sync_auto_save_controls()
        saveSizer = wx.BoxSizer(wx.HORIZONTAL)
        saveSizer.Add(self.saveSnippetButton, flag=wx.ALL, border=5)
        saveSizer.Add(
            self.checkAutoExport,
            flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL,
            border=5,
        )
        fittingBoxSizer.Add(saveSizer, flag=wx.ALL, border=0)

        listText = wx.StaticText(self.panel, label='Image List')
        # Make the list of images a bit smaller to free up space
        self.imageListBox = wx.ListBox(self.panel, size=(225, 255))
        self.imageListBox.SetToolTip("List of images found in the folder")
        self.Bind(wx.EVT_LISTBOX, self.chooseImg, self.imageListBox)
        fittingBoxSizer.Add(listText, flag=wx.ALL, border=5)
        fittingBoxSizer.Add(self.imageListBox, 1, wx.ALL, border=5)
        self.updateImageListBox()

        # Make the reading panel match the width of the settings panel
        vbox0.Add(fittingBoxSizer, 0, wx.EXPAND | wx.ALL, 5)

        toolsBox = wx.StaticBox(self.panel, label='Tools')
        toolsBoxSizer = wx.StaticBoxSizer(toolsBox, wx.VERTICAL)

        self.fitWindowBtn = wx.Button(self.panel, label="Fitting")
        self.fitWindowBtn.SetToolTip("Open the detailed fitting window")
        self.fitWindowBtn.Bind(wx.EVT_BUTTON, self.openFittingWindow)

        self.showPlotsBtn = wx.Button(self.panel, label="Live Plot")
        self.showPlotsBtn.SetToolTip("View live plot of image analysis.")
        self.showPlotsBtn.Bind(wx.EVT_BUTTON, self.togglePlots)

        self.avgPreviewBtn = wx.Button(self.panel, label="Avg Images")
        self.avgPreviewBtn.SetToolTip("Average last N images before analysis")
        self.avgPreviewBtn.Bind(wx.EVT_BUTTON, self.toggleAvgPreview)

        btns = [self.fitWindowBtn, self.showPlotsBtn, self.avgPreviewBtn]
        btn_width = max(btn.GetBestSize().GetWidth() for btn in btns)
        for btn in btns:
            btn.SetMinSize((btn_width, btn.GetBestSize().GetHeight()))

        toolsBoxSizer.Add(self.fitWindowBtn, flag=wx.ALL, border=5)
        toolsBoxSizer.Add(self.showPlotsBtn, flag=wx.ALL, border=5)

        avgSizer = wx.BoxSizer(wx.HORIZONTAL)
        avgSizer.Add(self.avgPreviewBtn, flag=wx.ALIGN_CENTER_VERTICAL)
        toolsBoxSizer.Add(avgSizer, flag=wx.ALL, border=5)

        vbox0.Add(toolsBoxSizer, 0, wx.EXPAND | wx.ALL, 5)

        # Left column added without expanding, keeping a narrow gap to images
        top_sizer.Add(vbox0, 0, wx.EXPAND | wx.LEFT | wx.TOP | wx.BOTTOM, 5)
        top_sizer.AddSpacer(2)

        imagesBox = wx.StaticBox(self.panel, label='Images')
        imagesBoxSizer = wx.StaticBoxSizer(imagesBox, wx.VERTICAL)

        # Subpanel to group canvas, layer selection, and cursor readout
        plotPanel = wx.Panel(self.panel)

        figure = Figure()
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace = 0, hspace = 0)

        self.axes1 = figure.add_subplot(gs[:-1, :-1])
        self.axes1.set_xticks([])
        self.axes1.set_title('Image Data', fontsize=12)
        for label in (self.axes1.get_xticklabels() + self.axes1.get_yticklabels()):
            label.set_fontsize(10)

        self.axes2 = figure.add_subplot(gs[-1, 0:-1])
        self.axes2.grid(False)
        for label in (self.axes2.get_xticklabels() + self.axes2.get_yticklabels()):
            label.set_fontsize(10)

        self.axes3 = figure.add_subplot(gs[:-1, -1])
        self.axes3.grid(False)
        for label in (self.axes3.get_xticklabels()):
            label.set_fontsize(10)

        for label in (self.axes3.get_yticklabels()):
            label.set_visible(False)

        # Canvas lives inside the dedicated plot panel
        self.canvas1 =  FigureCanvas(plotPanel, -1, figure)
        self.canvas1.SetToolTip("Main image display")
        self.canvas1.mpl_connect('button_press_event', self.on_press)
        self.canvas1.mpl_connect('button_release_event', self.on_release)
        self.canvas1.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas1.mpl_connect('motion_notify_event', self.showImgValue)

        self.press= None
        # Flag to defer auto loading when ROI box is being resized
        self._pending_autorun = False

        hbox41 = wx.BoxSizer(wx.HORIZONTAL)
        self.layer1Button = wx.RadioButton(plotPanel, label="Probe With Atoms", style = wx.RB_GROUP )
        self.layer1Button.SetToolTip("Display probe-with-atoms image")
        self.layer2Button = wx.RadioButton(plotPanel, label="Probe Without Atoms")
        self.layer2Button.SetToolTip("Display probe-without-atoms image")
        self.layer3Button = wx.RadioButton(plotPanel, label="Dark Field")
        self.layer3Button.SetToolTip("Display dark field image")
        self.layer4Button = wx.RadioButton(plotPanel, label="Absorption Image")
        self.layer4Button.SetToolTip("Display calculated absorption image")
        self.flipHCheck = wx.CheckBox(plotPanel, label="H Flip")
        self.flipHCheck.SetToolTip("Flip image horizontally")
        self.flipVCheck = wx.CheckBox(plotPanel, label="V Flip")
        self.flipVCheck.SetToolTip("Flip image vertically")
        self._colormap_options = ABSORPTION_COLORMAP_OPTIONS
        colormap_labels = [label for label, _ in self._colormap_options]
        text_widths = [plotPanel.GetTextExtent(label)[0] for label in colormap_labels]
        choice_width = max(text_widths, default=60) + 16  # Keep dropdown compact, similar to prior checkbox width
        self.colormapChoice = wx.Choice(
            plotPanel,
            choices=colormap_labels,
            size=(choice_width, -1),
        )
        self.colormapChoice.SetToolTip("Select the default absorption image color map")
        default_idx = next(
            (idx for idx, (_, cmap) in enumerate(self._colormap_options) if cmap == "gray_r"),
            0,
        )
        self.colormapChoice.SetSelection(default_idx)
        hasFileSizeChanged = False
        self.Bind(wx.EVT_RADIOBUTTON, lambda e: self.updateImageOnUI(1, hasFileSizeChanged), id=self.layer1Button.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, lambda e: self.updateImageOnUI(2, hasFileSizeChanged), id=self.layer2Button.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, lambda e: self.updateImageOnUI(3, hasFileSizeChanged), id=self.layer3Button.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, lambda e: self.updateImageOnUI(4, hasFileSizeChanged), id=self.layer4Button.GetId())
        self.flipHCheck.Bind(wx.EVT_CHECKBOX, self.onFlipImage)
        self.flipVCheck.Bind(wx.EVT_CHECKBOX, self.onFlipImage)
        self.colormapChoice.Bind(wx.EVT_CHOICE, self.onColormapChanged)
        self.layer4Button.SetValue(True)
        self.chosenLayerNumber = 4
        self.flipVCheck.SetValue(True)

        hbox41.Add(self.layer1Button, flag=wx.ALL, border=5)
        hbox41.Add(self.layer2Button, flag=wx.ALL, border=5)
        hbox41.Add(self.layer3Button, flag=wx.ALL, border=5)
        hbox41.Add(self.layer4Button, flag=wx.ALL, border=5)

        imageLayersBox = wx.StaticBox(plotPanel, label="Image Layers")
        imageLayersSizer = wx.StaticBoxSizer(imageLayersBox, wx.HORIZONTAL)
        imageLayersSizer.Add(hbox41, 0, wx.ALL, 5)

        imageOptionsBox = wx.StaticBox(plotPanel, label="Image Options")
        imageOptionsSizer = wx.StaticBoxSizer(imageOptionsBox, wx.HORIZONTAL)

        optionsRowSizer = wx.BoxSizer(wx.HORIZONTAL)
        optionsRowSizer.Add(self.flipHCheck, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        optionsRowSizer.Add(self.flipVCheck, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        optionsRowSizer.Add(self.colormapChoice, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)

        # Legacy processing options (defringing, AOI reset, and median filtering)
        # have been retired. Placeholders are kept for compatibility, but the
        # controls are intentionally omitted from the UI and processing
        # pipeline. See ``setData`` and related methods for the corresponding
        # deactivated code paths.
        self.checkApplyDefringing = None
        self.checkResetAOI = None
        self.checkMedianFilter = None

        imageOptionsSizer.Add(optionsRowSizer, 0, wx.ALL, 5)

        hbox42 = wx.BoxSizer(wx.HORIZONTAL)
        boldFont = wx.Font(10, wx.DECORATIVE, wx.NORMAL, wx.BOLD)
        st17 = wx.StaticText(plotPanel, label='X:')
        self.cursorX = wx.TextCtrl(plotPanel,  style=wx.TE_READONLY|wx.TE_CENTRE, size = (50, 22))
        self.cursorX.SetToolTip("Cursor X position")
        st18 = wx.StaticText(plotPanel, label='Y:')
        self.cursorY = wx.TextCtrl(plotPanel, style=wx.TE_READONLY|wx.TE_CENTRE,  size = (50, 22))
        self.cursorY.SetToolTip("Cursor Y position")
        st19 = wx.StaticText(plotPanel, label='Value:')
        self.cursorZ = wx.TextCtrl(plotPanel, style=wx.TE_READONLY|wx.TE_CENTRE,  size = (80, 22))
        self.cursorZ.SetToolTip("Pixel value at cursor")
        hbox42.Add(st17, flag=wx.ALL, border=4)
        hbox42.Add(self.cursorX, flag=wx.ALL, border=5)
        hbox42.Add(st18, flag=wx.ALL, border=4)
        hbox42.Add(self.cursorY, flag=wx.ALL, border=5)
        hbox42.Add(st19, flag=wx.ALL, border=5)
        hbox42.Add(self.cursorZ, flag=wx.ALL, border=5)

        pixelValueBox = wx.StaticBox(plotPanel, label="Pixel Value")
        pixelValueSizer = wx.StaticBoxSizer(pixelValueBox, wx.VERTICAL)

        self.odMaskInfo = wx.StaticText(plotPanel, label="")
        self.odMaskInfo.Hide()

        hbox42.Add(self.odMaskInfo, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        pixelValueSizer.Add(hbox42, 0, wx.ALL, 5)

        self.cursorX.SetFont(boldFont)
        self.cursorY.SetFont(boldFont)
        self.cursorZ.SetFont(boldFont)

        # Combine canvas, layer selection, and cursor readout into a dedicated subpanel
        plotSizer = wx.BoxSizer(wx.VERTICAL)
        plotSizer.Add(self.canvas1, 0, wx.EXPAND | wx.ALL, 5)
        plotSizer.Add(imageLayersSizer, 0, wx.EXPAND | wx.ALL, 5)
        plotSizer.Add(pixelValueSizer, 0, wx.EXPAND | wx.ALL, 5)
        plotSizer.Add(imageOptionsSizer, 0, wx.EXPAND | wx.ALL, 5)
        plotPanel.SetSizer(plotSizer)
        plotPanel.Fit()
        canvas_size = self.canvas1.GetSize()
        self.canvas1.SetMinSize(canvas_size)
        self.canvas1.SetMaxSize(canvas_size)
        plotPanel.SetMinSize(plotPanel.GetSize())
        plotPanel.SetMaxSize(plotPanel.GetSize())

        imagesBoxSizer.Add(plotPanel, 0, wx.ALL, 5)

        rotationBox = wx.StaticBox(self.panel, label="Angle Options")
        rotationBoxSizer = wx.StaticBoxSizer(rotationBox, wx.HORIZONTAL)

        angle = wx.StaticText(self.panel, label = "Image angle (" + u"\u00b0" + "):")
        self.angleBox = wx.TextCtrl(self.panel, value = str(self.imageAngle), size = (40, 22))
        self.angleBox.SetToolTip("Image rotation angle in degrees")
        pivot = wx.StaticText(self.panel, label = "Rotation pivot index (x, y):")
        self.pivotXBox = wx.TextCtrl(self.panel, value = str(self.imagePivotX), size = (40, 22))
        self.pivotXBox.SetToolTip("Rotation pivot X index")
        self.pivotYBox = wx.TextCtrl(self.panel, value = str(self.imagePivotY), size = (40, 22))
        self.pivotYBox.SetToolTip("Rotation pivot Y index")
        arrorText = wx.StaticText(self.panel, label = u"\u27A4" + u"\u27A4")
        self.rotationButton= wx.Button(self.panel, label ="Set angle && pivot")
        self.rotationButton.SetToolTip("Apply rotation to current image")

        self.angleBox.Bind(wx.EVT_TEXT, self.setImageAngle)
        self.pivotXBox.Bind(wx.EVT_TEXT, self.setImagePivotX)
        self.pivotYBox.Bind(wx.EVT_TEXT, self.setImagePivotY)
        self.rotationButton.Bind(wx.EVT_BUTTON, self.setImageRotationParams)

        rotationBoxSizer.Add(angle, flag = wx.ALL, border = 5)
        rotationBoxSizer.Add(self.angleBox, flag = wx.ALL, border = 5)
        rotationBoxSizer.Add(pivot, flag = wx.ALL, border = 5)
        rotationBoxSizer.Add(self.pivotXBox, flag = wx.ALL, border = 5)
        rotationBoxSizer.Add(self.pivotYBox, flag = wx.ALL, border = 5)
        rotationBoxSizer.Add(arrorText, flag = wx.ALL, border = 10)
        rotationBoxSizer.Add(self.rotationButton, flag = wx.ALL, border = 2)

        aoiBox = wx.StaticBox(self.panel, label="AOI Options")
        hbox14 = wx.BoxSizer(wx.HORIZONTAL)
        aoiBoxSizer = wx.StaticBoxSizer(aoiBox, wx.HORIZONTAL)
        aoiText = wx.StaticText(self.panel, label = 'AOI: (x,y)->(x,y)')
        hbox14.Add(aoiText, flag=wx.ALL, border=5)


        self.AOI1 = wx.TextCtrl(self.panel, value='-1', size=(40,22))
        self.AOI1.SetToolTip("AOI x start")
        self.AOI2 = wx.TextCtrl(self.panel, value='-1', size=(40,22))
        self.AOI2.SetToolTip("AOI y start")
        self.AOI3 = wx.TextCtrl(self.panel, value='-1', size=(40,22))
        self.AOI3.SetToolTip("AOI x end")
        self.AOI4 = wx.TextCtrl(self.panel, value='-1', size=(40,22))
        self.AOI4.SetToolTip("AOI y end")
        self.resetAOIButton = wx.Button(self.panel, label="Reset", size=(50, 22))
        self.resetAOIButton.SetToolTip("Reset AOI to the maximum allowable size")
        hbox14.Add(self.AOI1, flag=wx.ALL, border=2)
        hbox14.Add(self.AOI2, flag=wx.ALL, border=2)
        hbox14.Add(self.AOI3, flag=wx.ALL, border=2)
        hbox14.Add(self.AOI4, flag=wx.ALL, border=2)
        hbox14.Add(self.resetAOIButton, flag=wx.ALL, border=2)
        aoiBoxSizer.Add(
            hbox14,
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )

        hbox43 = wx.BoxSizer(wx.HORIZONTAL)
        removeLabel = wx.StaticText(self.panel, label="Remove:")
        self.leftRightEdge = wx.CheckBox(self.panel, label="L/R")
        self.leftRightEdge.SetToolTip(
            "Average the left and right 3-pixel edge strips to compute the offset "
            "subtracted from the atom count"
        )
        self.updownEdge = wx.CheckBox(self.panel, label="T/B")
        self.updownEdge.SetToolTip(
            "Average the top and bottom 3-pixel edge strips to compute the offset "
            "subtracted from the atom count"
        )
        self.Bind(wx.EVT_CHECKBOX, self.edgeUpdate, id = self.leftRightEdge.GetId())
        self.Bind(wx.EVT_CHECKBOX, self.edgeUpdate, id = self.updownEdge.GetId())
        hbox43.Add(removeLabel, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        hbox43.Add(self.leftRightEdge, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=5)
        hbox43.Add(self.updownEdge, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=5)
        self.leftRightEdge.SetValue(True)
        self.updownEdge.SetValue(True)
        self.resetAOIButton.Bind(wx.EVT_BUTTON, self.onResetAOI)
        aoiBoxSizer.Add(
            hbox43,
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )

        imagesBoxSizer.Add(aoiBoxSizer, flag=wx.ALL | wx.EXPAND, border= 5)
        imagesBoxSizer.Add(rotationBoxSizer, flag = wx.ALL | wx.EXPAND, border = 5)

        atomNum = wx.StaticBox(self.panel, label='# of Atoms')
        atomNumBoxSizer = wx.StaticBoxSizer(atomNum, wx.HORIZONTAL)

        atomKind = ['Cs', 'Li', 'Ag']
        self.atomRadioBox = wx.RadioBox(self.panel, choices=atomKind, majorDimension=2)
        self.atomRadioBox.SetToolTip("Select atomic species")
        self.atomRadioBox.Bind(wx.EVT_RADIOBOX, self.onAtomRadioClicked)

        atomTypeSizer = wx.BoxSizer(wx.VERTICAL)
        atomTypeSizer.Add(self.atomRadioBox, flag=wx.ALL, border=5)

        atomInfoBox = wx.StaticBox(self.panel, label="")
        atomInfoSizer = wx.StaticBoxSizer(atomInfoBox, wx.HORIZONTAL)

        magnif = wx.StaticText(atomInfoBox, label='Mag:')
        self.magnif = wx.TextCtrl(atomInfoBox, value=str(self.magnification), size=(30, 22))
        self.magnif.SetToolTip("Microscope magnification factor")
        self.magnif.Bind(wx.EVT_TEXT, self.setMagnification)

        detuning_label = wx.StaticText(atomInfoBox, label=u"\u0394 (MHz):")
        self.detuningCtrl = wx.TextCtrl(
            atomInfoBox,
            value=f"{self.detuning_mhz:g}",
            size=(45, 22),
            style=wx.TE_PROCESS_ENTER,
        )
        self.detuningCtrl.SetToolTip("Probe detuning from resonance in MHz")
        self.detuningCtrl.Bind(wx.EVT_TEXT_ENTER, self.setDetuning)
        self.detuningCtrl.Bind(wx.EVT_KILL_FOCUS, self.setDetuning)

        pixelSize = wx.StaticText(atomInfoBox, label=u"\u00B5" + "m//pix:")
        self.pxSize = wx.TextCtrl(atomInfoBox, value=str(self.pixelSize), size=(35, 22))
        self.pxSize.SetToolTip("Camera pixel size in μm")
        self.pxSize.Bind(wx.EVT_TEXT, self.setPixelSize)

        bigfont = wx.Font(12, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        bigNcountText3 = wx.StaticText(atomInfoBox, label='Atoms (M):')
        self.bigNcount3 = wx.TextCtrl(atomInfoBox, style=wx.TE_READONLY | wx.TE_CENTRE, size=(92, 22))
        self.bigNcount3.SetToolTip("Estimated atom number")
        self.bigNcount3.SetFont(bigfont)

        self.showAtomNumberCheck = wx.CheckBox(atomInfoBox, label="Pop Out")
        self.showAtomNumberCheck.SetToolTip("Open large atom number display")
        self.showAtomNumberCheck.Bind(wx.EVT_CHECKBOX, self.toggleAtomNumberFrame)

        atomInfoSizer.Add(magnif, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(self.magnif, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(pixelSize, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(self.pxSize, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(detuning_label, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(self.detuningCtrl, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(bigNcountText3, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(self.bigNcount3, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomInfoSizer.Add(self.showAtomNumberCheck, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)

        atomNumBoxSizer.Add(atomTypeSizer, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        atomNumBoxSizer.Add(atomInfoSizer, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)

        imagesBoxSizer.Add(atomNumBoxSizer, flag=wx.ALL | wx.EXPAND, border=5)

        fittingResultDisplay = wx.StaticBox(self.panel, label = "Fitting results")
        fittingResultDisplaySizer = wx.StaticBoxSizer(fittingResultDisplay, wx.VERTICAL)



        xRadiusText = wx.StaticText(
            self.panel, label="X Radius (" + u"\u00B5" + "m):",
        )
        self.xRadiusBox = wx.TextCtrl(
            self.panel,
            value="0 ± 0",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(100, 22),
        )
        self.xRadiusBox.SetToolTip(
            "Gaussian sigma radius along x in μm (scaled by pixel size and magnification)",
        )
        xCenterText = wx.StaticText(
            self.panel, label="X Center (px):",
        )
        self.xCenterBox = wx.TextCtrl(
            self.panel,
            value="0 ± 0",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(100, 22),
        )
        self.xCenterBox.SetToolTip("Gaussian center along x in pixels")
        xRSquaredText = wx.StaticText(self.panel, label="X Fit R^2:")
        self.xRSquaredBox = wx.TextCtrl(
            self.panel,
            value="N/A",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(70, 22),
        )
        self.xRSquaredBox.SetToolTip(
            "Coefficient of determination for the X-axis fit"
        )

        yRadiusText = wx.StaticText(
            self.panel, label="Y Radius (" + u"\u00B5" + "m):",
        )
        self.yRadiusBox = wx.TextCtrl(
            self.panel,
            value="0 ± 0",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(100, 22),
        )
        self.yRadiusBox.SetToolTip(
            "Gaussian sigma radius along y in μm (scaled by pixel size and magnification)",
        )
        yCenterText = wx.StaticText(
            self.panel, label="Y Center (px):",
        )
        self.yCenterBox = wx.TextCtrl(
            self.panel,
            value="0 ± 0",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(100, 22),
        )
        self.yCenterBox.SetToolTip("Gaussian center along y in pixels")
        yRSquaredText = wx.StaticText(self.panel, label="Y Fit R^2:")
        self.yRSquaredBox = wx.TextCtrl(
            self.panel,
            value="N/A",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(70, 22),
        )
        self.yRSquaredBox.SetToolTip(
            "Coefficient of determination for the Y-axis fit"
        )

        self.TcText = wx.StaticText(self.panel, label="(T//Tc, Nc//N) :")
        self.TcBox = wx.TextCtrl(
            self.panel,
            value=str(1) + ",  " + str(0),
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(75, 22),
        )
        self.TcBox.SetToolTip("Bose condensate fraction")
        self.TFRadiusText = wx.StaticText(
            self.panel, label="TF rad. (" + u"\u00B5" + "m):"
        )
        self.TFRadiusBox = wx.TextCtrl(
            self.panel,
            value=str(1),
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(55, 22),
        )
        self.TFRadiusBox.SetToolTip("Fermi radius in μm")

        self.tempEstText = wx.StaticText(
            self.panel, label="Temp Est. (" + u"\u00B5" + "K): "
        )
        self.tempLongText = wx.StaticText(
            self.panel, label="Temp Long Time Est. (" + u"\u00B5" + "K): "
        )
        self.tempBox = wx.TextCtrl(
            self.panel,
            value="(" + str(self.temperature[0]) + ", " + str(self.temperature[1]) + ")",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(160, 22),
        )
        self.tempBox.SetToolTip("Fitted temperature in μK")
        self.tempBox2 = wx.TextCtrl(
            self.panel,
            value="(" + str(self.temperature[0]) + ", " + str(self.temperature[1]) + ")",
            style=wx.TE_READONLY | wx.TE_CENTRE,
            size=(160, 22),
        )
        self.tempBox2.SetToolTip("Long time limit temperature in μK")
        bigfont2 = wx.Font(10, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tempBox.SetFont(bigfont2)
        self.tempBox2.SetFont(bigfont2)

        xResultRow = wx.BoxSizer(wx.HORIZONTAL)
        xResultRow.Add(xRadiusText, flag=wx.ALL, border=5)
        xResultRow.Add(self.xRadiusBox, flag=wx.ALL, border=5)
        xResultRow.Add(xCenterText, flag=wx.ALL, border=5)
        xResultRow.Add(self.xCenterBox, flag=wx.ALL, border=5)
        xResultRow.Add(xRSquaredText, flag=wx.ALL, border=5)
        xResultRow.Add(self.xRSquaredBox, flag=wx.ALL, border=5)

        yResultRow = wx.BoxSizer(wx.HORIZONTAL)
        yResultRow.Add(yRadiusText, flag=wx.ALL, border=5)
        yResultRow.Add(self.yRadiusBox, flag=wx.ALL, border=5)
        yResultRow.Add(yCenterText, flag=wx.ALL, border=5)
        yResultRow.Add(self.yCenterBox, flag=wx.ALL, border=5)
        yResultRow.Add(yRSquaredText, flag=wx.ALL, border=5)
        yResultRow.Add(self.yRSquaredBox, flag=wx.ALL, border=5)

        degeneracyRow = wx.BoxSizer(wx.HORIZONTAL)
        degeneracyRow.Add(self.TcText, flag=wx.ALL, border=5)
        degeneracyRow.Add(self.TcBox, flag=wx.ALL, border=5)
        degeneracyRow.Add(self.TFRadiusText, flag=wx.ALL, border=5)
        degeneracyRow.Add(self.TFRadiusBox, flag=wx.ALL, border=5)

        temperatureRow = wx.BoxSizer(wx.HORIZONTAL)
        temperatureRow.Add(self.tempEstText, flag=wx.ALL, border=5)
        temperatureRow.Add(self.tempBox, flag=wx.ALL, border=5)
        temperatureRow.Add(self.tempLongText, flag=wx.ALL, border=5)
        temperatureRow.Add(self.tempBox2, flag=wx.ALL, border=5)

        fittingResultDisplaySizer.Add(xResultRow, flag=wx.ALL, border=5)
        fittingResultDisplaySizer.Add(yResultRow, flag=wx.ALL, border=5)
        fittingResultDisplaySizer.Add(degeneracyRow, flag=wx.ALL, border=5)
        fittingResultDisplaySizer.Add(temperatureRow, flag=wx.ALL, border=5)
        self._degeneracy_rows = [degeneracyRow, temperatureRow]
        self._degeneracy_controls = [
            self.TcText,
            self.TcBox,
            self.TFRadiusText,
            self.TFRadiusBox,
            self.tempEstText,
            self.tempBox,
            self.tempLongText,
            self.tempBox2,
        ]
        self._degeneracy_parent = fittingResultDisplaySizer
        imagesBoxSizer.Add(fittingResultDisplaySizer, flag=wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, border=5)

        # Add the images panel to the top row
        top_sizer.Add(imagesBoxSizer, 2, wx.EXPAND | wx.RIGHT | wx.TOP | wx.BOTTOM, 5)

        # Terminal output window sized for roughly 10 lines of text
        self.consoleBox = wx.TextCtrl(
            self.panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            size=(-1, 210),  # increase height from ~7 lines (~150 px) to ~10 lines
        )
        self.consoleBox.SetToolTip("Logs and messages")
        self.consoleBox.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.consoleBox.SetForegroundColour(wx.Colour(255, 255, 255))
        # Keep console height fixed while allowing width to expand
        self.consoleBox.SetMinSize((-1, 210))
        self.consoleBox.SetMaxSize((-1, 210))

        # Combine top row and console into the panel
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(top_sizer, 1, wx.EXPAND)
        main_sizer.Add(self.consoleBox, 0, wx.EXPAND | wx.ALL, 5)

        # Set the sizer for the scrolled panel and enable scrolling
        self.panel.SetSizer(main_sizer)
        self.panel.Layout()
        self.panel.SetupScrolling()
        self._update_degeneracy_visibility()

        # Frame sizer containing only the panel
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(frame_sizer)

    def _load_settings(self):
        """Load persisted UI settings from disk and apply them to the controls."""

        try:
            with open(self._settings_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._settings = data
            else:
                self._settings = {}
        except FileNotFoundError:
            self._settings = {}
        except Exception as err:
            print(f"Failed to load settings: {err}")
            self._settings = {}

        self._apply_settings()

    def _apply_settings(self):
        """Update the UI controls using the currently loaded settings."""

        settings = self._settings if isinstance(self._settings, dict) else {}

        file_type = settings.get("file_type")
        file_buttons = {
            "fits": self.fitsFile,
            "aia": self.aiaFile,
            "tif": self.tifFile,
        }
        if file_type in file_buttons:
            file_buttons[file_type].SetValue(True)
            self.fileType = file_type

        # Image folder and snippet paths are intentionally not restored from
        # saved settings so that the default data hierarchy is used each time
        # the application starts.
        self.path = self.imageFolderPath.GetValue()
        self.snippetPath = self.snippetTextBox.GetValue()

        auto_export = settings.get("auto_export")
        if isinstance(auto_export, bool):
            self.checkAutoExport.SetValue(auto_export)
        self.autoExportEnabled = self.checkAutoExport.GetValue()
        self._sync_auto_save_controls()

        colormap_name = settings.get("absorption_colormap")
        if not isinstance(colormap_name, str):
            jet_colormap = settings.get("absorption_use_jet")
            if isinstance(jet_colormap, bool):
                colormap_name = "jet" if jet_colormap else "gray_r"
        if isinstance(colormap_name, str) and getattr(self, "colormapChoice", None):
            self._select_absorption_colormap(colormap_name)

        fit_buttons = {
            "Gaussian": self.fitMethodGaussian,
            "Fermion": self.fitMethodFermion,
            "Boson": self.fitMethodBoson,
        }

        preferred_fit = None
        fit_method = settings.get("fit_method")
        if isinstance(fit_method, str):
            normalized = fit_method.strip().lower()
            mapping = {
                "gaussian": "Gaussian",
                "fermion": "Fermion",
                "boson": "Boson",
            }
            preferred_fit = mapping.get(normalized)
            if preferred_fit is None and normalized:
                # If a modern ``fit_method`` key is present but unrecognised,
                # honour the user's intent by defaulting to Gaussian instead of
                # falling back to legacy boolean flags that might point to an
                # outdated choice.
                preferred_fit = "Gaussian"

        if preferred_fit is None:
            legacy_method = settings.get("fitMethod")
            if isinstance(legacy_method, str) and legacy_method in fit_buttons:
                preferred_fit = legacy_method

        if preferred_fit is None and "fit_method" not in settings:
            for key, method_name in (
                ("fitMethodGaussian", "Gaussian"),
                ("fitMethodFermion", "Fermion"),
                ("fitMethodBoson", "Boson"),
            ):
                legacy_value = settings.get(key)
                if isinstance(legacy_value, bool) and legacy_value:
                    preferred_fit = method_name
                    break

        if preferred_fit in fit_buttons:
            self._preferred_fit_method = preferred_fit
        else:
            # Default to Gaussian when no prior preference was saved so the
            # automatic fit starts from the expected profile.
            self._preferred_fit_method = "Gaussian"

        if isinstance(self._settings, dict) and self._preferred_fit_method:
            self._settings["fit_method"] = self._preferred_fit_method
        self._ensure_fit_method_selection()

        normalization = settings.get("normalization")
        if isinstance(normalization, bool):
            self.checkNormalization.SetValue(normalization)
        self.isNormalizationOn = self.checkNormalization.GetValue()

        tail_fit_only = settings.get("fit_tails_only")
        if isinstance(tail_fit_only, bool):
            self.checkTailFit.SetValue(tail_fit_only)

        tail_fraction_setting = settings.get("tail_fraction")
        if tail_fraction_setting is not None:
            sanitized = self._sanitize_tail_fraction(tail_fraction_setting)
        else:
            sanitized = self.TAIL_FRACTION_DEFAULT
        if getattr(self, "tailFractionCtrl", None) is not None:
            self.tailFractionCtrl.ChangeValue(self._format_tail_fraction(sanitized))

        for setting_key, attr_name in (
            ("tail_left", "tailLeftCheck"),
            ("tail_right", "tailRightCheck"),
            ("tail_top", "tailTopCheck"),
            ("tail_bottom", "tailBottomCheck"),
        ):
            ctrl = getattr(self, attr_name, None)
            if ctrl is None:
                continue
            value = settings.get(setting_key)
            if isinstance(value, bool):
                ctrl.SetValue(value)
            elif setting_key not in settings:
                ctrl.SetValue(True)

        radial_avg = settings.get("radial_avg")
        if isinstance(radial_avg, bool):
            self.checkDisplayRadialAvg.SetValue(radial_avg)

        limit_od = settings.get("limit_od")
        if isinstance(limit_od, bool):
            self.checkLimitOD.SetValue(limit_od)

        min_od = settings.get("min_od")
        if min_od is not None and getattr(self, "minODCtrl", None) is not None:
            self.minODCtrl.ChangeValue(str(min_od))

        max_od = settings.get("max_od")
        if max_od is not None:
            self.maxODCtrl.ChangeValue(str(max_od))

        self._update_od_limit_visibility()
        self._update_tail_fit_visibility()

        layer = settings.get("layer")
        layer_buttons = {
            1: self.layer1Button,
            2: self.layer2Button,
            3: self.layer3Button,
            4: self.layer4Button,
        }
        if layer in layer_buttons:
            layer_buttons[layer].SetValue(True)
            self.chosenLayerNumber = layer
        else:
            for idx, btn in layer_buttons.items():
                if btn.GetValue():
                    self.chosenLayerNumber = idx
                    break

        flip_h = settings.get("flip_horizontal")
        if isinstance(flip_h, bool):
            self.flipHCheck.SetValue(flip_h)

        flip_v = settings.get("flip_vertical")
        if isinstance(flip_v, bool):
            self.flipVCheck.SetValue(flip_v)

        # Legacy processing options (defringing, AOI reset, and median filtering)
        # have been removed. Preserve deterministic behaviour by forcing the
        # associated flags to ``False`` regardless of any persisted settings.
        self.isMedianFilterOn = False

        self._apply_aoi_settings(settings.get("aoi"))

        angle = settings.get("image_angle")
        if angle is not None:
            try:
                self.imageAngle = float(angle)
            except (TypeError, ValueError):
                pass
            else:
                self.angleBox.ChangeValue(str(angle))

        pivot_x = settings.get("pivot_x")
        if pivot_x is not None:
            try:
                self.imagePivotX = int(float(pivot_x))
            except (TypeError, ValueError):
                pass
            else:
                self.pivotXBox.ChangeValue(str(pivot_x))

        pivot_y = settings.get("pivot_y")
        if pivot_y is not None:
            try:
                self.imagePivotY = int(float(pivot_y))
            except (TypeError, ValueError):
                pass
            else:
                self.pivotYBox.ChangeValue(str(pivot_y))

        use_left = settings.get("use_left_right_edge")
        if isinstance(use_left, bool):
            self.leftRightEdge.SetValue(use_left)

        use_top = settings.get("use_top_bottom_edge")
        if isinstance(use_top, bool):
            self.updownEdge.SetValue(use_top)

        atom_choice = settings.get("atom")
        if isinstance(atom_choice, str) and atom_choice in [self.atomRadioBox.GetString(i) for i in range(self.atomRadioBox.GetCount())]:
            self.atomRadioBox.SetStringSelection(atom_choice)
            self.atom = atom_choice

        magnification = settings.get("magnification")
        if magnification is not None:
            self.magnif.ChangeValue(str(magnification))
            try:
                self.magnification = float(magnification)
            except (TypeError, ValueError):
                pass

        pixel_size = settings.get("pixel_size")
        if pixel_size is not None:
            self.pxSize.ChangeValue(str(pixel_size))
            try:
                self.pixelSize = float(pixel_size)
            except (TypeError, ValueError):
                pass

        detuning = settings.get("detuning_mhz")
        if detuning is not None:
            self.detuningCtrl.ChangeValue(str(detuning))
            try:
                self.detuning_mhz = float(detuning)
            except (TypeError, ValueError):
                pass

        show_atom = settings.get("show_atom_number")
        if isinstance(show_atom, bool):
            self.showAtomNumberCheck.SetValue(show_atom)
            if show_atom:
                self.toggleAtomNumberFrame(None)

        live_plot_avg = settings.get("live_plot_show_avg")
        self.live_plot_show_avg = bool(live_plot_avg) if isinstance(live_plot_avg, bool) else False

        self.pixelToDistance = (self.pixelSize / self.magnification) * 1e-6
        self.setConstants()
        self.updateImageListBox()

    def _ensure_aoi_patch(self):
        """Create and attach the AOI rectangle patch if it does not exist."""

        rect = getattr(self, "rect", None)
        if rect is not None:
            return rect

        axes = getattr(self, "axes1", None)
        if axes is None:
            return None

        try:
            rect = matplotlib.patches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="none",
                linewidth=2,
                edgecolor="#0000ff",
            )
        except Exception:
            return None

        try:
            axes.add_patch(rect)
        except Exception:
            return None

        self.rect = rect
        return rect

    def _active_image_shape_for_aoi(self):
        """Return the (rows, cols) shape backing the AOI controls, if available."""

        if isinstance(self.atomImage, np.ndarray):
            return self.atomImage.shape

        if isinstance(self.imageData, (list, tuple)) and self.imageData:
            first = self.imageData[0]
            if isinstance(first, np.ndarray):
                return first.shape

        current = getattr(self, "currentImg", None)
        if current is not None:
            try:
                data = current.get_array()
            except Exception:
                data = None
            if isinstance(data, np.ndarray):
                return data.shape

        return None

    def onResetAOI(self, _event=None):
        """Reset the AOI text boxes to the maximum allowable region."""

        shape = self._active_image_shape_for_aoi()
        if not shape:
            return

        y_size, x_size = shape
        x_min = 3
        y_min = 3
        x_max = max(x_min, x_size - 4)
        y_max = max(y_min, y_size - 4)

        self.xLeft = x_min
        self.yTop = y_min
        self.xRight = x_max
        self.yBottom = y_max
        self.AOI = [[self.xLeft, self.yTop], [self.xRight, self.yBottom]]

        self.AOI1.SetValue(str(self.xLeft))
        self.AOI2.SetValue(str(self.yTop))
        self.AOI3.SetValue(str(self.xRight))
        self.AOI4.SetValue(str(self.yBottom))

        rect = self._ensure_aoi_patch()
        if rect is not None:
            rect.set_width(self.xRight - self.xLeft)
            rect.set_height(self.yBottom - self.yTop)
            rect.set_xy((self.xLeft, self.yTop))
            self.canvas1.draw()

        self.updateAOIImageAndProfiles()
        self.setAtomNumber()

    def _apply_aoi_settings(self, data):
        """Update AOI controls and cached coordinates from persisted settings."""

        if not isinstance(data, dict):
            return

        def _coerce(value):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

        coords = {
            "x_left": _coerce(data.get("x_left")),
            "y_top": _coerce(data.get("y_top")),
            "x_right": _coerce(data.get("x_right")),
            "y_bottom": _coerce(data.get("y_bottom")),
        }

        for key, ctrl_name in (
            ("x_left", "AOI1"),
            ("y_top", "AOI2"),
            ("x_right", "AOI3"),
            ("y_bottom", "AOI4"),
        ):
            value = coords[key]
            ctrl = getattr(self, ctrl_name, None)
            if ctrl is not None and value is not None:
                try:
                    ctrl.ChangeValue(str(value))
                except Exception:
                    pass

        if any(value is not None for value in coords.values()):
            if coords["x_left"] is not None:
                self.xLeft = coords["x_left"]
            if coords["y_top"] is not None:
                self.yTop = coords["y_top"]
            if coords["x_right"] is not None:
                self.xRight = coords["x_right"]
            if coords["y_bottom"] is not None:
                self.yBottom = coords["y_bottom"]

            if all(value is not None for value in coords.values()):
                self.AOI = [
                    [self.xLeft, self.yTop],
                    [self.xRight, self.yBottom],
                ]
                rect = self._ensure_aoi_patch()
                if rect is not None:
                    try:
                        rect.set_width(self.xRight - self.xLeft)
                        rect.set_height(self.yBottom - self.yTop)
                        rect.set_xy((self.xLeft, self.yTop))
                    except Exception:
                        pass
                try:
                    self.updateAOIImageAndProfiles()
                except Exception:
                    pass

    def apply_settings_snapshot(self, data):
        """Apply a previously captured settings dictionary to the UI."""

        if not isinstance(data, dict):
            return
        self._settings = data
        self._apply_settings()
        saver = getattr(self, "_save_settings", None)
        if callable(saver):
            try:
                saver()
            except Exception as err:
                print(f"Failed to persist restored settings: {err}")
        if getattr(self, "currentImg", None):
            self._apply_display_settings()
            self.canvas1.draw()

    def _current_fit_method_from_controls(self):
        """Return the label of the currently selected fit method radio."""

        fit_buttons = [
            ("Gaussian", getattr(self, "fitMethodGaussian", None)),
            ("Fermion", getattr(self, "fitMethodFermion", None)),
            ("Boson", getattr(self, "fitMethodBoson", None)),
        ]

        for label, btn in fit_buttons:
            if btn is None:
                continue
            try:
                if btn.GetValue():
                    return label
            except Exception:
                continue

        return getattr(self, "_preferred_fit_method", "Gaussian")

    def _ensure_fit_method_selection(self, preferred=None):
        """Ensure the radio buttons reflect the preferred fit method."""

        if preferred is None:
            preferred = getattr(self, "_preferred_fit_method", None)

        fit_buttons = {
            "Gaussian": getattr(self, "fitMethodGaussian", None),
            "Fermion": getattr(self, "fitMethodFermion", None),
            "Boson": getattr(self, "fitMethodBoson", None),
        }

        if preferred not in fit_buttons:
            return

        for label, btn in fit_buttons.items():
            if btn is None:
                continue
            try:
                btn.SetValue(label == preferred)
            except Exception:
                # Some environments can raise if the control is being destroyed
                # (e.g. during shutdown). Ignore those cases so start-up logic
                # remains robust.
                pass

    def _update_od_limit_visibility(self):
        """Show or hide the OD limit text control based on the checkbox state."""

        should_show = bool(getattr(self, "checkLimitOD", None) and self.checkLimitOD.GetValue())

        for ctrl_name in ("minODLabel", "minODCtrl", "maxODLabel", "maxODCtrl"):
            ctrl = getattr(self, ctrl_name, None)
            if ctrl is not None:
                ctrl.Show(should_show)

        if hasattr(self, "panel") and self.panel:
            try:
                self.panel.Layout()
            except Exception:
                # In some startup sequences Layout can fail before the window is fully
                # initialised; ignore these cases so visibility stays consistent.
                pass

    def _update_tail_fit_visibility(self):
        """Show or hide tail-fit controls based on the checkbox state."""

        should_show = bool(getattr(self, "checkTailFit", None) and self.checkTailFit.GetValue())

        controls = [
            getattr(self, "tailFractionCtrl", None),
            getattr(self, "tailLeftCheck", None),
            getattr(self, "tailRightCheck", None),
            getattr(self, "tailTopCheck", None),
            getattr(self, "tailBottomCheck", None),
        ]
        for ctrl in controls:
            if ctrl is not None:
                ctrl.Show(should_show)
                ctrl.Enable(should_show)

        tail_container = getattr(self, "tailFractionRow", None)
        tail_controls_row = getattr(self, "tailControlsRow", None)
        if tail_container is not None and tail_controls_row is not None:
            try:
                tail_container.Show(tail_controls_row, should_show)
            except Exception:
                pass

        if hasattr(self, "panel") and self.panel:
            try:
                self.panel.Layout()
            except Exception:
                pass

    def _sanitize_tail_fraction(self, value):
        """Return a bounded, numeric tail-fraction value."""

        try:
            fraction = float(value)
        except (TypeError, ValueError):
            fraction = self.TAIL_FRACTION_DEFAULT

        if not np.isfinite(fraction):
            fraction = self.TAIL_FRACTION_DEFAULT

        return float(
            max(
                self.TAIL_FRACTION_MIN,
                min(self.TAIL_FRACTION_MAX, fraction),
            )
        )

    def _format_tail_fraction(self, value):
        return f"{value:.2f}"

    def _get_tail_fraction(self):
        ctrl = getattr(self, "tailFractionCtrl", None)
        value = ctrl.GetValue() if ctrl is not None else self.TAIL_FRACTION_DEFAULT
        return self._sanitize_tail_fraction(value)

    def _get_tail_side_keywords(self):
        """Return the requested tail selections for each axis."""

        def _axis_keyword(negative_ctrl, positive_ctrl, negative_label, positive_label):
            neg = getattr(self, negative_ctrl, None)
            pos = getattr(self, positive_ctrl, None)
            neg_val = bool(neg.GetValue()) if neg is not None else True
            pos_val = bool(pos.GetValue()) if pos is not None else True

            if neg_val and not pos_val:
                return negative_label
            if pos_val and not neg_val:
                return positive_label
            return None

        return {
            "x": _axis_keyword("tailLeftCheck", "tailRightCheck", "left", "right"),
            "y": _axis_keyword("tailTopCheck", "tailBottomCheck", "top", "bottom"),
        }

    def onTailFitToggle(self, event):
        self._update_tail_fit_visibility()
        if getattr(self, "tailFractionCtrl", None) is not None:
            sanitized = self._sanitize_tail_fraction(self.tailFractionCtrl.GetValue())
            self.tailFractionCtrl.ChangeValue(self._format_tail_fraction(sanitized))
        self.update1DProfilesAndFit(event)

    def onTailFractionCommit(self, event):
        if getattr(self, "tailFractionCtrl", None) is not None:
            sanitized = self._sanitize_tail_fraction(self.tailFractionCtrl.GetValue())
            self.tailFractionCtrl.ChangeValue(self._format_tail_fraction(sanitized))
        if getattr(self, "checkTailFit", None) and self.checkTailFit.GetValue():
            self.update1DProfilesAndFit(event)
        if event:
            event.Skip()

    def _sync_preferred_fit_method_from_ui(self):
        """Update the preferred fit method based on the current selection."""

        if getattr(self, "_suppress_fit_sync", False):
            return

        try:
            if self.fitMethodFermion.GetValue():
                self._preferred_fit_method = "Fermion"
            elif self.fitMethodBoson.GetValue():
                self._preferred_fit_method = "Boson"
            else:
                self._preferred_fit_method = "Gaussian"
        except Exception:
            self._preferred_fit_method = "Gaussian"

    def _get_selected_layer(self):
        """Return the currently selected image layer number."""

        buttons = [self.layer1Button, self.layer2Button, self.layer3Button, self.layer4Button]
        for idx, btn in enumerate(buttons, start=1):
            if btn.GetValue():
                return idx
        return self.chosenLayerNumber

    def _collect_settings(self):
        """Gather the current UI state into a serializable dictionary."""

        self._sync_preferred_fit_method_from_ui()

        settings = {
            "file_type": self.fileType,
            "auto_export": self.checkAutoExport.GetValue(),
            "fit_method": (
                "Fermion"
                if self.fitMethodFermion.GetValue()
                else "Boson"
                if self.fitMethodBoson.GetValue()
                else "Gaussian"
            ),
            "normalization": self.checkNormalization.GetValue(),
            "fit_tails_only": self.checkTailFit.GetValue(),
            "tail_fraction": self._get_tail_fraction(),
            "tail_left": getattr(self, "tailLeftCheck", None).GetValue()
            if getattr(self, "tailLeftCheck", None)
            else True,
            "tail_right": getattr(self, "tailRightCheck", None).GetValue()
            if getattr(self, "tailRightCheck", None)
            else True,
            "tail_top": getattr(self, "tailTopCheck", None).GetValue()
            if getattr(self, "tailTopCheck", None)
            else True,
            "tail_bottom": getattr(self, "tailBottomCheck", None).GetValue()
            if getattr(self, "tailBottomCheck", None)
            else True,
            "radial_avg": self.checkDisplayRadialAvg.GetValue(),
            "limit_od": self.checkLimitOD.GetValue(),
            "min_od": self.minODCtrl.GetValue(),
            "max_od": self.maxODCtrl.GetValue(),
            "layer": self._get_selected_layer(),
            "flip_horizontal": self.flipHCheck.GetValue(),
            "flip_vertical": self.flipVCheck.GetValue(),
            "absorption_colormap": self._get_selected_colormap(),
            "absorption_use_jet": self._get_selected_colormap() == "jet",
            "image_angle": self.angleBox.GetValue(),
            "pivot_x": self.pivotXBox.GetValue(),
            "pivot_y": self.pivotYBox.GetValue(),
            "use_left_right_edge": self.leftRightEdge.GetValue(),
            "use_top_bottom_edge": self.updownEdge.GetValue(),
            "atom": self.atomRadioBox.GetStringSelection(),
            "magnification": self.magnif.GetValue(),
            "pixel_size": self.pxSize.GetValue(),
            "detuning_mhz": self.detuningCtrl.GetValue(),
            "show_atom_number": self.showAtomNumberCheck.GetValue(),
            "live_plot_show_avg": bool(getattr(self, "live_plot_show_avg", False)),
        }

        aoi_snapshot = self._collect_aoi_snapshot()
        if aoi_snapshot is not None:
            settings["aoi"] = aoi_snapshot

        return settings

    def _collect_aoi_snapshot(self):
        """Return the current AOI coordinates as a dict, if available."""

        ctrl_mapping = (
            ("x_left", "AOI1", "xLeft"),
            ("y_top", "AOI2", "yTop"),
            ("x_right", "AOI3", "xRight"),
            ("y_bottom", "AOI4", "yBottom"),
        )

        snapshot = {}
        found = False

        for key, ctrl_name, attr_name in ctrl_mapping:
            value = None
            ctrl = getattr(self, ctrl_name, None)
            if ctrl is not None:
                try:
                    value = int(float(ctrl.GetValue()))
                except (TypeError, ValueError):
                    raw = ctrl.GetValue()
                    if isinstance(raw, str):
                        stripped = raw.strip()
                        value = stripped if stripped else None
                    else:
                        value = None
            if value is None:
                attr_value = getattr(self, attr_name, None)
                if isinstance(attr_value, (int, float)) and not isinstance(attr_value, bool):
                    value = int(attr_value)
            if value is not None:
                found = True
            snapshot[key] = value

        return snapshot if found else None

    def _save_settings(self):
        """Persist the current UI state to disk."""

        settings = self._collect_settings()
        try:
            with open(self._settings_path, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=2, sort_keys=True)
        except Exception as err:
            print(f"Failed to save settings: {err}")
        else:
            self._settings = settings

    def setDefringingRefPath(self):
        tempPath = self.path[:-1] + "_ref\\"
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)
        self.defringingRefPath = tempPath

    def applyDefringing(self):
        """Compatibility stub for the retired defringing option."""
        # The defringing workflow has been decommissioned. Retain the method so
        # any lingering event bindings or external references fail gracefully
        # while ensuring the image refresh logic still runs when invoked.
        self.setDataAndUpdate()

    def doFitList(self, numOfImages):
        """Fit the most recent ``numOfImages`` files without user interaction."""
        self.true_x_width_list = []
        self.true_y_width_list = []
        self.atom_number_list = []
        self.avg_atom_number = 0

        self.updateFileList()
        num = min(numOfImages, len(self.fileList))
        files_to_fit = list(reversed(self.fileList[-num:]))

        for fname in files_to_fit:
            self.filename = fname
            # Reuse existing fitting routine with current ROI
            self.fitImage(None)
            self.true_x_width_list.append(self.true_x_width)
            self.true_y_width_list.append(self.true_y_width)
            self.atom_number_list.append(
                self.rawAtomNumber * (self.pixelToDistance ** 2) / self.crossSection
            )

        return 0

    def openFittingWindow(self, event):
        """Open the standalone fitting window."""
        if getattr(self, "fitWindow", None) is None or not self.fitWindow:
            self.fitWindow = FittingWindow(self)
        if getattr(self, "fitWindowBtn", None) is not None:
            try:
                self.fitWindowBtn.Disable()
            except Exception:
                pass
        self.fitWindow.Show()
        self.fitWindow.Raise()

    def showTOFFit(self, e):
        """Start an interactive TOF fit over the most recent images."""
        print("Starting TOF fit")
        self.tofFitStartButton.Disable()
        self.stopTOFFitButton.Enable()
        self._tof_fit_running = True
        try:
            base_list = [float(t) for t in self.TOFFitList.GetValue().splitlines() if t.strip()]
            try:
                self.tofOffset = float(self.tofOffsetBox.GetValue())
            except Exception:
                self.tofOffset = 0.0
            # The newest images appear first in the list box. Reverse the TOF
            # values so the entered order (oldest first) matches the image
            # ordering.
            self.tof_values = list(reversed([t + self.tofOffset for t in base_list]))
        except Exception:
            print("------TOF List format is wrong-----")
            msg = wx.MessageDialog(self, "TOF time input format is wrong!", 'TOF Time Input Error', wx.OK)
            if msg.ShowModal() == wx.ID_OK:
                msg.Destroy()
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            return

        if len(self.tof_values) == 0:
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            return

        self.updateFileList()
        num = min(len(self.tof_values), len(self.fileList))
        self.tof_files_to_fit = list(reversed(self.fileList[-num:]))
        self.tof_fit_index = 0
        self.true_x_width_list = []
        self.true_y_width_list = []
        self.true_x_center_list = []
        self.true_y_center_list = []
        self.atom_number_list = []
        self.avg_atom_number = 0
        self.selected_tof_fit = self.tofFitTypeChoice.GetStringSelection()
        self.tof_run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.tof_run_records = []

        if self.tofAutoCheck.GetValue():
            self._tof_auto_mode = True
            self._tof_auto_paused = False
            self.continueTOFFitButton.Disable()
            self._autoTOFFit()
        else:
            self._tof_auto_mode = False
            self.continueTOFFitButton.Enable()
            self._loadCurrentTOFImage()
            print("Waiting for continue...")

    def _loadCurrentTOFImage(self):
        """Load and display the current TOF image for ROI selection."""
        fname = self.tof_files_to_fit[self.tof_fit_index]
        tof_ms = self.tof_values[self.tof_fit_index]
        print(f"Loading {fname} with TOF {tof_ms} ms")
        self.filename = fname
        # Load and display the selected file without overriding the filename
        self.fitImage(wx.EVT_BUTTON)

    def continueTOFFit(self, e):
        """Record ROI from the current image and proceed to the next."""
        exclude = False
        if self.tofExcludePointCheck.IsShown():
            exclude = self.tofExcludePointCheck.GetValue()
            self.tofExcludePointCheck.Hide()
            self._tofExcludePlaceholder.Hide()
            self.panel.Layout()

        if exclude:
            del self.tof_values[self.tof_fit_index]
            del self.tof_files_to_fit[self.tof_fit_index]
        else:
            tof_ms = self.tof_values[self.tof_fit_index]
            fname = self.tof_files_to_fit[self.tof_fit_index]
            atom_num = (
                self.rawAtomNumber * (self.pixelToDistance ** 2) / self.crossSection
            )
            self.true_x_width_list.append(self.true_x_width)
            self.true_y_width_list.append(self.true_y_width)
            self.true_x_center_list.append(self.x_center * self.pixelToDistance)
            self.true_y_center_list.append(self.y_center * self.pixelToDistance)
            self.atom_number_list.append(atom_num)
            self.tof_run_records.append(
                {
                    "image_file": fname,
                    "tof_ms": tof_ms,
                    "sigma_x": self.true_x_width,
                    "sigma_y": self.true_y_width,
                    "x_center": self.x_center * self.pixelToDistance,
                    "y_center": self.y_center * self.pixelToDistance,
                    "atom_number": atom_num,
                }
            )
            self.tof_fit_index += 1

        if self.tof_fit_index < len(self.tof_files_to_fit):
            if self._tof_auto_mode and self._tof_auto_paused:
                self._tof_auto_paused = False
                self.continueTOFFitButton.Disable()
                self._autoTOFFit()
                return
            self._loadCurrentTOFImage()
            print("Waiting for continue...")
            return

        # All images processed
        self.continueTOFFitButton.Disable()
        self._finalizeTOFFit()

    def _autoTOFFit(self):
        """Run TOF fit automatically and handle failed Gaussian fits."""
        while self.tof_fit_index < len(self.tof_files_to_fit):
            # Ensure failed-fit options are hidden before each attempt
            self.tofExcludePointCheck.Hide()
            self._tofExcludePlaceholder.Hide()
            self.panel.Layout()

            self.filename = self.tof_files_to_fit[self.tof_fit_index]
            # Fit current image using existing ROI
            self.fitImage(None)
            # Let the UI process any queued events before continuing
            wx.YieldIfNeeded()

            fit_failed = (
                not self.isXFitSuccessful
                or not self.isYFitSuccessful
                or self.x_width == 0
                or self.y_width == 0
            )

            if fit_failed:
                # Pause auto mode for manual adjustment or exclusion
                self._tof_auto_paused = True
                self.continueTOFFitButton.Enable()
                self.tofExcludePointCheck.SetValue(False)
                self.tofExcludePointCheck.Show()
                self._tofExcludePlaceholder.Show()
                self.panel.Layout()
                self._loadCurrentTOFImage()
                print(
                    "Gaussian fit failed or zero width. Adjust ROI or choose to exclude this point..."
                )
                return

            tof_ms = self.tof_values[self.tof_fit_index]
            fname = self.tof_files_to_fit[self.tof_fit_index]
            atom_num = (
                self.rawAtomNumber * (self.pixelToDistance ** 2) / self.crossSection
            )
            self.true_x_width_list.append(self.true_x_width)
            self.true_y_width_list.append(self.true_y_width)
            self.true_x_center_list.append(self.x_center * self.pixelToDistance)
            self.true_y_center_list.append(self.y_center * self.pixelToDistance)
            self.atom_number_list.append(atom_num)
            self.tof_run_records.append(
                {
                    "image_file": fname,
                    "tof_ms": tof_ms,
                    "sigma_x": self.true_x_width,
                    "sigma_y": self.true_y_width,
                    "x_center": self.x_center * self.pixelToDistance,
                    "y_center": self.y_center * self.pixelToDistance,
                    "atom_number": atom_num,
                }
            )
            self.tof_fit_index += 1

        # Completed all images
        self._finalizeTOFFit()

    def stopTOFFit(self, e):
        """Abort the TOF fit and reset the UI."""
        self.continueTOFFitButton.Disable()
        self.stopTOFFitButton.Disable()
        self.tofFitStartButton.Enable()
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False
        self.tof_fit_index = 0
        self.true_x_width_list = []
        self.true_y_width_list = []
        self.true_x_center_list = []
        self.true_y_center_list = []
        self.tof_files_to_fit = []
        print("TOF fit stopped")


    def _finalizeTOFFit(self):
        """Finalize TOF fit calculations once all images are processed."""
        if self.selected_tof_fit == "Gravity":
            self._finalizeGravityFit()
            return
        if self.selected_tof_fit == "Molasses":
            self._finalizeMolassesFit()
            return
        if self.selected_tof_fit == "MOT Lifetime":
            self._finalizeLifetimeFit()
            return
        if self.selected_tof_fit == "MOT Ringdown":
            self._finalizeMotRingdownFit()
            return
        if self.selected_tof_fit == "MOT COM_y k_spring":
            self._finalizeMotComYkSpringFit()
            return

        print("Running regression...")
        # Ensure the mass corresponds to the currently selected atom
        self.setConstants()
        atom = self.atom

        # Use all recorded widths and TOF values
        time_sec = np.array(self.tof_values) * 1e-3
        widths_x = np.array(self.true_x_width_list)
        widths_y = np.array(self.true_y_width_list)

        try:
            tx, ty, wx_f, wy_f = dataFit(atom, time_sec, widths_x, widths_y)

            # Regression of $\sigma^2$ versus $t^2$ using all data points
            slopeX, bX, rX, pX, sX = stats.linregress(
                np.square(time_sec), np.square(widths_x)
            )
            slopeY, bY, rY, pY, sY = stats.linregress(
                np.square(time_sec), np.square(widths_y)
            )
        except Exception as err:
            print(f"Regression failed: {err}")
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return

        self.temperature[0] = slopeX * self.mass / kB * 1e6
        self.temperature[1] = slopeY * self.mass / kB * 1e6
        self.tempBox.SetValue("(%.3f, %.3f)" % (self.temperature[0], self.temperature[1]))
        tx_err = sX * self.mass / kB * 1e6
        ty_err = sY * self.mass / kB * 1e6

        print(f"Temperatures are {self.temperature[0]:.3f}, {self.temperature[1]:.3f}")

        # Phase space density calculation using Gaussian volume
        sigma_x0 = np.sqrt(np.maximum(bX, 0))
        sigma_y0 = np.sqrt(np.maximum(bY, 0))
        sigma_z0 = sigma_x0  # estimate if z not measured

        if self.atom_number_list:
            atom_number = np.mean(self.atom_number_list)
        else:
            atom_number = self.rawAtomNumber * (self.pixelToDistance ** 2) / self.crossSection
        atom_number_err = np.sqrt(abs(atom_number))
        self.avg_atom_number = atom_number

        temp_x_K = self.temperature[0] * 1e-6
        temp_y_K = self.temperature[1] * 1e-6
        tx_err_K = tx_err * 1e-6
        ty_err_K = ty_err * 1e-6

        psd, psd_err = calculate_phase_space_density(
            atom_number,
            temp_x_K,
            temp_y_K,
            self.mass,
            sigma_x0,
            sigma_y0,
            sigma_z0,
            temp_z_K=temp_x_K,
            atom_number_err=atom_number_err,
            sigma_x_err=self.true_x_width_std,
            sigma_y_err=self.true_y_width_std,
            sigma_z_err=self.true_x_width_std,
            temp_x_err_K=tx_err_K,
            temp_y_err_K=ty_err_K,
            temp_z_err_K=tx_err_K,
        )

        density, density_err = calculate_peak_density(
            atom_number,
            sigma_x0,
            sigma_y0,
            sigma_z0,
            atom_number_err=atom_number_err,
            sigma_x_err=self.true_x_width_std,
            sigma_y_err=self.true_y_width_std,
            sigma_z_err=self.true_x_width_std,
        )

        self.phase_space_density = psd
        self.tof_fit_result = {
            'tof_list': self.tof_values,
            'temp_x': self.temperature[0],
            'temp_y': self.temperature[1],
            'trap_wx': wx_f,
            'trap_wy': wy_f,
            'avg_atom_number': self.avg_atom_number,
            'temp_x_err': tx_err,
            'temp_y_err': ty_err,
            'psd': self.phase_space_density,
            'psd_err': psd_err,
            'density': density,
            'density_err': density_err,
        }
        print("TOF fit finished")
        if self.tofPlotCheck.GetValue():
            self._plotTOFFit(
                self.tof_values,
                widths_x,
                widths_y,
                slopeX,
                bX,
                rX,
                slopeY,
                bY,
                rY,
                self.temperature[0],
                self.temperature[1],
                tx_err,
                ty_err,
                self.phase_space_density,
                psd_err,
                density,
                density_err,
                self.avg_atom_number,
            )
        self.tofFitStartButton.Enable()
        self.stopTOFFitButton.Disable()
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False
        if self.tofAutoCheck.GetValue():
            self.saveTOFFit(None)

    def _finalizeGravityFit(self):
        time_sec = np.array(self.tof_values) * 1e-3
        positions_m = np.array(self.true_y_center_list)
        try:
            coeffs = np.polyfit(time_sec, positions_m, 2)
            fit_poly = np.poly1d(coeffs)
            fitted = fit_poly(time_sec)
            ss_res = np.sum((positions_m - fitted) ** 2)
            ss_tot = np.sum((positions_m - np.mean(positions_m)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            vy0 = coeffs[1]
            a0 = coeffs[0]
            accel = 2 * a0
        except Exception as err:
            print(f"Gravity fit failed: {err}")
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return
        self.gravity_fit_result = {
            'tof_list': self.tof_values,
            'y_center': self.true_y_center_list,
            'y0': coeffs[2],
            'vy0': vy0,
            'accel': accel,
            'r2': r2,
        }
        print("Gravity fit finished")
        if self.tofPlotCheck.GetValue():
            self._plotGravityFit(time_sec, positions_m, coeffs, r2, vy0, a0)

        self.tofFitStartButton.Enable()
        self.stopTOFFitButton.Disable()
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False
        if self.tofAutoCheck.GetValue():
            self.saveTOFFit(None)

    def _plotGravityFit(self, time_sec, positions_m, coeffs, r2, vy0, a0):
        time_ms = time_sec * 1e3
        pos_um = positions_m * 1e6
        t_range = np.linspace(np.min(time_sec), np.max(time_sec), 100)
        fit_poly = np.poly1d(coeffs)
        fig, ax = plt.subplots()
        ax.scatter(time_ms, pos_um, color="tab:blue", label="Data")
        ax.plot(
            t_range * 1e3,
            fit_poly(t_range) * 1e6,
            color="tab:red",
            label=f"y={coeffs[2]:.2e}+{coeffs[1]:.2e}t+{coeffs[0]:.2e}t^2\n$R^2$={r2:.2f}",
        )
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("y (\u03bcm)")
        ax.legend()
        fig.suptitle(
            f"Gravity Measurement\nv_y0={vy0:.2e} m/s, 2*a_0={2 * a0:.2e} m/s^2"
        )
        fig.tight_layout()
        plt.show()

    def _finalizeMotComYkSpringFit(self):
        time_sec = np.array(self.tof_values) * 1e-3
        positions_m = np.array(self.true_y_center_list)
        if len(time_sec) == 0 or len(positions_m) == 0:
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return

        self.setConstants()
        m = self.mass

        def com_func(t, z_eq, B, alpha, k_s, phi):
            gamma = alpha / (2 * m)
            omega_d_sq = k_s / m - gamma ** 2
            omega_d = np.sqrt(np.maximum(omega_d_sq, 0))
            return z_eq + B * np.exp(-gamma * t) * np.cos(omega_d * t + phi)

        z_eq_guess = positions_m[-1] if len(positions_m) > 0 else 0.0
        B_guess = positions_m[0] - z_eq_guess
        alpha_guess = 1e-20
        k_s_guess = m * (2 * np.pi * 100) ** 2
        p0 = [z_eq_guess, B_guess, alpha_guess, k_s_guess, 0.0]

        try:
            popt, _ = curve_fit(com_func, time_sec, positions_m, p0=p0, maxfev=10000)
            fitted = com_func(time_sec, *popt)
            ss_res = np.sum((positions_m - fitted) ** 2)
            ss_tot = np.sum((positions_m - np.mean(positions_m)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        except Exception as err:
            print(f"MOT COM_y fit failed: {err}")
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return

        z_eq, B, alpha, k_s, phi = popt
        f_trap = np.sqrt(k_s / m) / (2 * np.pi)

        self.mot_com_y_fit_result = {
            'tof_list': self.tof_values,
            'y_center': self.true_y_center_list,
            'z_eq': z_eq,
            'B': B,
            'alpha': alpha,
            'k_s': k_s,
            'phi': phi,
            'f_trap_Hz': f_trap,
            'r2': r2,
        }

        print("MOT COM_y fit finished")
        if self.tofPlotCheck.GetValue():
            self._plotMotComYkSpringFit(time_sec, positions_m, popt, r2, f_trap)

        self.tofFitStartButton.Enable()
        self.stopTOFFitButton.Disable()
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False
        if self.tofAutoCheck.GetValue():
            self.saveTOFFit(None)

    def _plotMotComYkSpringFit(self, time_sec, positions_m, popt, r2, f_trap):
        time_ms = time_sec * 1e3
        pos_um = positions_m * 1e6

        m = self.mass

        def com_func(t, z_eq, B, alpha, k_s, phi):
            gamma = alpha / (2 * m)
            omega_d_sq = k_s / m - gamma ** 2
            omega_d = np.sqrt(np.maximum(omega_d_sq, 0))
            return z_eq + B * np.exp(-gamma * t) * np.cos(omega_d * t + phi)

        t_fit = np.linspace(np.min(time_sec), np.max(time_sec), 200)

        fig, ax = plt.subplots()
        ax.scatter(time_ms, pos_um, color="tab:blue")
        ax.plot(t_fit * 1e3, com_func(t_fit, *popt) * 1e6, color="tab:red")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("y (\u03bcm)")

        formula = (
            r"$z(t)=z_{eq}+Be^{-\alpha t/(2m)}\cos(\omega_d t+\phi)$" f"\n$R^2$={r2:.2f}"
        )
        ax.legend([], [], title=formula)

        z_eq, B, alpha, k_s, phi = popt
        fig.suptitle(
            "MOT COM_y\n"
            f"\u03b1={alpha:.2e} kg/s\n"
            f"k_s={k_s:.2e} N/m\n"
            f"f_trap={f_trap/1e3:.2f} kHz",
        )
        fig.tight_layout()
        plt.show()

    def _finalizeMotRingdownFit(self):
        time_ms = np.array(self.tof_values)
        widths = np.array(self.true_x_width_list)
        if len(time_ms) == 0 or len(widths) == 0:
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return

        w2 = widths ** 2

        # Initial guess for w_eq as mean of last 30% of data
        tail_start = int(0.7 * len(w2))
        w_eq_guess = np.mean(w2[tail_start:]) if len(w2[tail_start:]) > 0 else np.mean(w2)

        peaks, _ = find_peaks(w2)
        omega_d_guess = 1.0
        tau_b_guess = 1.0
        B_guess = max(w2[0] - w_eq_guess, 0)
        if len(peaks) >= 2:
            T_b = np.mean(np.diff(time_ms[peaks]))
            if T_b > 0:
                omega_d_guess = np.pi / T_b
            amp0 = w2[peaks[0]] - w_eq_guess
            amp_last = w2[peaks[-1]] - w_eq_guess
            if amp0 > 0 and amp_last > 0 and amp_last < amp0:
                tau_b_guess = (time_ms[peaks[-1]] - time_ms[peaks[0]]) / np.log(amp0 / amp_last)
            B_guess = max(amp0, 0)

        def ringdown_func(t, w_eq, B, tau_b, omega_d, phi):
            return w_eq + B * np.exp(-t / tau_b) * np.cos(2 * omega_d * t + phi)

        p0 = [w_eq_guess, B_guess, tau_b_guess, omega_d_guess, 0.0]
        bounds = ([0, 0, 0, 0, -np.pi], [np.inf, np.inf, np.inf, np.inf, np.pi])

        try:
            popt, _ = curve_fit(ringdown_func, time_ms, w2, p0=p0, bounds=bounds, maxfev=10000)
            fitted = ringdown_func(time_ms, *popt)
            ss_res = np.sum((w2 - fitted) ** 2)
            ss_tot = np.sum((w2 - np.mean(w2)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        except Exception as err:
            print(f"MOT ringdown fit failed: {err}")
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return

        w_eq, B, tau_b, omega_d, phi = popt
        f_COM = omega_d / (2 * np.pi)
        f_breath = 2 * omega_d / (2 * np.pi)

        self.mot_ringdown_fit_result = {
            'tof_list': self.tof_values,
            'sigma_x2': w2.tolist(),
            'w_eq': w_eq,
            'B': B,
            'tau_b_ms': tau_b,
            'omega_d_rad_per_ms': omega_d,
            'phi': phi,
            'f_COM_kHz': f_COM,
            'f_breath_kHz': f_breath,
            'r2': r2,
        }

        print("MOT ringdown fit finished")

        if self.tofPlotCheck.GetValue():
            self._plotMotRingdownFit(time_ms, w2, popt, r2, f_COM, f_breath)

        self.tofFitStartButton.Enable()
        self.stopTOFFitButton.Disable()
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False
        if self.tofAutoCheck.GetValue():
            self.saveTOFFit(None)

    def _plotMotRingdownFit(self, time_ms, w2, popt, r2, f_COM, f_breath):
        def ringdown_func(t, w_eq, B, tau_b, omega_d, phi):
            return w_eq + B * np.exp(-t / tau_b) * np.cos(2 * omega_d * t + phi)

        w_eq, B, tau_b, omega_d, phi = popt
        t_fit = np.linspace(np.min(time_ms), np.max(time_ms), 200)

        fig, ax = plt.subplots()
        ax.scatter(time_ms, w2, color="tab:blue")
        ax.plot(t_fit, ringdown_func(t_fit, *popt), color="tab:red")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(r"$\sigma_x^2$")

        formula = (
            r"$\sigma_x^2(t)=\sigma_{eq}^2+Be^{-t/\tau_b}\cos(2\omega_d t+\phi)$"
            f"\n$R^2$={r2:.2f}"
        )
        ax.legend([], [], title=formula)

        fig.suptitle(
            "Mot Rindown\n"
            f"σ_eq^2={w_eq:.2e}\n"
            f"B={B:.2e}\n"
            f"τ_b={tau_b:.2f} ms\n"
            f"ω_d={omega_d:.2f} rad/ms\n"
            f"f_COM={f_COM:.2f} kHz\n"
            f"f_breath={f_breath:.2f} kHz"
        )
        fig.tight_layout()
        plt.show()

    def _finalizeLifetimeFit(self):
        # Use the unit selection from the fitting window if available
        factor = 1e-3
        unit = "ms"
        if getattr(self, "fitWindow", None):
            unit = self.fitWindow.unit_scale
            factor = self.fitWindow.unit_factors["time"][unit]
        time_sec = np.array(self.tof_values) * factor
        atom_numbers = np.array(self.atom_number_list)
        if len(time_sec) == 0 or len(atom_numbers) == 0:
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return

        def exp_decay(t, n0, tau):
            return n0 * np.exp(-t / tau)

        try:
            n0_guess = atom_numbers.max() if len(atom_numbers) > 0 else 1.0
            popt, _ = curve_fit(exp_decay, time_sec, atom_numbers, p0=[n0_guess, 1.0])
            fitted = exp_decay(time_sec, *popt)
            ss_res = np.sum((atom_numbers - fitted) ** 2)
            ss_tot = np.sum((atom_numbers - np.mean(atom_numbers)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        except Exception as err:
            print(f"Lifetime fit failed: {err}")
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return

        n0, tau = popt
        self.lifetime_fit_result = {
            'tof_list': self.tof_values,
            'atom_number': self.atom_number_list,
            'n0': n0,
            'tau': tau,
            'r2': r2,
            'unit': unit,
            'factor': factor,
        }
        print("MOT lifetime fit finished")
        if self.tofPlotCheck.GetValue():
            self._plotLifetimeFit(time_sec, atom_numbers, n0, tau, r2, unit, factor)

        self.tofFitStartButton.Enable()
        self.stopTOFFitButton.Disable()
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False

    def _plotLifetimeFit(self, time_sec, atom_numbers, n0, tau, r2, unit, factor):
        time_disp = time_sec / factor
        tau_disp = tau / factor
        t_fit = np.linspace(np.min(time_sec), np.max(time_sec), 100)
        fig, ax = plt.subplots()
        ax.scatter(time_disp, atom_numbers, color="tab:blue", label="Data")
        ax.plot(
            t_fit / factor,
            n0 * np.exp(-t_fit / tau),
            color="tab:red",
            label=f"N(t)={n0:.2e}e^{{-t/{tau_disp:.1f}}}\n$R^2$={r2:.2f}",
        )
        ax.set_xlabel(f"t ({unit})")
        ax.set_ylabel("Atom Number")
        ax.legend()
        fig.suptitle(f"Mot lifetime\nτ={tau_disp:.2f} {unit}")
        fig.tight_layout()
        plt.show()

    def _finalizeMolassesFit(self):
        time_sec = np.array(self.tof_values) * 1e-3
        x_positions = np.array(self.true_x_center_list)
        y_positions = np.array(self.true_y_center_list)
        if len(time_sec) == 0:
            self.tofFitStartButton.Enable()
            self.stopTOFFitButton.Disable()
            self._tof_fit_running = False
            self._tof_auto_mode = False
            self._tof_auto_paused = False
            return
        self.setConstants()

        def molasses_func(t, r0, v_inf, v0, tau):
            return r0 + v_inf * t + (v0 - v_inf) * tau * (1 - np.exp(-t / tau))

        def fit_axis(positions, axis):
            r0_guess = positions[0]
            v0_guess = 0.0
            if len(time_sec) > 1:
                v0_guess = (positions[1] - positions[0]) / (time_sec[1] - time_sec[0])
            tau_guess = 0.01
            popt, _ = curve_fit(
                molasses_func,
                time_sec,
                positions,
                p0=[r0_guess, 0.0, v0_guess, tau_guess],
                maxfev=10000,
            )
            fitted = molasses_func(time_sec, *popt)
            ss_res = np.sum((positions - fitted) ** 2)
            ss_tot = np.sum((positions - np.mean(positions)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            r0, v_inf, v0, tau = popt
            gamma = self.mass / tau
            g = 9.81 if axis == 'y' else 0.0
            F0 = gamma * v_inf + self.mass * g
            return popt, r2, gamma, v_inf, F0

        results = {}
        for positions, axis in [(x_positions, 'x'), (y_positions, 'y')]:
            if len(positions) == 0:
                continue
            try:
                popt, r2, gamma, v_inf, F0 = fit_axis(positions, axis)
            except Exception as err:
                print(f"Molasses fit failed for {axis}: {err}")
                continue
            results[axis] = {
                'tof_list': self.tof_values,
                f'{axis}_center': self.true_x_center_list if axis == 'x' else self.true_y_center_list,
                f'{axis}0': popt[0],
                'v_inf': v_inf,
                'v0': popt[2],
                'tau': popt[3],
                'gamma': gamma,
                'F0': F0,
                'r2': r2,
            }
            if self.tofPlotCheck.GetValue():
                self._plotMolassesFit(time_sec, positions, popt, r2, gamma, v_inf, F0, axis)

        self.molasses_fit_result = results
        print("Molasses fit finished")

        self.tofFitStartButton.Enable()
        self.stopTOFFitButton.Disable()
        self._tof_fit_running = False
        self._tof_auto_mode = False
        self._tof_auto_paused = False
        if self.tofAutoCheck.GetValue():
            self.saveTOFFit(None)

    def _plotMolassesFit(self, time_sec, positions_m, popt, r2, gamma, v_inf, F0, axis_label):
        time_ms = time_sec * 1e3
        pos_um = positions_m * 1e6

        def molasses_func(t, r0, v_inf, v0, tau):
            return r0 + v_inf * t + (v0 - v_inf) * tau * (1 - np.exp(-t / tau))

        # Generate a smooth time grid for plotting the fitted curve. If only a
        # single time point is present, fall back to that value to avoid
        # creating an empty line.
        if len(time_sec) > 1:
            t_fit = np.linspace(np.min(time_sec), np.max(time_sec), 100)
        else:
            t_fit = time_sec

        fig, ax = plt.subplots()
        ax.scatter(time_ms, pos_um, color="tab:blue")
        ax.plot(
            t_fit * 1e3,
            molasses_func(t_fit, *popt) * 1e6,
            color="tab:red",
        )
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(f"{axis_label} (\u03bcm)")

        formula = (
            r"$r(t)=r_0+v_\infty t+(v_0-v_\infty)\tau(1-e^{-t/\tau})$"
            f"\n$R^2$={r2:.2f}"
        )
        ax.legend([], [], title=formula)
        r0_um = popt[0] * 1e6
        fig.suptitle(
            "Molasses Fit\n"
            f"γ={gamma:.2e} kg/s\n"
            f"v_∞={v_inf:.2e} m/s\n"
            f"F₀={F0:.2e} N\n"
            f"r₀={r0_um:.2f} µm"
        )
        fig.tight_layout()
        plt.show()

    def _plotTOFFit(
        self,
        tof_ms,
        widths_x,
        widths_y,
        slopeX,
        bX,
        rX,
        slopeY,
        bY,
        rY,
        temp_x,
        temp_y,
        temp_x_err,
        temp_y_err,
        psd,
        psd_err,
        density,
        density_err,
        avg_atom_number,
    ):
        """Display scatter plots of $\sigma^2$ versus $t^2$ for both axes."""
        tof_s = np.array(tof_ms) * 1e-3
        t_sq = np.square(tof_s)
        # Convert units for plotting only
        t_sq_ms = t_sq * 1e6

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(
            f"Temp: ({temp_x:.1f} ± {temp_x_err:.1f}, {temp_y:.1f} ± {temp_y_err:.1f}) μk\nPSD: {psd:.2e} ± {psd_err:.2e} | Density: {density:.2e} ± {density_err:.2e} cm^-3 | Avg N: {avg_atom_number / 1e6:.1f}M"
        )

        wx_sq = np.square(widths_x)
        wx_sq_um = wx_sq * 1e12
        axes[0].scatter(t_sq_ms, wx_sq_um, color="tab:blue")
        x_range = np.linspace(np.min(t_sq), np.max(t_sq), 100)
        x_range_ms = x_range * 1e6
        axes[0].plot(
            x_range_ms,
            (slopeX * 1e6) * x_range_ms + bX * 1e12,
            color="tab:red",
            label=f"$\sigma_x^2={(slopeX * 1e6):.2e}t^2+{(bX * 1e12):.2e}$\n$R^2$={rX**2:.2f}",
        )
        axes[0].set_xlabel("$t^2$ (ms$^2$)")
        axes[0].set_ylabel("$\sigma_x^2$ (\u03bcm$^2$)")
        axes[0].legend()

        wy_sq = np.square(widths_y)
        wy_sq_um = wy_sq * 1e12
        axes[1].scatter(t_sq_ms, wy_sq_um, color="tab:blue")
        y_range = np.linspace(np.min(t_sq), np.max(t_sq), 100)
        y_range_ms = y_range * 1e6
        axes[1].plot(
            y_range_ms,
            (slopeY * 1e6) * y_range_ms + bY * 1e12,
            color="tab:red",
            label=f"$\sigma_y^2={(slopeY * 1e6):.2e}t^2+{(bY * 1e12):.2e}$\n$R^2$={rY**2:.2f}",
        )
        axes[1].set_xlabel("$t^2$ (ms$^2$)")
        axes[1].set_ylabel("$\sigma_y^2$ (\u03bcm$^2$)")
        axes[1].legend()

        fig.tight_layout()
        plt.show()

    def setSnippetPath(self, e):
        snippetPath = e.GetEventObject()
        self.snippetPath = snippetPath.GetValue()
        print(self.snippetPath)

    def chooseSnippetPath(self, e):
        dialog = wx.DirDialog(
            None,
            "Choose a directory:",
            defaultPath=self.snippetPath,
            style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON,
        )
        if dialog.ShowModal() == wx.ID_OK:
            self.snippetPath = dialog.GetPath()
            self.snippetTextBox.SetValue(self.snippetPath)
        dialog.Destroy()

    def setTOFCsvPath(self, e):
        pathCtrl = e.GetEventObject()
        self.tofCsvPath = pathCtrl.GetValue()

    def setTOFOffset(self, e):
        ctrl = e.GetEventObject()
        try:
            self.tofOffset = float(ctrl.GetValue())
        except Exception:
            self.tofOffset = 0.0

    def setXTrapFreq(self, e):
        omega = e.GetEventObject()
        self.xTrapFreq = float(omega.GetValue())
        self.updateTemp()

    def setYTrapFreq(self, e):
        omega = e.GetEventObject()
        self.yTrapFreq = float(omega.GetValue())
        self.updateTemp()

    def checkIfFileSizeChanged(self):
        previousFileSize = self.actualFileSize
        self.actualFileSize = os.stat(self.filename).st_size
        hasFileSizeChanged = False
        if self.actualFileSize != previousFileSize:
            hasFileSizeChanged = True

        return hasFileSizeChanged

    def setTOF(self, e):
        tof = e.GetEventObject()
        self.TOF = float(tof.GetValue())
        self.updateTemp()

    def updateTrueWidths(self):
        self.true_x_width = self.x_width * self.pixelToDistance
        self.true_y_width = self.y_width * self.pixelToDistance

        self.true_x_width_std = self.x_width_std * self.pixelToDistance
        self.true_y_width_std = self.y_width_std * self.pixelToDistance

        self.true_x_center = self.x_center * self.pixelToDistance
        self.true_y_center = self.y_center * self.pixelToDistance

        self.true_x_center_std = self.x_center_std * self.pixelToDistance
        self.true_y_center_std = self.y_center_std * self.pixelToDistance

        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(" The true X width = " + str("%.3f"%(self.true_x_width*1E6)) + " um // " + str("%.3f"%(self.true_x_width_std*1E6)) + " um")
        print(" The true Y width = " + str("%.3f"%(self.true_y_width*1E6)) + " um // " + str("%.3f"%(self.true_y_width_std*1E6)) + " um")
        print(" The true X center = " + str("%.3f"%(self.true_x_center*1E6)) + " um // " + str("%.3f"%(self.true_x_center_std*1E6)) + " um")
        print(" The true Y center = " + str("%.3f"%(self.true_y_center*1E6)) + " um // " + str("%.3f"%(self.true_y_center_std*1E6)) + " um")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("")

        self.xRadiusBox.SetValue(str("%.1f"%(self.true_x_width*1E6)) + " +/- " + str("%.1f"%(self.true_x_width_std*1E6)))
        self.yRadiusBox.SetValue(str("%.1f"%(self.true_y_width*1E6)) + " +/- " + str("%.1f"%(self.true_y_width_std*1E6)))
        self.xCenterBox.SetValue(str("%.1f"%(self.x_center)) + " +/- " + str("%.1f"%(self.x_center_std)))
        self.yCenterBox.SetValue(str("%.1f"%(self.y_center)) + " +/- " + str("%.1f"%(self.y_center_std)))

    def updateTemp(self):
        if self.isXFitSuccessful is False:
            self.true_x_width = 0
        if self.isYFitSuccessful is False:
            self.true_y_width = 0

        self.temperature[0] = 1E+6 * self.mass * (self.true_x_width * 2 * np.pi * self.xTrapFreq) ** 2 / (
            kB * (1 + (2 * np.pi * self.xTrapFreq * self.TOF * 1E-3) ** 2)
        )
        self.temperature[1] = 1E+6 * self.mass * (self.true_y_width * 2 * np.pi * self.yTrapFreq) ** 2 / (
            kB * (1 + (2 * np.pi * self.yTrapFreq * self.TOF * 1E-3) ** 2)
        )

        temp = "(" + str("%.3f"%(self.temperature[0])) +", " + str("%.3f"%(self.temperature[1])) + ")"
        self.tempBox.SetValue(temp)

        self.tempLongTime[0] = 1E+6 * self.mass * (self.true_x_width * 1E+3 / self.TOF) ** 2 / kB
        self.tempLongTime[1] = 1E+6 * self.mass * (self.true_y_width * 1E+3 / self.TOF) ** 2 / kB

        temp2 = "(" + str("%.3f"%(self.tempLongTime[0])) +", " + str("%.3f"%(self.tempLongTime[1])) + ")"
        self.tempBox2.SetValue(temp2)

    def updateFittingResults(self):
        self.updateTrueWidths()
        self._refresh_fit_quality()
        self.updateTemp()

        if self.fitMethodBoson.GetValue() is True:
            self.updateBosonParams()
            print(" ~~~~ BEC population ratio__x: " + str(self.x_becPopulationRatio))
            print(" ~~~~ BEC population ratio__y: " + str(self.y_becPopulationRatio))
        elif self.fitMethodFermion.GetValue() is True:
            print(" ~~~~ T//T_F__x: " + str(self.x_TOverTF))
            print(" ~~~~ T//T_F__y: " + str(self.y_TOverTF))
            self.updateFermionParams()

    def updateTc(self):
        self.TcBox.SetValue(str("%.2f"%(self.x_tOverTc)) + ", " + str("%.2f"%(self.x_becPopulationRatio)))

    def updateTFRadius(self):
        self.TFRadiusBox.SetValue(str("%.2f"%(self.x_thomasFermiRadius * 1e6)))

    def updateBosonParams(self):
        self.updateTc()
        self.updateTFRadius()

    def updateFermionParams(self):
        print(" --------- DO NOTHING YET ----------")

    def _select_absorption_colormap(self, cmap_name):
        if not getattr(self, "colormapChoice", None):
            return
        if not getattr(self, "_colormap_options", None):
            self._colormap_options = ABSORPTION_COLORMAP_OPTIONS
        cmap_name = (cmap_name or "").strip().lower()
        fallback_idx = next(
            (idx for idx, (_, cmap) in enumerate(self._colormap_options) if cmap == "gray_r"),
            0,
        )
        for idx, (_, cmap) in enumerate(self._colormap_options):
            if cmap.lower() == cmap_name:
                self.colormapChoice.SetSelection(idx)
                break
        else:
            self.colormapChoice.SetSelection(fallback_idx)

    def _get_selected_colormap(self):
        if not getattr(self, "colormapChoice", None):
            return "gray_r"
        idx = self.colormapChoice.GetSelection()
        if idx == wx.NOT_FOUND:
            return "gray_r"
        try:
            return self._colormap_options[idx][1]
        except (AttributeError, IndexError):
            return "gray_r"

    def _absorption_colormap_with_blank_mask(self, cmap_name):
        """Return a copy of the chosen colormap with masked pixels hidden."""

        base = _safe_get_cmap(cmap_name, fallback='gray_r')

        try:
            cmap = base.copy()
        except AttributeError:
            # Older Matplotlib versions do not expose ``copy``.
            if isinstance(base, mcolors.ListedColormap):
                cmap = mcolors.ListedColormap(base.colors, name=base.name)
            else:
                sample = base(np.linspace(0, 1, 256))
                cmap = mcolors.LinearSegmentedColormap.from_list(base.name, sample)

        with contextlib.suppress(Exception):
            cmap.set_bad(color=(1.0, 1.0, 1.0), alpha=1.0)

        return cmap

    def _get_initial_display_params(self):
        cmap = 'gray_r'
        clim = None
        if getattr(self, "chosenLayerNumber", None) == 4:
            cmap_name = self._get_selected_colormap()
            cmap = self._absorption_colormap_with_blank_mask(cmap_name)
            if cmap_name == 'gray_r':
                clim = (-1, 1)
            else:
                clim = (0, 5)
        return cmap, clim

    def _apply_display_settings(self):
        if not getattr(self, "currentImg", None):
            return

        if getattr(self, "chosenLayerNumber", None) == 4:
            cmap_name = self._get_selected_colormap()
            cmap = self._absorption_colormap_with_blank_mask(cmap_name)
            self.currentImg.set_cmap(cmap)
            if cmap_name == 'gray_r':
                self.currentImg.set_clim(vmin=-1, vmax=1)
            else:
                self.currentImg.set_clim(vmin=0, vmax=5)
        else:
            self.currentImg.set_cmap('gray_r')
            self.currentImg.autoscale()

    def _compute_r_squared(self, data, fit):
        if data is None or fit is None:
            return None

        data = np.asarray(data, dtype=float)
        fit = np.asarray(fit, dtype=float)
        if data.size == 0 or fit.size == 0:
            return None

        data = np.ravel(data)
        fit = np.ravel(fit)
        if data.size != fit.size:
            min_len = min(data.size, fit.size)
            if min_len < 2:
                return None
            data = data[:min_len]
            fit = fit[:min_len]

        mask = np.isfinite(data) & np.isfinite(fit)
        if np.count_nonzero(mask) < 2:
            return None

        masked_data = data[mask]
        masked_fit = fit[mask]
        residuals = masked_data - masked_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((masked_data - np.mean(masked_data)) ** 2)
        if ss_tot == 0:
            return None

        return 1.0 - ss_res / ss_tot

    def _format_r_squared(self, value):
        if value is None:
            return "N/A"
        return f"{value:.3f}"

    def _refresh_fit_quality(self):
        self.x_r_squared = self._compute_r_squared(self.x_summed, self.x_fitted)
        self.y_r_squared = self._compute_r_squared(self.y_summed, self.y_fitted)

        if hasattr(self, "xRSquaredBox"):
            self.xRSquaredBox.SetValue(self._format_r_squared(self.x_r_squared))
        if hasattr(self, "yRSquaredBox"):
            self.yRSquaredBox.SetValue(self._format_r_squared(self.y_r_squared))

    def _update_degeneracy_visibility(self):
        show_rows = not (
            getattr(self, "fitMethodGaussian", None)
            and self.fitMethodGaussian.GetValue()
        )

        rows = getattr(self, "_degeneracy_rows", [])
        parent_sizer = getattr(self, "_degeneracy_parent", None)

        for row in rows:
            if parent_sizer is not None:
                try:
                    parent_sizer.Show(row, show_rows)
                except Exception:
                    pass
            if hasattr(row, "ShowItems"):
                try:
                    row.ShowItems(show_rows)
                except Exception:
                    pass

        for ctrl in getattr(self, "_degeneracy_controls", []):
            if ctrl is not None:
                try:
                    ctrl.Show(show_rows)
                except Exception:
                    continue

        if parent_sizer is not None:
            parent_sizer.Layout()

        if getattr(self, "panel", None):
            self.panel.Layout()
            fit_inside = getattr(self.panel, "FitInside", None)
            if callable(fit_inside):
                fit_inside()
            send_size = getattr(self.panel, "SendSizeEvent", None)
            if callable(send_size):
                send_size()

        frame_send_size = getattr(self, "SendSizeEvent", None)
        if callable(frame_send_size):
            frame_send_size()

        self.Layout()

    def setCurrentImg(self, data, hasFileSizeChanged):
        if data is None:
            return

        if isinstance(data, np.ndarray):
            base_array = np.asarray(data)
            existing_mask = np.ma.getmaskarray(data) if isinstance(data, np.ma.MaskedArray) else None
            mask = ~np.isfinite(base_array)
            if existing_mask is not None:
                mask = mask | existing_mask
            if np.any(mask):
                display_data = np.ma.masked_array(base_array, mask=mask)
            else:
                display_data = base_array
        else:
            display_data = data

        if hasFileSizeChanged or self.currentImg is None:
            cmap, clim = self._get_initial_display_params()
            imshow_kwargs = dict(aspect='auto', cmap=cmap)
            if clim:
                imshow_kwargs['vmin'], imshow_kwargs['vmax'] = clim
            self.currentImg = self.axes1.imshow(display_data, **imshow_kwargs)
        else:
            self.currentImg.set_data(display_data)

    def _read_float_ctrl(self, attr_name):
        ctrl = getattr(self, attr_name, None)
        if ctrl is None:
            return None

        try:
            value = float(ctrl.GetValue())
        except (AttributeError, ValueError, TypeError):
            return None

        if not np.isfinite(value):
            return None

        return value

    def _get_od_bounds(self):
        min_od = self._read_float_ctrl("minODCtrl")
        max_od = self._read_float_ctrl("maxODCtrl")

        if max_od is not None and max_od <= 0:
            max_od = None

        return min_od, max_od

    def _od_limits_active(self):
        limit_ctrl = getattr(self, "checkLimitOD", None)
        if not limit_ctrl or not limit_ctrl.GetValue():
            return False
        min_od, max_od = self._get_current_od_bounds()
        return min_od is not None or max_od is not None

    def _get_current_od_bounds(self):
        return self._get_od_bounds()

    def _limit_image(self, data):
        if data is None:
            return None, None, None

        arr = np.ma.array(data, copy=False)
        values = np.array(np.ma.getdata(arr), dtype=float, copy=True)
        mask = np.ma.getmaskarray(arr).copy()

        mask |= ~np.isfinite(values)

        min_od, max_od = self._get_current_od_bounds()
        limit_mask = np.zeros_like(values, dtype=bool)

        if max_od is not None:
            limit_mask |= values > max_od
        if min_od is not None:
            limit_mask |= values < min_od

        if np.any(limit_mask):
            mask |= limit_mask
            od_mask = limit_mask
        else:
            limit_mask = None
            od_mask = None

        if not np.any(mask):
            return values, None, None

        if od_mask is not None and not np.any(od_mask):
            od_mask = None

        return values, mask, od_mask

    def _invalidate_limited_atom_cache(self):
        self._limited_atom_image = None
        self._limited_atom_mask = None
        self._limited_atom_od_mask = None

    def _compute_limited_atom_image(self):
        atom_img = getattr(self, "atomImage", None)
        if atom_img is None:
            self._invalidate_limited_atom_cache()
            return None, None, None

        if not self._od_limits_active():
            self._invalidate_limited_atom_cache()
            return atom_img, None, None

        if self._limited_atom_image is not None:
            return self._limited_atom_image, self._limited_atom_mask, self._limited_atom_od_mask

        values, mask, od_mask = self._limit_image(atom_img)
        if values is None:
            self._invalidate_limited_atom_cache()
            return None, None, None

        if mask is None:
            limited = np.array(values, dtype=float, copy=True)
        else:
            limited = self._limited_copy_with_nans(values, mask)

        self._limited_atom_image = limited
        self._limited_atom_mask = mask
        self._limited_atom_od_mask = od_mask
        return limited, mask, od_mask

    def _get_active_atom_image(self):
        limited, mask, _ = self._compute_limited_atom_image()
        if limited is None:
            return None

        if mask is None and not self._od_limits_active():
            return limited

        return limited

    def _limited_copy_with_nans(self, data, mask):
        if data is None:
            return None
        if mask is None:
            return np.array(data, dtype=float, copy=True)
        limited = np.array(data, dtype=float, copy=True)
        limited[mask] = np.nan
        return limited

    def _layout_pixel_value_section(self):
        ctrl = getattr(self, "odMaskInfo", None)
        if ctrl is None:
            return

        parent = ctrl.GetParent()
        if parent is not None:
            parent.Layout()

        panel = getattr(self, "panel", None)
        if panel is not None:
            panel.Layout()

    def _set_od_mask_summary(self, od_mask, total_pixels):
        ctrl = getattr(self, "odMaskInfo", None)
        if ctrl is None:
            return

        if not self._od_limits_active() or not total_pixels:
            ctrl.SetLabel("")
            ctrl.Hide()
            self._layout_pixel_value_section()
            return

        masked_pixels = int(np.count_nonzero(od_mask)) if od_mask is not None else 0
        percentage = (masked_pixels / float(total_pixels)) * 100.0
        ctrl.SetLabel(f"OD-masked pixels: {percentage:.1f}%")
        ctrl.Show()
        self._layout_pixel_value_section()

    def _get_active_aoi_image(self):
        limited = getattr(self, "_limited_aoi_image", None)
        if limited is not None:
            return limited
        if self.AOIImage is None:
            return None
        return np.array(self.AOIImage, dtype=float, copy=True)

    def _update_od_metrics(self, working_aoi):
        if working_aoi is None:
            self.od = [float("nan"), float("nan")]
            return

        finite_vals = working_aoi[np.isfinite(working_aoi)]
        if finite_vals.size == 0:
            self.od = [float("nan"), float("nan")]
            return

        finite_sorted = np.sort(finite_vals)
        try:
            threshold = float(-np.log(minT))
            match_indices = np.where(finite_sorted == threshold)[0]
            if match_indices.size and match_indices[0] > 0:
                idx = match_indices[0] - 1
                max_od = finite_sorted[idx]
            else:
                max_od = finite_sorted[-1]
        except Exception:
            max_od = finite_sorted[-1]

        avg_od = float(np.mean(finite_vals))
        self.od = [max_od, avg_od]

    def _recompute_aoi_statistics(self):
        if getattr(self, "atomImage", None) is None:
            self._limited_aoi_image = None
            return

        try:
            aoi = self.atomImage[self.yTop - 3 : self.yBottom + 4, self.xLeft - 3 : self.xRight + 4]
        except Exception:
            return

        self.AOIImage = aoi

        if self._od_limits_active():
            values, mask, _ = self._limit_image(aoi)
        else:
            values = mask = None
        if mask is not None:
            self._limited_aoi_image = self._limited_copy_with_nans(values, mask)
            working_aoi = self._limited_aoi_image
        elif values is not None:
            self._limited_aoi_image = None
            working_aoi = values
        else:
            self._limited_aoi_image = None
            working_aoi = np.array(aoi, dtype=float, copy=True)

        self.offsetEdge = aoiEdge(working_aoi, self.leftRightEdge.GetValue(), self.updownEdge.GetValue())
        self.rawAtomNumber = atomNumber(working_aoi, self.offsetEdge)
        self._update_od_metrics(working_aoi)
        self.setAtomNumber()

    def _get_atom_image_for_display(self):
        limited, mask, od_mask = self._compute_limited_atom_image()

        if limited is None:
            self._set_od_mask_summary(None, None)
            return None

        if not self._od_limits_active():
            self._set_od_mask_summary(None, None)
            return limited

        total_pixels = limited.size if limited is not None else None
        self._set_od_mask_summary(od_mask, total_pixels)

        if mask is None:
            return np.array(limited, dtype=float, copy=True)

        display = np.ma.array(limited, mask=mask, copy=False)
        with contextlib.suppress(Exception):
            display.set_fill_value(np.nan)
        return display

    def _refresh_absorption_display(self):
        if self.chosenLayerNumber != 4:
            self._set_od_mask_summary(None, None)
            return

        image_to_show = self._get_atom_image_for_display()
        if image_to_show is None:
            self._set_od_mask_summary(None, None)
            return

        self.setCurrentImg(image_to_show, hasFileSizeChanged=False)
        self._apply_display_settings()
        self._update_image_title()
        self.canvas1.draw()

    def checkTimeChange(self):
        current = datetime.date.today()

        if (current != self.today):
            self.timeChanged = True
            self.today = current
        else:
            self.timeChanged = False

    def setImageAngle(self, e):
        tx = e.GetEventObject()
        rotation = tx.GetValue()

        self.imageAngle = float(rotation)

    def setImagePivotX(self, e):
        tx = e.GetEventObject()
        temp = int(tx.GetValue())

        x = self.atomImage.shape[0]
        if (x < temp) or (temp <= 0):
            temp = x//2

        self.imagePivotX = temp
        self.pivotXBox.SetValue(str(self.imagePivotX))


    def setImagePivotY(self, e):
        tx = e.GetEventObject()
        temp = int(tx.GetValue())

        y = self.atomImage.shape[1]
        if (y < temp) or (temp <= 0):
            temp = y//2

        self.imagePivotY = temp
        self.pivotYBox.SetValue(str(self.imagePivotY))

    def setImageRotationParams(self, e):
        if self.imageAngle == 0.  and self.imageAngle == self.prevImageAngle:
            self.isRotationNeeded = False
        else:
            self.isRotationNeeded = True

        self.setDataAndUpdate()

    def setPixelSize(self, e):
        tx = e.GetEventObject()
        self.pixelSize = float(tx.GetValue())

        self.pixelToDistance = self.pixelSize / self.magnification * 10**-6
        self.setAtomNumber()
        self.updateFittingResults()

        print("PIXEL SIZE:")
        print(self.pixelSize)

    def setMagnification(self, e):
        mg = e.GetEventObject()
        self.magnification = float(mg.GetValue())

        self.pixelToDistance = self.pixelSize / self.magnification * 10**-6

        self.setAtomNumber()
        self.updateFittingResults()

        print("MAGNIFICATION:")
        print(self.magnification )

    def setDetuning(self, event):
        ctrl = event.GetEventObject()
        value = ctrl.GetValue().strip()
        try:
            detuning = float(value)
        except ValueError:
            ctrl.SetValue(f"{self.detuning_mhz:g}")
        else:
            if detuning != self.detuning_mhz:
                self.detuning_mhz = detuning
                self.setAtomNumber()
            ctrl.SetValue(f"{self.detuning_mhz:g}")

        if isinstance(event, wx.FocusEvent):
            event.Skip()

    def setConstants(self):
        self.pixelToDistance = self.pixelSize / self.magnification * 10**-6

        massUnit = 1.66053906892E-27
        wavelength = 1E-9
        if (self.atom == 'Cs'):
            wavelength = 852E-9
            self.mass = 132.905 * massUnit
            print("------------- Cs chosen ------------")
        elif (self.atom == 'Li'):
            wavelength = 671E-9
            self.mass = 6 * massUnit
            print("------------- Li chosen ------------")
        elif (self.atom == 'Ag'):
            wavelength = 328.1E-9
            self.mass = 107.8682 * massUnit
            print("------------- Ag chosen ------------")

        linewidths_mhz = {
            'Cs': 5.2,
            'Li': 5.9,
            'Ag': 23.4
        }

        gamma_mhz = linewidths_mhz.get(self.atom, 5.2)
        detuning = getattr(self, "detuning_mhz", 0.0)
        denominator = 1.0
        if gamma_mhz:
            denominator += (2.0 * detuning / gamma_mhz) ** 2

        base_cross_section = 6.0 * np.pi * np.divide(wavelength, (2 * np.pi)) ** 2
        self.crossSection = base_cross_section / denominator


    def onAtomRadioClicked(self,e):
        self.atom = self.atomRadioBox.GetStringSelection()
        print(self.atom)
        self.setAtomNumber()
        self.updateFittingResults()

        self.snippetTextBox.SetValue(self.snippetPath)

        print("new snippet path -----> " + self.snippetPath)

    def updateFileList(self):
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except Exception as err:
                print(f"Failed to create directory {self.path}: {err}")
                
        os.chdir(self.path)
        start = time.time()
        self.fileList =  sorted(glob.iglob(self.path + '*.' + self.fileType), key=os.path.getctime)
        end = time.time()

    def updateLatestFileName(self):
        self.updateFileList()
        if self.fileList:
            self.filename = self.fileList[-1]  # this is the filename
            self.highlight_image_in_list(self.filename)
        else:
            self.filename = None

    def isAOIoutside(self):
        flag = False
        shape = self.atomImage.shape

        if int(self.AOI1.GetValue()) >= shape[1] or int(self.AOI1.GetValue()) < 0:
            self.xLeft = 3
            flag = True
        else:
            self.xLeft = int(self.AOI1.GetValue())

        if int(self.AOI2.GetValue()) >= shape[0] or int(self.AOI2.GetValue()) < 0:
            self.yTop = 3
            flag = True
        else:
            self.yTop = int(self.AOI2.GetValue())

        if int(self.AOI3.GetValue()) >= shape[1] or int(self.AOI3.GetValue()) < 0:
            self.xRight = shape[1] - 4
            flag = True
        else:
            self.xRight = int(self.AOI3.GetValue())

        if int(self.AOI4.GetValue()) >= shape[0] or int(self.AOI4.GetValue()) < 0:
            self.yBottom = shape[0] - 4
            flag = True
        else:
            self.yBottom = int(self.AOI4.GetValue())

        return flag

    def initializeAOI(self):

        if self.isAOIoutside() is True:
            print("")
            print("#################################################")
            print("AOI initializing....")
            print("#################################################")
            print("")
            self.AOI = [[self.xLeft,self.yTop],[self.xRight,self.yBottom]]

            self.AOI1.SetValue(str(self.xLeft))
            self.AOI2.SetValue(str(self.yTop))
            self.AOI3.SetValue(str(self.xRight))
            self.AOI4.SetValue(str(self.yBottom))

            rect = self._ensure_aoi_patch()
            if rect is not None:
                rect.set_width(self.xRight - self.xLeft)
                rect.set_height(self.yBottom - self.yTop)
                rect.set_xy((self.xLeft, self.yTop))
                self.canvas1.draw()

            self.setAtomNumber()

    def applyFilter(self):
        """Compatibility stub for the retired median filter option."""
        # The median filter has been removed from the image-processing pipeline.
        # Maintain the method to avoid attribute errors if legacy bindings fire.
        self.isMedianFilterOn = False
        self.setDataAndUpdate()

    def onFlipImage(self, event):
        if self.filename:
            self.setDataAndUpdate()
        event.Skip()

    def onColormapChanged(self, event):
        if getattr(self, "currentImg", None):
            self._apply_display_settings()
            self.canvas1.draw()
        if isinstance(self._settings, dict):
            cmap = self._get_selected_colormap()
            self._settings["absorption_colormap"] = cmap
            self._settings["absorption_use_jet"] = cmap == "jet"
        try:
            self._save_settings()
        except Exception as err:
            print(f"Failed to save settings: {err}")
        event.Skip()

    def onLimitODToggle(self, event):
        """Handle toggling of the OD limit option."""

        self._update_od_limit_visibility()
        self.update1DProfilesAndFit(event)

    def update1DProfilesAndFit(self, i=0, recompute_aoi=True):
        self._update_degeneracy_visibility()
        self._invalidate_limited_atom_cache()
        min_od = max_od = None
        if self.checkLimitOD.GetValue():
            min_od, max_od = self._get_od_bounds()

        if recompute_aoi:
            self._recompute_aoi_statistics()

        try:
            self.calc1DProfiles(min_od=min_od, max_od=max_od)
            self.calc1DRadialAvgAndRefit()
            self.update1DProfiles()
            self.updateFittingResults()
        finally:
            self._refresh_absorption_display()

            if getattr(self, "_suppress_fit_sync", False):
                return

            event_source = None
            if hasattr(i, "GetEventObject"):
                try:
                    event_source = i.GetEventObject()
                except Exception:
                    event_source = None

            if event_source in (
                getattr(self, "fitMethodGaussian", None),
                getattr(self, "fitMethodFermion", None),
                getattr(self, "fitMethodBoson", None),
            ) or event_source is None:
                self._sync_preferred_fit_method_from_ui()

    def fit(self, axis = 'xy'):
        try:
            self.doGaussianFit(axis)
            self.atomNumFromFitX_std = getattr(self, "atomNumFromGaussianX_std", 0.0)
            self.atomNumFromFitY_std = getattr(self, "atomNumFromGaussianY_std", 0.0)
            print("x center is " + str(round(self.x_center, 1)))
            print("x width is " + str(round(self.x_width, 2)))
            print("")
            print("y center is " + str(round(self.y_center, 2)))
            print("y width is " + str(round(self.y_width, 2)))

            if self.fitMethodFermion.GetValue() is True:
                print("")
                print(" ------------- Fermion fit......... -----------")
                print("")

                isXon = True
                isYon = False
                if isXon is True:
                    self.degenFitter.setInitialCenter(self.x_center)
                    self.degenFitter.setInitialWidth(self.x_width)
                    self.degenFitter.setInitialPeakHeight(self.x_peakHeight)
                    self.degenFitter.setInitialOffset(self.x_offset)
                    self.degenFitter.setInitialSlope(self.x_slope)

                    self.degenFitter.setTOF(self.TOF)

                    self.degenFitter.setData(self.x_basis, self.x_summed)
                    self.degenFitter.doDegenerateFit(False)
                    self.x_fitted = self.degenFitter.getFittedProfile()

                    self.x_TOverTF = self.degenFitter.getTOverTF()
                    self.x_fermiRadius = self.degenFitter.getFermiRadius()

                if isYon is True:
                    self.degenFitter.setInitialCenter(self.y_center)
                    self.degenFitter.setInitialWidth(self.y_width)
                    self.degenFitter.setInitialPeakHeight(self.y_peakHeight)
                    self.degenFitter.setInitialOffset(self.y_offset)
                    self.degenFitter.setInitialSlope(self.y_slope)

                    self.degenFitter.setTOF(self.TOF)

                    self.degenFitter.setData(self.y_basis, self.y_summed)
                    self.degenFitter.doDegenerateFit( False)
                    self.y_fitted = self.degenFitter.getFittedProfile()

                    self.y_TOverTF = self.degenFitter.getTOverTF()
                    self.y_fermiRadius = self.degenFitter.getFermiRadius()

            elif self.fitMethodBoson.GetValue() is True:
                print("")
                print(" ------------- Boson fit......... -----------")
                print("")

                isXon = False
                isYon = True
                if isXon is True:
                    self.degenFitter.setInitialCenter(self.x_center)
                    self.degenFitter.setInitialWidth(self.x_width)
                    self.degenFitter.setInitialPeakHeight(self.x_peakHeight)
                    self.degenFitter.setInitialOffset(self.x_offset)
                    self.degenFitter.setInitialSlope(self.x_slope)

                    self.degenFitter.setData(self.x_basis, self.x_summed)
                    self.degenFitter.doDegenerateFit()
                    self.x_fitted = self.degenFitter.getFittedProfile()
                    self.x_tOverTc = self.degenFitter.getTOverTc()
                    self.x_thomasFermiRadius = self.degenFitter.getThomasFermiRadius() * self.pixelToDistance
                    self.x_becPopulationRatio = self.degenFitter.getBecPopulationRatio()
                    self.atomNumFromDegenFitX = (
                        self.degenFitter.getTotalPopulation()
                        * (self.pixelToDistance ** 2)
                        / self.crossSection
                    )
                    self.atomNumFromFitX = self.atomNumFromDegenFitX
                    self.atomNumFromFitX_std = getattr(
                        self, "atomNumFromGaussianX_std", 0.0
                    )

                    print("x_width -------" + str(self.x_width))
                    self.x_width = self.degenFitter.getThermalWidth()
                    print("x_width -------" + str(self.x_width))
                if isYon is True:
                    self.degenFitter.setInitialCenter(self.y_center)
                    self.degenFitter.setInitialWidth(self.y_width)
                    self.degenFitter.setInitialPeakHeight(self.y_peakHeight)
                    self.degenFitter.setInitialOffset(self.y_offset)
                    self.degenFitter.setInitialSlope(self.y_slope)

                    self.degenFitter.setData(self.y_basis, self.y_summed)
                    self.degenFitter.doDegenerateFit()
                    self.y_fitted = self.degenFitter.getFittedProfile()
                    self.y_tOverTc = self.degenFitter.getTOverTc()
                    self.y_thomasFermiRadius = self.degenFitter.getThomasFermiRadius() * self.pixelToDistance
                    self.y_becPopulationRatio = self.degenFitter.getBecPopulationRatio()
                    self.atomNumFromDegenFitY = (
                        self.degenFitter.getTotalPopulation()
                        * (self.pixelToDistance ** 2)
                        / self.crossSection
                    )
                    self.atomNumFromFitY = self.atomNumFromDegenFitY
                    self.atomNumFromFitY_std = getattr(
                        self, "atomNumFromGaussianY_std", 0.0
                    )

                    self.y_width = self.degenFitter.getThermalWidth()


            else:
                self.atomNumFromFitX = self.atomNumFromGaussianX
                self.atomNumFromFitY = self.atomNumFromGaussianY
                self.atomNumFromFitX_std = getattr(
                    self, "atomNumFromGaussianX_std", 0.0
                )
                self.atomNumFromFitY_std = getattr(
                    self, "atomNumFromGaussianY_std", 0.0
                )
        except Exception as err:
            print("------ Fitting Failed -------")
        finally:
            self._refresh_fit_quality()

    def doGaussianFit(self, axis = 'xy'):
        max_od = None
        if self.checkLimitOD.GetValue():
            _, max_od = self._get_od_bounds()

        tail_fraction = self._get_tail_fraction()
        tail_sides = self._get_tail_side_keywords()
        tail_side_x = tail_sides.get("x")
        tail_side_y = tail_sides.get("y")

        atom_scale = np.sqrt(2 * np.pi) * (self.pixelToDistance ** 2) / self.crossSection

        if axis == 'xy':
            self.x_center, self.x_width, self.x_offset, self.x_peakHeight, self.x_fitted, self.isXFitSuccessful, self.x_slope, err_x = gaussianFit(
                self.x_basis,
                self.x_summed,
                self.AOI,
                axis='x',
                max_od=max_od,
                fit_tails_only=self.checkTailFit.GetValue(),
                tail_fraction=tail_fraction,
                tail_side=tail_side_x,
            )
            self.atomNumFromGaussianX = (
                self.x_peakHeight
                * np.sqrt(2 * np.pi)
                * self.x_width
                * (self.pixelToDistance ** 2)
                / self.crossSection
            )
            err_x_arr = (
                np.asarray(err_x, dtype=float) if err_x is not None else np.zeros(5)
            )
            amp_err_x = err_x_arr[0] if err_x_arr.size > 0 else 0.0
            sigma_err_x = err_x_arr[2] if err_x_arr.size > 2 else 0.0
            self.atomNumFromGaussianX_std = self._gaussian_atom_number_uncertainty(
                self.x_peakHeight,
                self.x_width,
                amp_err_x,
                sigma_err_x,
                atom_scale,
            )
            self.x_width_std = err_x[2]
            self.x_center_std = err_x[1]
            print("")
            print("x fit err = " +str(err_x))
            self.y_center, self.y_width, self.y_offset, self.y_peakHeight, self.y_fitted, self.isYFitSuccessful, self.y_slope , err_y= gaussianFit(
                self.y_basis,
                self.y_summed,
                self.AOI,
                axis='y',
                max_od=max_od,
                fit_tails_only=self.checkTailFit.GetValue(),
                tail_fraction=tail_fraction,
                tail_side=tail_side_y,
            )
            self.atomNumFromGaussianY = (
                self.y_peakHeight
                * np.sqrt(2 * np.pi)
                * self.y_width
                * (self.pixelToDistance ** 2)
                / self.crossSection
            )
            err_y_arr = (
                np.asarray(err_y, dtype=float) if err_y is not None else np.zeros(5)
            )
            amp_err_y = err_y_arr[0] if err_y_arr.size > 0 else 0.0
            sigma_err_y = err_y_arr[2] if err_y_arr.size > 2 else 0.0
            self.atomNumFromGaussianY_std = self._gaussian_atom_number_uncertainty(
                self.y_peakHeight,
                self.y_width,
                amp_err_y,
                sigma_err_y,
                atom_scale,
            )
            self.y_width_std = err_y[2]
            self.y_center_std = err_y[1]
            print("y fit err = " +str(err_y))
            print("")

            # print("------------------- see here -------------------------")
            # print(self.x_center)
            print("------------------- Gaussian Center Co-Ords -------------------------")
        elif axis == 'x':
            self.x_center, self.x_width, self.x_offset, self.x_peakHeight, self.x_fitted, self.isXFitSuccessful, self.x_slope, err_x = gaussianFit(
                self.x_basis,
                self.x_summed,
                self.AOI,
                axis='x',
                max_od=max_od,
                fit_tails_only=self.checkTailFit.GetValue(),
                tail_fraction=tail_fraction,
                tail_side=tail_side_x,
            )
            self.atomNumFromGaussianX = (
                self.x_peakHeight
                * np.sqrt(2 * np.pi)
                * self.x_width
                * (self.pixelToDistance ** 2)
                / self.crossSection
            )
            err_x_arr = (
                np.asarray(err_x, dtype=float) if err_x is not None else np.zeros(5)
            )
            amp_err_x = err_x_arr[0] if err_x_arr.size > 0 else 0.0
            sigma_err_x = err_x_arr[2] if err_x_arr.size > 2 else 0.0
            self.atomNumFromGaussianX_std = self._gaussian_atom_number_uncertainty(
                self.x_peakHeight,
                self.x_width,
                amp_err_x,
                sigma_err_x,
                atom_scale,
            )
            self.x_width_std = err_x[2]
            self.x_center_std = err_x[1]
        else:
            self.y_center, self.y_width, self.y_offset, self.y_peakHeight, self.y_fitted, self.isYFitSuccessful, self.y_slope, err_y = gaussianFit(
                self.y_basis,
                self.y_summed,
                self.AOI,
                axis='y',
                max_od=max_od,
                fit_tails_only=self.checkTailFit.GetValue(),
                tail_fraction=tail_fraction,
                tail_side=tail_side_y,
            )
            self.atomNumFromGaussianY = (
                self.y_peakHeight
                * np.sqrt(2 * np.pi)
                * self.y_width
                * (self.pixelToDistance ** 2)
                / self.crossSection
            )
            err_y_arr = (
                np.asarray(err_y, dtype=float) if err_y is not None else np.zeros(5)
            )
            amp_err_y = err_y_arr[0] if err_y_arr.size > 0 else 0.0
            sigma_err_y = err_y_arr[2] if err_y_arr.size > 2 else 0.0
            self.atomNumFromGaussianY_std = self._gaussian_atom_number_uncertainty(
                self.y_peakHeight,
                self.y_width,
                amp_err_y,
                sigma_err_y,
                atom_scale,
            )
            self.y_width_std = err_y[2]
            self.y_center_std = err_y[1]


    def histogramEq(self, image, number_bins = 1000):

        image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)

        cdf = image_histogram.cumsum() # cumulative distribution function

        cdf =  cdf//cdf[-1] # normalize

        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)

    def testPCA(self, data, dims_rescaled_data=2):
        m, n = data.shape
        data -= data.mean(axis=0)
        R = np.cov(data, rowvar=False)
        evals, evecs = LA.eigh(R)        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        evecs = evecs[:, :dims_rescaled_data]
        return np.dot(evecs.T, data.T).T

    def setData(self, pca = False, gaussianFilter = False, histogramEqualization = False, rotation  = True):
        try:
            # Defringing is intentionally disabled. Preserve the pipeline call
            # signature while forcing the feature off.
            # defringing = self.checkApplyDefringing.GetValue()
            defringing = False
            absorbImg, self.imageData = readData(
                self.filename,
                self.fileType,
                [defringing, self.betterRef],
            )

            if pca is True:
                try:
                    print("----------------1=================")
                    #pca = sklearnPCA('mle')
                    print("----------------2================="                    )
                    temp = pca.fit_transform(-np.log(absorbImg))
                    print("----------------3=================")

                except Exception:
                    raise Exception("====== PCA ERROR ========")

            if gaussianFilter is True:
                try:
                    print("1111111111")
                    tempp = -np.log(absorbImg)
                    print("22222222222"                    )
                    signal = tempp[self.yTop:self.yBottom, self.xLeft:self.xRight]
                    print(signal)
                    print("33333333333")
                    filtered = gaussian_filter(tempp, 2, order = 0, truncate = 2)
                    print("44444444444")
                    print("55555555555"                    )
                    temp = filtered

                    print('====== Gaussian filter success ======')
                except Exception:
                    raise Exception("====== Gaussian Filter ERROR =======")

            if histogramEqualization is True:
                try:
                    temp = self.histogramEq(temp)
                    print('====== histogram equalization success ======')
                except Exception:
                    raise Exception("====== Histogram Equalization ERROR =======")

            if (histogramEqualization is False) and (gaussianFilter is False) and (pca is False):
                print('#################################################')
                print('Start Program')
                print('====== no filters IN =======')
                temp = -np.log(absorbImg)
                print('====== no filters OUT ======')

            if self.isNormalizationOn is True:
                temp = -np.log(createNormalizedAbsorbImg(self.imageData, self.AOI))

            # Median filtering has been retired from the processing pipeline.
            # if self.isMedianFilterOn is True:
            #     try:
            #        temp = medfilt(temp)
            #     except Exception:
            #         raise Exception("====== Median Filter ERROR ========")

            if rotation is True:
                try:
                    if self.isRotationNeeded is True:
                        temp = self.rotateImage(temp, self.imageAngle, [self.imagePivotX, self.imagePivotY])
                        print("====== rotation executed =======")
                    else:
                        print("====== No Rotation required for 0 deg. =======")
                except Exception:
                    raise Exception("====== rotation ERROR =======")
            if self.flipHCheck.GetValue():
                temp = np.fliplr(temp)
                self.imageData = [np.fliplr(img) for img in self.imageData]
            if self.flipVCheck.GetValue():
                temp = np.flipud(temp)
                self.imageData = [np.flipud(img) for img in self.imageData]

            self.atomImage = temp
            self._invalidate_limited_atom_cache()
            del absorbImg

            self.initializeAOI()

            self.updateAOIImageAndProfiles()
            self.updateFittingResults()
            self.setAtomNumber()
        except Exception as e:
            msg = wx.MessageDialog(self, str(e),'Setting Data failed', wx.OK)
            if msg.ShowModal() == wx.ID_OK:
                msg.Destroy()
            print("====== setData error =======")

    def update1DProfiles(self):
        
        self.axes2.clear()
        self.axes3.clear()

        if self.isXFitSuccessful is False:
            self.x_fitted = self.x_offset * np.ones(self.x_summed.shape[0])
            self.x_peakHeight = 0
            self.x_width = 0

        if self.isYFitSuccessful is False:
            self.y_fitted = self.y_offset * np.ones(self.y_summed.shape[0])
            self.y_peakHeight = 0
            self.y_width = 0

        ysize, xsize = self.atomImage.shape

        # if (self.currentXProfile is not None):
        #     #self.axes2.lines.remove(self.currentXProfile)
        #     self.axes2.clear()

        self.currentXProfile, = self.axes2.plot(self.x_basis, self.x_summed, 'b')

        # if (self.currentXProfileFit is not None):
        #     #self.axes2.lines.remove(self.currentXProfileFit)
        #     self.axes2.clear()

        self.currentXProfileFit, = self.axes2.plot(self.x_basis, self.x_fitted, 'r', label = "OD$_{max}$: " + str(round(self.od[0], 2)) + "\n" + "OD$_{avg}$: " + str(round(self.od[1], 2)))
        lx = self.axes2.legend(bbox_to_anchor = (1.4, 0.5), loc = "upper right")
        axes2_height = getattr(getattr(self.axes2, "bbox", None), "height", None)
        if lx is not None and axes2_height not in (None, 0):
            delta_x = 5 / axes2_height
            lx.set_bbox_to_anchor((1.4, 0.5 - delta_x))
        if self.isXFitSuccessful is False:
            for text in lx.get_texts():
                text.set_color("red")

        xMax = np.maximum(self.x_summed.max(), self.x_fitted.max())
        xMin = np.minimum(self.x_summed.min(), self.x_fitted.min())
        self.axes2.set_xlim([0, xsize])
        self.axes2.set_ylim([xMin, xMax])
        self.axes2.set_yticks(np.linspace(xMin, xMax, 4))

        # if self.checkDisplayRadialAvg.GetValue() is False:
        #     if (self.currentYProfile is not None):
        #         #self.axes3.lines.remove(self.currentYProfile)
        #         self.axes3.clear()

        self.currentYProfile, = self.axes3.plot(self.y_summed, self.y_basis,'b')

            # if (self.currentYProfileFit is not None):
            #     #self.axes3.lines.remove(self.currentYProfileFit)
            #     self.axes3.clear()

        self.currentYProfileFit, = self.axes3.plot(
            self.y_fitted,
            self.y_basis,
            'r',
            label=(
                f"X$_{{atom}}$: {round(self.atomNumFromFitX // 1E6, 2)}\n"
                f"Y$_{{atom}}$: {round(self.atomNumFromFitY // 1E6, 2)}"
            ),
        )
        ly = self.axes3.legend(bbox_to_anchor = (1.6, -0.168), loc = "lower right")
        axes3_height = getattr(getattr(self.axes3, "bbox", None), "height", None)
        if ly is not None and axes3_height not in (None, 0):
            delta_y = 5 / axes3_height
            ly.set_bbox_to_anchor((1.6, -0.168 - delta_y))
        if self.isYFitSuccessful is False:
            for text in ly.get_texts():
                text.set_color("red")


        yMax = np.maximum(self.y_summed.max(), self.y_fitted.max())
        yMin = np.minimum(self.y_summed.min(), self.y_fitted.min())
        self.axes3.set_xlim([yMin, yMax])
        self.axes3.set_ylim([ysize, 0])
        self.axes3.set_xticks(np.linspace(yMin, yMax, 3))
        self.axes3.set_yticks([])
        self.axes3.xaxis.set_ticks_position('top')


        self.deletePrev2DContour()
        figure = self.canvas1.figure

        right_margin = 0.88
        legend_boxes = []

        try:
            # ``tight_layout`` needs a renderer in order to calculate the space
            # required by artists that sit outside of the axes (the legends in
            # this case).  Force a draw so the renderer is available before we
            # ask the legends for their bounding boxes.
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for legend in (lx, ly):
                if legend is not None:
                    bbox = legend.get_window_extent(renderer)
                    bbox_fig = bbox.transformed(figure.transFigure.inverted())
                    legend_boxes.append(bbox_fig)
        except Exception:
            # If anything goes wrong while querying the legend bounds we fall
            # back to the default margin.  This is a best effort calculation
            # whose failure should not stop the UI update.
            legend_boxes = []

        if legend_boxes:
            overflow = max(box.x1 for box in legend_boxes) - 1.0
            if overflow > 0:
                # Leave a small cushion so the axes do not abut the legends.
                right_margin = 1.0 - overflow - 0.02
                right_margin = max(0.1, min(right_margin, 0.95))

        with warnings.catch_warnings():
            # ``tight_layout`` may still decline to make adjustments when the
            # axes would have to become extremely narrow.  Suppress the warning
            # because we handle that scenario manually below.
            warnings.simplefilter("ignore", UserWarning)
            figure.tight_layout()

        figure.subplots_adjust(right=right_margin)
        self.canvas1.draw()

    def _get_listbox_strings(self):
        try:
            return list(self.imageListBox.GetStrings())
        except AttributeError:
            return [
                self.imageListBox.GetString(i)
                for i in range(self.imageListBox.GetCount())
            ]

    def _find_listbox_index(self, items, label):
        if not label:
            return -1
        try:
            return items.index(label)
        except ValueError:
            return -1

    def _select_listbox_index(self, index):
        if index < 0:
            self.imageListBox.SetSelection(wx.NOT_FOUND)
            return
        self.imageListBox.SetSelection(index)
        ensure_visible = getattr(self.imageListBox, "EnsureVisible", None)
        if callable(ensure_visible):
            try:
                ensure_visible(index)
            except TypeError:
                ensure_visible(index, 0)

    def _apply_preferred_selection(self, items, preferred=None, fallback=None):
        chosen = None
        attempted = set()
        for label in (preferred, fallback):
            if not label or label in attempted:
                continue
            attempted.add(label)
            idx = self._find_listbox_index(items, label)
            if idx != -1:
                self._select_listbox_index(idx)
                chosen = items[idx]
                break
        if chosen is None:
            if items:
                self._select_listbox_index(0)
                chosen = items[0]
            else:
                self._select_listbox_index(-1)
        if chosen and self._pending_list_highlight == chosen:
            self._pending_list_highlight = None
        return chosen

    def updateImageListBox(self):
        self.updateFileList()

        new_items = [os.path.basename(name) for name in reversed(self.fileList)]
        current_items = self._get_listbox_strings()

        selection_idx = self.imageListBox.GetSelection()
        if selection_idx != wx.NOT_FOUND and 0 <= selection_idx < len(current_items):
            selection_label = current_items[selection_idx]
        else:
            selection_label = None

        preferred_label = self._pending_list_highlight
        if not preferred_label and self.filename:
            preferred_label = os.path.basename(self.filename)

        fallback_label = None
        if selection_label and selection_label != preferred_label:
            fallback_label = selection_label

        lists_differ = new_items != current_items

        needs_update = lists_differ
        if not needs_update:
            current_selection_idx = self.imageListBox.GetSelection()
            if current_selection_idx != wx.NOT_FOUND and 0 <= current_selection_idx < len(new_items):
                current_label = new_items[current_selection_idx]
            else:
                current_label = None

            desired_label = None
            for label in (preferred_label, fallback_label):
                if label and label in new_items:
                    desired_label = label
                    break

            if desired_label:
                needs_update = current_label != desired_label
            else:
                if new_items:
                    needs_update = current_selection_idx == wx.NOT_FOUND
                else:
                    needs_update = current_selection_idx != wx.NOT_FOUND

        if not needs_update:
            return

        self.imageListBox.Freeze()
        try:
            if lists_differ:
                self.imageListBox.Set(new_items)
            self._apply_preferred_selection(new_items, preferred_label, fallback_label)
        finally:
            self.imageListBox.Thaw()

    def highlight_image_in_list(self, filename):
        label = os.path.basename(filename) if filename else None
        self._pending_list_highlight = label or None
        if not label:
            return
        items = self._get_listbox_strings()
        idx = self._find_listbox_index(items, label)
        if idx != -1:
            self._select_listbox_index(idx)
            self._pending_list_highlight = None

    def highlight_top_image(self):
        if not getattr(self, "fileList", None):
            return
        try:
            top_file = self.fileList[-1]
        except IndexError:
            return
        if top_file:
            self.highlight_image_in_list(top_file)

    def setFileType(self, e):
        rb = e.GetEventObject()
        self.fileType = rb.GetLabel()
        self.updateImageListBox()
        print(self.fileType)


    def choosePath(self, e):
        myStyle = wx.FD_CHANGE_DIR | wx.FD_FILE_MUST_EXIST

        dialog = wx.DirDialog(
            None,
            "Choose a directory:",
            defaultPath=self.path,
            style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON,
        )
        path_changed = False
        if dialog.ShowModal() == wx.ID_OK:
            chosen_path = dialog.GetPath() + "\\"
            path_changed = chosen_path != self.path
            self.path = chosen_path
            self.imageFolderPath.SetValue(self.path)

        self.updateImageListBox()

        try:
            if path_changed and self.imageListBox.GetCount():
                if self.imageListBox.GetSelection() == wx.NOT_FOUND:
                    self.imageListBox.SetSelection(0)
                filename = self._load_selected_image_from_listbox()
                if filename:
                    self.setDataAndUpdate()
        finally:
            dialog.Destroy()

    def on_press(self, event):
        rect = self._ensure_aoi_patch()
        if rect is None:
            return
        self.xLeft = int(event.xdata)
        self.yTop = int(event.ydata)
        x0 = self.xLeft
        y0 = self.yTop

        # Initialize the drag endpoint so a simple click without dragging
        # still defines a valid rectangle.
        self.x1 = event.xdata
        self.y1 = event.ydata

        self.press = x0, y0, event.xdata, event.ydata


    def on_motion(self, event):
        if self.press is None: return

        if event.inaxes != self.axes1:

            return




        x0, y0, xpress, ypress = self.press



        self.x1 = event.xdata
        self.y1 = event.ydata




        rect = self._ensure_aoi_patch()
        if rect is not None:
            rect.set_width(self.x1 - xpress)
            rect.set_height(self.y1 - ypress)
            rect.set_xy((xpress, ypress))

        self.canvas1.draw_idle()

    def on_release(self, event):

        self.press = None
        self.xRight = int(self.x1)
        self.yBottom = int(self.y1)

        if self.xRight < self.xLeft:
            temp = self.xRight
            self.xRight = self.xLeft
            self.xLeft = temp
        if self.yBottom < self.yTop:
            temp = self.yBottom
            self.yBottom = self.yTop
            self.yTop = temp

        if (self.xLeft - 3 < 0): self.xLeft = 3
        if (self.yTop - 3 < 0): self.yTop = 3
        if (self.xRight + 4 >= self.imageData[0].shape[1]): self.xRight = self.imageData[0].shape[1] - 5
        if (self.yBottom + 4 >= self.imageData[0].shape[0]): self.yBottom = self.imageData[0].shape[0] - 5

        self.AOI1.SetValue(str(self.xLeft))
        self.AOI2.SetValue(str(self.yTop))
        self.AOI3.SetValue(str(self.xRight))
        self.AOI4.SetValue(str(self.yBottom))
        self.AOI = [[self.xLeft,self.yTop],[self.xRight,self.yBottom]]

        rect = self._ensure_aoi_patch()
        if rect is not None:
            rect.set_width(self.xRight - self.xLeft)
            rect.set_height(self.yBottom - self.yTop)
            rect.set_xy((self.xLeft, self.yTop))
            self.canvas1.draw()

        # The reset-AOI behaviour tied to defringing has been retired. Always
        # refresh the AOI view without re-running the deprecated pipeline.
        self.updateAOIImageAndProfiles()

        self.setAtomNumber()

        # If a new image arrived during ROI adjustment, load it now
        if self._pending_autorun:
            self._flush_pending_autorun()

    def setAtomNumber(self):


        self.setConstants()
        value = self.rawAtomNumber * (self.pixelToDistance**2) / self.crossSection
        plot_val = value / 1E6
        self.bigNcount3.SetValue(str("%.3f" % (plot_val)))

        if self.filename != getattr(self, "_last_plotted_file", None):
            self.plot_data["Atom_Number"].append(plot_val)
            self.plot_errors["Atom_Number"].append(np.sqrt(abs(value)) / 1E6)

            x_atom_val = getattr(self, "atomNumFromFitX", 0.0)
            y_atom_val = getattr(self, "atomNumFromFitY", 0.0)
            x_atom_err = getattr(self, "atomNumFromFitX_std", 0.0)
            y_atom_err = getattr(self, "atomNumFromFitY_std", 0.0)

            try:
                x_atom_val = float(x_atom_val)
            except (TypeError, ValueError):
                x_atom_val = 0.0
            try:
                y_atom_val = float(y_atom_val)
            except (TypeError, ValueError):
                y_atom_val = 0.0
            try:
                x_atom_err = float(x_atom_err)
            except (TypeError, ValueError):
                x_atom_err = 0.0
            try:
                y_atom_err = float(y_atom_err)
            except (TypeError, ValueError):
                y_atom_err = 0.0

            if not np.isfinite(x_atom_val) or x_atom_val < 0:
                x_atom_val = 0.0
            if not np.isfinite(y_atom_val) or y_atom_val < 0:
                y_atom_val = 0.0
            if not np.isfinite(x_atom_err) or x_atom_err < 0:
                x_atom_err = 0.0
            if not np.isfinite(y_atom_err) or y_atom_err < 0:
                y_atom_err = 0.0

            self.plot_data["X_Atom_Number"].append(x_atom_val / 1e6)
            self.plot_errors["X_Atom_Number"].append(x_atom_err / 1e6)
            self.plot_data["Y_Atom_Number"].append(y_atom_val / 1e6)
            self.plot_errors["Y_Atom_Number"].append(y_atom_err / 1e6)
            self.plot_data["True_X_Width"].append(self.true_x_width)
            self.plot_errors["True_X_Width"].append(self.true_x_width_std)
            self.plot_data["True_Y_Width"].append(self.true_y_width)
            self.plot_errors["True_Y_Width"].append(self.true_y_width_std)
            self.plot_data["X_Center"].append(self.x_center)
            self.plot_errors["X_Center"].append(self.x_center_std)
            self.plot_data["Y_Center"].append(self.y_center)
            self.plot_errors["Y_Center"].append(self.y_center_std)
            self._last_plotted_file = self.filename

        if self.atomNumberFrame:
            self.atomNumberFrame.set_number(self.bigNcount3.GetValue())
        if self.plotsFrame:
            self.plotsFrame.update_plot()





    def calc1DRadialAvgAndRefit(self):
        if self.checkDisplayRadialAvg.GetValue() is False:
            self.fit()
            return

        xCenter = self.x_center
        yCenter = self.y_center
        if self.isXFitSuccessful is False:
            xCenter = np.argmax(self.x_summed)

        if self.isYFitSuccessful is False:
            yCenter = np.argmax(self.y_summed)

        aoi_image = self._get_active_aoi_image()
        if aoi_image is None:
            return
        xarr = radialAverage(
            aoi_image,
            center=[xCenter, yCenter],
            boundary=[self.xLeft, self.yTop, self.xRight, self.yBottom],
        )
        if xarr.size == 0:
            self.fit()
            return
        num = len(xarr)
        self.x_basis = np.linspace(xCenter - num + 1, xCenter +  num - 1, 2* num - 2)
        self.x_summed = self.x_peakHeight * np.concatenate((np.flipud(xarr)[:-2], xarr), axis = 0)
        self.fit('x')


    def calc1DProfiles(self, min_od=None, max_od=None):
        source = self._get_active_aoi_image()
        if source is None:
            return

        y_size, x_size = source.shape

        if getattr(self, "_limited_aoi_image", None) is not None:
            img = source
        else:
            img = source.astype(float)
            if min_od is not None or max_od is not None:
                mask = np.isfinite(img)
                if max_od is not None:
                    mask &= np.abs(img) <= max_od
                if min_od is not None:
                    mask &= img >= min_od
                if not np.all(mask):
                    img = img.copy()
                    img[~mask] = np.nan

        self.x_summed = np.nansum(img, axis=0)
        self.x_valid_counts = np.sum(np.isfinite(img), axis=0)
        self.x_summed[self.x_valid_counts == 0] = np.nan
        self.x_basis = np.linspace(self.xLeft, self.xRight, x_size)

        self.y_summed = np.nansum(img, axis=1)
        self.y_valid_counts = np.sum(np.isfinite(img), axis=1)
        self.y_summed[self.y_valid_counts == 0] = np.nan
        self.y_basis = np.linspace(self.yTop, self.yBottom, y_size)

    def updateAOIImageAndProfiles(self):
        self._recompute_aoi_statistics()
        self.update1DProfilesAndFit(recompute_aoi=False)

    def edgeUpdate(self, _event=None):
        self._recompute_aoi_statistics()

    def displayRadialAvg(self, e):
        self.calc1DRadialAvgAndRefit()
        self.update1DProfiles()
        self.updateFittingResults()

    def displayNormalization(self, e):
        self.isNormalizationOn = self.checkNormalization.GetValue()
        self.setDataAndUpdate()

    def _sync_auto_save_controls(self):
        button = getattr(self, "saveSnippetButton", None)
        if button is None:
            return
        if getattr(self, "autoExportEnabled", False):
            button.Disable()
        else:
            button.Enable()

    def toggleAutoExport(self, e):
        self.autoExportEnabled = self.checkAutoExport.GetValue()
        self._sync_auto_save_controls()

    def toggleAtomNumberFrame(self, e):
        show = self.showAtomNumberCheck.GetValue()
        if show:
            if not self.atomNumberFrame:
                self.atomNumberFrame = AtomNumberDisplayFrame(self)
            self.atomNumberFrame.Show()
            self.atomNumberFrame.set_number(self.bigNcount3.GetValue())
        else:
            if self.atomNumberFrame:
                self.atomNumberFrame.Close()

    def set_live_plot_avg_enabled(self, enabled):
        self.live_plot_show_avg = bool(enabled)
        if isinstance(self._settings, dict):
            self._settings["live_plot_show_avg"] = self.live_plot_show_avg
        self._save_settings()

    def on_live_plot_opened(self):
        if getattr(self, "showPlotsBtn", None):
            self.showPlotsBtn.Disable()

    def on_live_plot_closed(self, show_avg_enabled):
        if getattr(self, "showPlotsBtn", None):
            self.showPlotsBtn.Enable()
        self.set_live_plot_avg_enabled(show_avg_enabled)
        self.plotsFrame = None

    def togglePlots(self, e):
        if self.plotsFrame:
            self.plotsFrame.Close()
        else:
            for key in self.plot_data:
                self.plot_data[key] = []
            for key in self.plot_errors:
                self.plot_errors[key] = []
            self._last_plotted_file = None
            self.plotsFrame = PlotsFrame(self, show_avg_default=self.live_plot_show_avg)
            self.on_live_plot_opened()
            self.plotsFrame.Show()
            self.plotsFrame.update_plot()

    def toggleAvgPreview(self, e):
        if self.avgFrame:
            self.avgFrame.Close()
        else:
            self.updateImageListBox()
            self.avgFrame = AvgPreviewFrame(self, self.avg_preview_count)
            if getattr(self, "avgPreviewBtn", None):
                self.avgPreviewBtn.Disable()
            self.avgFrame.Show()
            self.avgPreviewActive = True

    def saveSnippetNow(self, e):
        print("Saving fit...")
        self.snippetCommunicate(self.rawAtomNumber)
        print("Finished saving fit")

    def saveTOFFit(self, e):
        """Append detailed TOF fit data to the CSV file."""
        if not self.tof_fit_result or not self.tof_run_records:
            return

        print("Saving TOF fit results...")

        self.setConstants()
        headers = [
            "Run_ID",
            "Entry_Type",
            "Datetime",
            "Image_File",
            "TOF_ms",
            "Sigma_X_um",
            "Sigma_Y_um",
            "X_Center_um",
            "Y_Center_um",
            "Atom_Number",
            "Avg_Atom_Number",
            "Temp_X_uK",
            "Temp_Y_uK",
            "Trap_wX",
            "Trap_wY",
            "Temp_X_Err_uK",
            "Temp_Y_Err_uK",
            "PSD",
            "PSD_Err",
            "Fit_Type",
            "TOF_List_ms",
            "TOF_Offset_ms",
            "Num_Images",
        ]

        now = datetime.datetime.today().strftime("%a-%b-%d-%H_%M_%S-%Y")
        file_exists = os.path.exists(self.tofCsvPath) and os.path.getsize(self.tofCsvPath) > 0
        with open(self.tofCsvPath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            if not file_exists:
                writer.writerow(headers)

            # Write a row for each image used in the fit
            for rec in self.tof_run_records:
                row = [
                    self.tof_run_id,
                    "image",
                    now,
                    rec["image_file"],
                    rec["tof_ms"],
                    rec["sigma_x"],
                    rec["sigma_y"],
                    rec.get("x_center", ""),
                    rec.get("y_center", ""),
                    rec["atom_number"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    self.selected_tof_fit,
                    "",
                    "",
                    "",
                ]
                writer.writerow(row)

            # Summary row with final fit results
            summary_row = [
                self.tof_run_id,
                "summary",
                now,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                self.tof_fit_result.get("avg_atom_number", ""),
                self.tof_fit_result.get("temp_x", ""),
                self.tof_fit_result.get("temp_y", ""),
                self.tof_fit_result.get("trap_wx", ""),
                self.tof_fit_result.get("trap_wy", ""),
                self.tof_fit_result.get("temp_x_err", ""),
                self.tof_fit_result.get("temp_y_err", ""),
                self.tof_fit_result.get("psd", ""),
                self.tof_fit_result.get("psd_err", ""),
                self.selected_tof_fit,
                ";".join(str(v) for v in self.tof_values),
                self.tofOffset,
                len(self.tof_run_records),
            ]

            writer.writerow(summary_row)

        print("Finished saving TOF fit results")

    def FermionFitChosen(self, e):
        print("Mode: Fermion Fit")
        self.cleanValue()
        self.fermionResult.SetLabel('Fermion Fit Result')
        self.fText1.SetLabel('Size')
        self.fText2.SetLabel('Fugacity')
        self.tOverTFLabel.SetLabel('T//T_F')
        self._sync_preferred_fit_method_from_ui()

    def BosonFitChosen(self, e):
        print("Mode: Boson Fit")
        self.cleanValue()
        self.fermionResult.SetLabel('Boson Fit Result')
        self.fText1.SetLabel('Thermal Size')
        self.fText2.SetLabel('BEC Size')
        self.tOverTFLabel.SetLabel('BEC fraction')
        self._sync_preferred_fit_method_from_ui()

    def GaussianFitChosen(self, e):
        print("Mode: Gaussian Fit")
        self.cleanValue()
        self._sync_preferred_fit_method_from_ui()


    def cleanValue(self):
        self.fWidth.SetValue('')
        self.fq.SetValue('')
        self.tOverTF.SetValue('')
        self.gCenter.SetValue('')
        self.gSigma.SetValue('')
        self.gTemperature.SetValue('')

    def _update_image_title(self):
        if self.fileType == "fits" and self.filename:
            title = os.path.basename(self.filename)
        else:
            title = "Image Data"
        self.axes1.set_title(title, fontsize=12)

    def updateImageOnUI(self, layerNumber, hasFileSizeChanged):
        self.chosenLayerNumber = layerNumber
        if self.imageData:
            if layerNumber == 4:
                image_to_show = self._get_atom_image_for_display()
                if image_to_show is None:
                    image_to_show = self.atomImage
                if image_to_show is not None:
                    self.setCurrentImg(image_to_show, hasFileSizeChanged)
            else:
                self.setCurrentImg(self.imageData[layerNumber - 1], hasFileSizeChanged)
                self._set_od_mask_summary(None, None)
            self._apply_display_settings()
            self._update_image_title()
            self.canvas1.draw()


    def setDataAndUpdate(self):
        self.setData()
        hasFileSizeChanged = self.checkIfFileSizeChanged()



        self.updateImageOnUI(self.chosenLayerNumber, hasFileSizeChanged)
        self.edgeUpdate()

    def _load_selected_image_from_listbox(self, warn_on_missing=False):
        ind = self.imageListBox.GetSelection()
        if ind == wx.NOT_FOUND:
            return None

        oldFileNumber = len(getattr(self, "fileList", []))
        self.updateFileList()
        newFileNumber = len(self.fileList)

        if not newFileNumber:
            return None

        if warn_on_missing and oldFileNumber != newFileNumber:
            msg = wx.MessageDialog(
                self,
                "Such image file may not exist in the file directory",
                "Index Error",
                wx.OK,
            )
            try:
                if msg.ShowModal() == wx.ID_OK:
                    self.updateImageListBox()
            finally:
                msg.Destroy()
            return None

        target_index = newFileNumber - ind - 1
        if target_index < 0 or target_index >= newFileNumber:
            return None

        self.filename = self.fileList[target_index]
        self.highlight_image_in_list(self.filename)
        return self.filename

    def chooseImg(self, e):
        start = time.time()
        filename = self._load_selected_image_from_listbox(warn_on_missing=True)
        if not filename:
            return

        if not getattr(self, "_startup_complete", False):
            return

        print("----the filename----")
        print(filename)
        end = time.time()
        self.setDataAndUpdate()

    def showImgValue(self, e):
        if e.xdata and e.ydata:
            x = int(e.xdata)
            y = int(e.ydata)

            if self.imageData and (x >= 0  and x < self.imageData[0].shape[1]) and (y >= 0 and y < self.imageData[0].shape[0]):
                self.cursorX.SetValue(str(x))
                self.cursorY.SetValue(str(y))
                if self.layer1Button.GetValue():
                    self.cursorZ.SetValue(str(int(self.imageData[0][y][x])))
                elif self.layer2Button.GetValue():
                    self.cursorZ.SetValue(str(int(self.imageData[1][y][x])))
                elif self.layer3Button.GetValue():
                    self.cursorZ.SetValue(str(int(self.imageData[2][y][x])))
                elif self.layer4Button.GetValue():
                    active = self._get_active_atom_image()
                    if active is not None and 0 <= y < active.shape[0] and 0 <= x < active.shape[1]:
                        value = active[y][x]
                        if np.isfinite(value):
                            self.cursorZ.SetValue(f"{value:0.4f}")
                        else:
                            self.cursorZ.SetValue("--")
                    else:
                        self.cursorZ.SetValue("--")

    def fitImage(self, e):
        format = "%a-%b-%d-%H_%M_%S-%Y"
        today = datetime.datetime.today()
        self.timeString = today.strftime(format)

        self.benchmark_startTime=time.time()
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.suspend_processing_controls())
            if getattr(self, "_watch_suspend_depth", 0) == 0:
                stack.enter_context(self.suspend_auto_watch())
            else:
                stack.enter_context(contextlib.nullcontext())
            if self.readImage():
                print("Begin to fit...")
                self.showImg(e)
            else:
                self.filename = None
                self._update_image_title()
                self.canvas1.draw()
        tmp = time.time()

    def deletePrev2DContour(self):
        if self.quickFitBool==True:
            self.quickFitBool=False
            if self.fitOverlay is not None:
                for coll in self.fitOverlay.collections:
                    coll.remove()
                self.canvas1.draw()
            return True

        if self.quickFitBool==False:
            self.quickFitBool=True
        return False

    def _iter_processing_controls(self):
        control_names = [
            "fitsFile",
            "aiaFile",
            "tifFile",
            "flipHCheck",
            "flipVCheck",
            "colormapChoice",
            "fitMethodGaussian",
            "fitMethodFermion",
            "fitMethodBoson",
            "checkTailFit",
            "tailFractionCtrl",
            "tailLeftCheck",
            "tailRightCheck",
            "tailTopCheck",
            "tailBottomCheck",
            "checkNormalization",
            "checkDisplayRadialAvg",
            "checkLimitOD",
            "maxODCtrl",
            "imageFolderPath",
            "choosePathButton",
            "snippetTextBox",
            "chooseSnippetButton",
            "checkAutoExport",
            "atomRadioBox",
            "magnif",
            "pxSize",
            "detuningCtrl",
            "showAtomNumberCheck",
            "imageListBox",
            "AOI1",
            "AOI2",
            "AOI3",
            "AOI4",
            "leftRightEdge",
            "updownEdge",
            "angleBox",
            "pivotXBox",
            "pivotYBox",
            "rotationButton",
        ]
        for name in control_names:
            ctrl = getattr(self, name, None)
            if ctrl is None:
                continue
            yield ctrl

    @contextlib.contextmanager
    def suspend_processing_controls(self):
        if not hasattr(self, "_processing_disable_depth"):
            self._processing_disable_depth = 0

        self._processing_disable_depth += 1
        disabled_here = False
        disabled_controls = []
        previous_fit_method = None
        try:
            if self._processing_disable_depth == 1:
                disabled_here = True
                previous_fit_method = self._current_fit_method_from_controls()
                for ctrl in self._iter_processing_controls():
                    if not hasattr(ctrl, "Disable") or not hasattr(ctrl, "Enable"):
                        continue
                    try:
                        is_enabled = ctrl.IsEnabled()
                    except AttributeError:
                        is_enabled = True
                    if not is_enabled:
                        continue
                    try:
                        ctrl.Disable()
                    except Exception:
                        continue
                    disabled_controls.append(ctrl)
            yield
        finally:
            if disabled_here:
                for ctrl in disabled_controls:
                    try:
                        ctrl.Enable()
                    except Exception:
                        continue
                if previous_fit_method:
                    previous_state = getattr(self, "_suppress_fit_sync", False)
                    self._suppress_fit_sync = True
                    try:
                        self._ensure_fit_method_selection(previous_fit_method)
                        self._preferred_fit_method = previous_fit_method
                    finally:
                        self._suppress_fit_sync = previous_state
            self._processing_disable_depth = max(self._processing_disable_depth - 1, 0)

    @contextlib.contextmanager
    def suspend_auto_watch(self):
        """Temporarily pause directory watching while processing fits."""

        monitor = getattr(self, "monitor", None)
        can_pause = (
            self.autoRunning
            and monitor is not None
            and hasattr(monitor, "pause")
            and hasattr(monitor, "resume")
        )

        if not hasattr(self, "_watch_suspend_depth"):
            self._watch_suspend_depth = 0

        self._watch_suspend_depth += 1
        paused_here = False
        label_state = None

        try:
            if self._watch_suspend_depth == 1:
                auto_btn = getattr(self, "autoButton", None)
                if can_pause:
                    try:
                        paused_here = bool(monitor.pause())
                    except Exception as exc:
                        print(f"Unable to pause directory watcher: {exc}")
                        paused_here = False
                    else:
                        if paused_here and auto_btn is not None and auto_btn.GetLabel() == "Watching":
                            auto_btn.SetLabel("Paused")
                            auto_btn.Disable()
                            label_state = "watching"
                if (
                    label_state is None
                    and not self.autoRunning
                    and auto_btn is not None
                    and auto_btn.GetLabel() != "Paused"
                ):
                    auto_btn.SetLabel("Paused")
                    auto_btn.Disable()
                    label_state = "manual"
            yield
        finally:
            self._watch_suspend_depth = max(self._watch_suspend_depth - 1, 0)
            resume_attempted = False
            resume_succeeded = False
            resume_failed = False
            if (
                can_pause
                and paused_here
                and self._watch_suspend_depth == 0
                and self.autoRunning
            ):
                resume_attempted = True
                try:
                    resume_succeeded = bool(monitor.resume())
                except Exception as exc:
                    print(f"Unable to resume directory watcher: {exc}")
                    resume_failed = True
                else:
                    resume_failed = not resume_succeeded

            auto_btn = getattr(self, "autoButton", None)
            if label_state == "watching" and auto_btn is not None:
                if resume_attempted:
                    if resume_failed:
                        auto_btn.SetLabel("Start")
                        if self.autoRunning:
                            auto_btn.Disable()
                        else:
                            auto_btn.Enable()
                    else:
                        if resume_succeeded:
                            auto_btn.SetLabel("Watching")
                            auto_btn.Enable()
                        else:
                            auto_btn.SetLabel("Start")
                            if self.autoRunning:
                                auto_btn.Disable()
                            else:
                                auto_btn.Enable()
                elif self._watch_suspend_depth == 0:
                    auto_btn.SetLabel("Start")
                    if self.autoRunning:
                        auto_btn.Disable()
                    else:
                        auto_btn.Enable()
            elif label_state == "manual" and self._watch_suspend_depth == 0:
                if auto_btn is not None:
                    auto_btn.SetLabel("Start")
                    auto_btn.Enable()

    def show2DContour(self, e):
        if self.deletePrev2DContour():
            return
        y_size,x_size= self.AOIImage.shape
        x_basis = np.linspace(self.xLeft, self.xRight, x_size)
        y_basis = np.linspace(self.yTop, self.yBottom, y_size)
        x_basis, y_basis = np.meshgrid(x_basis, y_basis)


        g = lambda coords: np.exp(-1*((coords[0] - self.x_center)**2)//(2*self.x_width**2))*np.exp(-1*((coords[1] - self.y_center)**2)//(2*self.y_width**2))
        plot_overlay_data=g((x_basis,y_basis))
        self.fitOverlay = self.axes1.contour(x_basis, y_basis, plot_overlay_data.reshape(y_size, x_size), 8, cmap='afmhot')

        self.canvas1.draw()

    def readImage(self):
        plotMin = 0.0
        plotMax = 0.3
        try:
            if self.autoRunning == False:
                print(self.path)
                if not self.path:
                    print("------------Wrong Folder!--------")
                    return None
                # Only update to the newest file when not in the middle of a TOF fit
                if not self._tof_fit_running and not self.filename:
                    self.updateLatestFileName()
                if not self.fileList:
                    print("Warning: fileList is empty.")
                    return None
            elif self.autoRunning == True:
                # In auto mode we normally update to the newest file, but
                # skip this when processing a TOF fit so each specified file is used
                if not self._tof_fit_running:
                    self.updateLatestFileName()

            self.updateImageListBox()
            self.setData()


            print("Successfully Read Image")
            return True
        except Exception as err:
            print("Failed to read this image.")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def saveAsRef(self):
        if self.checkSaveAsRef.GetValue() is True:
            path = self.path[:-2] + "_ref\\"
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy2(self.filename, path)
            self.defringingRefPath = path

    def showImg(self, e):
        #  TODO: This only saves the data if the prog is set to autoread. 
        
        if not self.autoRunning:
            mode = "Single"
        else:
            mode = "Auto"

        print(mode)


        N_int = 0 #atomNumber(self.AOIImage, self.offset)


        self.setAtomNumber()

        if (self.chosenLayerNumber == 4):
            imageToShow = self._get_atom_image_for_display()
            if imageToShow is None:
                imageToShow = self.atomImage
        else:
            imageToShow = self.imageData[self.chosenLayerNumber - 1]
            self._set_od_mask_summary(None, None)

        hasFileSizeChanged = self.checkIfFileSizeChanged()




        self.setCurrentImg(imageToShow, hasFileSizeChanged)
        self._apply_display_settings()
        self._update_image_title()

        if mode=="Auto" and self.checkAutoExport.GetValue():
            self.snippetCommunicate(self.rawAtomNumber)

        self.canvas1.draw()

        self.Update()







        self.benchmark_endTime=time.time()
        print( "Process Time: " + str(round(abs(self.benchmark_startTime-self.benchmark_endTime), 1)) + " seconds")
        gc.collect()


    def startAutoRun(self, e):
        self.filename = None
        self.updateImageListBox()
        try:
            if self.autoRunning == False:
                self.snippetTextBox.Disable()
                self.chooseSnippetButton.Disable()
                self.imageFolderPath.Disable()
                self.choosePathButton.Disable()
                print("Start Auto Run.. Begin Watching")
                self.autoButton.SetLabel('Watching')
                self.autoButton.Enable()
                print("Observing " + self.imageFolderPath.GetValue())

                self.monitor = Monitor(self.path, self.autoRun, self.expectedFileSize_mb)
                self.monitor.createObserverAndStart()
                self.autoRunning = True
            elif self.autoRunning == True:
                self.snippetTextBox.Enable()
                self.chooseSnippetButton.Enable()
                self.imageFolderPath.Enable()
                self.choosePathButton.Enable()
                print("Stop Watching Folder.")

                self.autoButton.SetLabel('Start')
                self.autoButton.Enable()

                if self.monitor:
                    self.monitor.stop()
                    self.monitor.join()

                else:
                    print("------------There's NO monitor pointer-----------")
                self.autoRunning = False
        except:
            print("Sorry. There is some problem about auto fit. Please just restart the program.")


    def _finish_startup(self):
        """Mark the frame as ready and kick off directory monitoring."""

        def _post_startup_tasks():
            # Ensure the radio buttons match the saved preference before any
            # later user actions trigger a load or fit.
            self._ensure_fit_method_selection()

            if not self.autoRunning:
                # ``startAutoRun`` toggles the watcher and manages related UI
                # state. Call it here so the monitor begins only after
                # initialisation is complete.
                self.startAutoRun(None)

            try:
                self.imageListBox.SetSelection(wx.NOT_FOUND)
            except Exception:
                pass

            self._startup_complete = True
            self._flush_pending_autorun()

        wx.CallAfter(_post_startup_tasks)

    def _flush_pending_autorun(self):
        """Run any deferred auto-run request once startup is complete."""

        if getattr(self, "_pending_autorun", False):
            self._pending_autorun = False
            self.autoRun()


    def autoRun(self):
        if not getattr(self, "_startup_complete", False):
            # When images already exist in the watched folder the observer can
            # trigger before the UI has finished initialising. Defer until the
            # saved settings (including the fit method) are in place.
            self._pending_autorun = True
            print("Auto-run triggered before start-up completed; deferring fit.")
            return
        self._ensure_fit_method_selection()
        # Defer loading if the ROI selection box is being resized
        if self.press is not None:
            self._pending_autorun = True
            return

        if self.monitor.oldObserver is not None:
            self.monitor.oldObserver.stop()
            self.monitor.oldObserver.join()
            self.monitor.oldObserver = None

        self.checkTimeChange()



        print(self.path)

        if (self.timeChanged):
            self.today = datetime.date.today()
            previousPath = self.path

            # Determine the base directory that does not include the
            # date/atom subdirectories. ``normpath`` strips any trailing
            # separator so that repeated ``dirname`` calls behave as
            # expected.
            base = os.path.normpath(self.path)
            for _ in range(4):
                base = os.path.dirname(base)

            # Reconstruct the path for the new day while keeping ``self.path``
            # as a string. ``os.path.join`` is used to handle path separators
            # appropriately on any platform.
            self.path = os.path.join(
                base,
                str(self.today.year),
                str(self.today.month),
                str(self.today.day),
                self.atom,
                "",
            )

            print("################# Time changed ##################")
            print("new path:    " + self.path)

            if not os.path.exists(self.path):
                print("+++++++++++++ TRIED TO CHANGE THE DIRECTORY BUT NO EXIST +++++++++++++++")
                self.path = previousPath


            self.imageFolderPath.SetValue(self.path)
            self.updateImageListBox()
            if self.monitor:
                self.monitor.changeFilePath(self.path)
            else:
                print("------NO monitor pointer //// Failed to change file Path /////")



        print("########################## Found new image #########################")
        self.fitImage(wx.EVT_BUTTON)


    def saveResult(self, e):
        if self.fitMethodFermion.GetValue():
            self.saveFermionResult(e)
        elif self.fitMethodBoson.GetValue():
            self.saveBosonResult(e)
        elif self.fitMethodGaussian.GetValue():
            self.saveGaussianResult(e)

    def saveBosonResult(self, e):
        f = open("C:\\AndorImg\\boson_data.txt", "a")
        f.writelines(self.timeString + '\t' + self.tof.GetValue() + '\t'\
         + self.atomNumberInt.GetValue() + '\t' \
         + str(self.bosonParams[2]) + '\t' + str(self.bosonParams[3]) + '\t' \
         + str(1//np.sqrt(self.bosonParams[7])) + '\t' + str(1//np.sqrt(self.bosonParams[8]))  \
            + '\n')

        f.close()

    def saveFermionResult(self, e):
        f = open("C:\\AndorImg\\fermion_data.txt", "a")

        f.writelines(self.timeString + '\t' + self.tof.GetValue() + '\t'\
         + str(self.gaussionParams[0]) + '\t' + str(self.gaussionParams[1]) + '\t' \
         + self.atomNumberInt.GetValue() + '\t' \
         + str(self.fermionParams[2]) + '\t' + str(self.fermionParams[3]) + '\t' \
         + str(self.fermionParams[4]) + '\n')

        f.close()

    def saveGaussianResult(self, e):
        f = open("C:\\AndorImg\\gaussian_data.txt", "a")
        f.writelines(self.timeString + '\t' + self.tof.GetValue() + '\t'\
         + str(self.gaussionParams[0]) + '\t' + str(self.gaussionParams[1]) + '\t' \
         + str(self.atomNumberInt.GetValue()) + '\t' + str(self.atomNumberIntFit.GetValue()) + '\t'\
         + str(self.gaussionParams[2]) + '\t' + str(self.gaussionParams[3]) \
         + '\n')

        f.close()

    def cleanData(self, e):
        if self.fitMethodFermion.GetValue():
            f = open("C:\\AndorImg\\fermion_data.txt", "w")
        elif self.fitMethodBoson.GetValue():
            f = open("C:\\AndorImg\\boson_data.txt", "w")
        f.close()

    # Write to CSV file when in autoread mode.

    def snippetCommunicate(self, N_intEdge):
        self.setConstants()
        
        # Ensure destination folder exists and construct dated file name
        os.makedirs(self.snippetPath, exist_ok=True)
        csv_name = datetime.datetime.now().strftime("%Y%m%d") + "_Image_Analysis_Raw_Data.csv"
        csv_path = os.path.join(self.snippetPath, csv_name)

        var_label = "Variable"
        var_value = ""
        if getattr(self, "fitWindow", None) and getattr(self.fitWindow, "results", None) and self.filename:
            files = self.fitWindow.results.get("image_file", [])
            if self.filename in files:
                idx = files.index(self.filename)
                var_value = self.fitWindow.var_values[idx] / self.fitWindow.var_scale
                var_label = self.fitWindow.varLabelCtrl.GetValue()

        headers = [
            "Time",
            "Image File Name",
            var_label,
            "Atom",
            "Fit Type",
            "Normalization",
            "Radially Averaged",
            "OD Limit",
            "Avg Images",
            "Magnification",
            "Pixel_Size",
            "Atom_Number",
            "X_Atom_Count",
            "Y_Atom_Count",
            "X_Center",
            "Y_Center",
            "True_X_Width",
            "True_X_Width_Std",
            "True_Y_Width",
            "True_Y_Width_Std",
            "Temp_X",
            "Temp_Y",
            "X_BEC_Ratio",
            "Y_BEC_Ratio",
            "X_T_over_Tc",
            "Y_T_over_Tc",
            "X_TF_Radius",
            "Y_TF_Radius",
            "X_T_over_TF",
            "Y_T_over_TF",
            "X_Fermi_Radius",
            "Y_Fermi_Radius",
        ]

        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

        try:
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                if not file_exists:
                    writer.writerow(headers)

                if not N_intEdge:
                    N_intEdge = -1

                if self.fitMethodGaussian.GetValue():
                    fit_type = "Gaussian"
                elif self.fitMethodFermion.GetValue():
                    fit_type = "Fermion"
                elif self.fitMethodBoson.GetValue():
                    fit_type = "Boson"
                else:
                    fit_type = ""

                normalization = self.checkNormalization.GetValue()
                radial_avg = self.checkDisplayRadialAvg.GetValue()
                limit_od = (
                    float(self.maxODCtrl.GetValue())
                    if self.checkLimitOD.GetValue()
                    else 5
                )
                avg_images = self.avg_preview_count if self.avgPreviewActive else 1

                image_file = os.path.basename(self.filename) if self.filename else ""

                data_row = [
                    self.timeString,
                    image_file,
                    var_value,
                    self.atom,
                    fit_type,
                    normalization,
                    radial_avg,
                    limit_od,
                    avg_images,
                    self.magnification,
                    self.pixelSize,
                    float(self.rawAtomNumber) * (self.pixelToDistance**2) / self.crossSection,
                    self.atomNumFromFitX,
                    self.atomNumFromFitY,
                    self.x_center,
                    self.y_center,
                    self.true_x_width,
                    self.true_x_width_std,
                    self.true_y_width,
                    self.true_y_width_std,
                    self.temperature[0],
                    self.temperature[1],
                    self.x_becPopulationRatio,
                    self.y_becPopulationRatio,
                    self.x_tOverTc,
                    self.y_tOverTc,
                    self.x_thomasFermiRadius,
                    self.y_thomasFermiRadius,
                    self.x_TOverTF,
                    self.y_TOverTF,
                    self.x_fermiRadius,
                    self.y_fermiRadius,
                ]

                writer.writerow(data_row)

                if getattr(self, "fitWindow", None) and self.fitWindow.plot_data:
                    writer.writerow([])
                    for idx, pdata in sorted(self.fitWindow.plot_data.items()):
                        if pdata.get("popt") is not None:
                            writer.writerow([f"Plot {idx}", pdata["func_name"]])
                            writer.writerow(FIT_FUNCTIONS[pdata["func_name"]]["param_names"])
                            writer.writerow([f"{v:.6g}" for v in pdata["popt"]])
                            if pdata.get("derived"):
                                writer.writerow(list(pdata["derived"].keys()))
                                writer.writerow([f"{v:.6g}" for v in pdata["derived"].values()])

        except Exception as e:
            print(f"Error writing to file: {e}")

        except IOError:
            msg = wx.MessageDialog(self, 'The file path for SnippetServer is not correct', 'Incorrect File Path', wx.OK)
            if msg.ShowModal() == wx.ID_OK:
                msg.Destroy()
            return


    def rotateImage(self, img, angle, pivot):
        padX = [int(img.shape[1] - pivot[0]), int(pivot[0])]
        padY = [int(img.shape[0] - pivot[1]), int(pivot[1])]
        imgP = np.pad(img, [padY, padX], 'constant', constant_values=[(0,0), (0,0)])
        imgR = ndimage.rotate(imgP, angle, reshape = False)
        return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

    def copy3Layer(self):
        pass

    def saveAbsorbImg(self, atomImage):
        pass


    def readListData(self, e):
        if self.fitMethodFermion.GetValue():
            f = open("C:\\AndorImg\\fermion_data.txt", "r")
            self.data = f.readlines()
        elif self.fitMethodBoson.GetValue():
            f = open("C:\\AndorImg\\boson_data.txt", "r")
            self.data = f.readlines()
        elif self.fitMethodGaussian.GetValue():
            f = open("C:\\AndorImg\\guassian_data.txt", "r")
            self.data = f.readlines()


        self.dataReadedText.SetValue("input: %i"%len(self.data))

        for i in range(len(self.data)):
            self.data[i] = self.data[i].split(' , ')


        f.close()

    def fitListData(self, e):
        tofList = []
        RXList = []
        RYList = []

        atom = ""

        n = len(self.data)
        for i in self.data:
            # Convert TOF from milliseconds to seconds to preserve fractional values
            tofList.append(float(i[1]) / 1000.0)
            RXList.append(float(i[3]) * self.pixelToDistance)
            RYList.append(float(i[4]) * self.pixelToDistance)

        if self.fitMethodBoson.GetValue():
            atom = "Cs"
        elif self.fitMethodFermion.GetValue():
            atom = "Li"
        
        tx, ty, wx, wy = dataFit(atom, tofList, RXList, RYList)
        # Display temperatures in microkelvin for consistency with other TOF fit outputs
        self.fitTempText.SetValue('(%.1f' %(tx*1E6) + ' , ' + '%.1f )' %(ty*1E6))

        self.fitTrapAxialFreqText.SetValue(str('%.1f' % (wy//(2*np.pi))))
        self.fitTrapRadialFreqText.SetValue(str('%.1f' % (wx//(2*np.pi))))

    def drawAtomNumber(self, e):
        atomNumberI = []
        n = len(self.data)
        for i in self.data:
            atomNumberI.append(int(i[2]))
        atomNumberPlot(n, atomNumberI)

    def on_close(self, event):
        """Handle the main window closing by saving settings and cleaning up."""

        try:
            self._save_settings()
        except Exception as err:
            print(f"Error saving settings on close: {err}")

        monitor = getattr(self, "monitor", None)
        if monitor and getattr(self, "autoRunning", False):
            try:
                monitor.stop()
            except Exception:
                pass
            try:
                monitor.join()
            except Exception:
                pass

        for attr in ("atomNumberFrame", "plotsFrame", "avgFrame"):
            frame = getattr(self, attr, None)
            if frame:
                try:
                    frame.Destroy()
                except Exception:
                    try:
                        frame.Close()
                    except Exception:
                        pass

        event.Skip()


if __name__ == '__main__':
    app = wx.App()
    ImageUI(None, title='Atom Image Analysis 10.0')
    app.MainLoop()
