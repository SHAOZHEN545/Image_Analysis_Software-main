"""Average image preview window and processing logic."""

from __future__ import annotations

import datetime
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.cm as cm
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import wx


DEFAULT_T_MIN = 1e-6


class AvgPreviewFrame(wx.Frame):
    """Preview window to average the last N images and accept the result."""

    def __init__(self, parent: wx.Frame, n: int) -> None:
        parent_size = parent.GetSize()
        parent_canvas = getattr(parent, "canvas1", None)
        if parent_canvas:
            canvas_size = parent.canvas1.GetSize()
            width = max(canvas_size.GetWidth() + 100, parent_size[0] // 2)
            height = max(canvas_size.GetHeight() + 140, parent_size[1] // 2)
        else:
            width = parent_size[0] // 2
            height = parent_size[1] // 2
        height = max(1, int(height * 1.2))
        new_size = (width, height)
        super().__init__(parent, title="Average Preview", size=new_size)
        self.parent = parent
        self.n = max(1, n)

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        controls_box = wx.BoxSizer(wx.VERTICAL)
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        mode_label = wx.StaticText(panel, label="Mode:")
        mode_choices = [
            "Average per-shot OD (Mean column density)",
            "Pooled transmission (Effective OD)",
        ]
        self.mode_choice = wx.Choice(panel, choices=mode_choices)
        self.mode_choice.SetSelection(0)
        self._mode_tooltips = {
            0: (
                "Computes OD for each shot after normalization and optional "
                "saturation correction, then averages the OD images."
            ),
            1: (
                "Pools atom/reference counts across shots before computing a "
                "single effective OD."
            ),
        }
        self.mode_choice.SetToolTip(self._mode_tooltips[0])
        input_sizer.Add(mode_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        input_sizer.Add(self.mode_choice, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        input_label = wx.StaticText(panel, label="# Images:")
        self.count_ctrl = wx.TextCtrl(
            panel, value=str(self.n), size=(60, -1), style=wx.TE_PROCESS_ENTER
        )
        self.count_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_count_enter)
        input_sizer.Add(input_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        input_sizer.Add(self.count_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        controls_box.Add(input_sizer, 0, wx.ALL | wx.EXPAND, 5)

        options_box = wx.StaticBoxSizer(wx.StaticBox(panel, label="Options"), wx.VERTICAL)

        self.saturation_checkbox = wx.CheckBox(panel, label="Saturation Correction")
        self.saturation_checkbox.SetToolTip(
            "Use two-level correction: OD → OD + (I_atoms − I_ref)/I_sat_eff with "
            "I_sat_eff = I_sat(1+4Δ²/Γ²). Requires intensities normalized to the "
            "same physical units as I_sat."
        )
        options_box.Add(self.saturation_checkbox, 0, wx.BOTTOM, 4)

        guard_sizer = wx.BoxSizer(wx.VERTICAL)

        rmin_row = wx.BoxSizer(wx.HORIZONTAL)
        rmin_label = wx.StaticText(panel, label="Min Reference:")
        self.rmin_ctrl = wx.TextCtrl(
            panel, value="0", size=(80, -1), style=wx.TE_PROCESS_ENTER
        )
        self.rmin_ctrl.SetToolTip(
            "Pixels with reference below this level are masked to avoid unstable ratios."
        )
        rmin_row.Add(rmin_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        rmin_row.Add(self.rmin_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        guard_sizer.Add(rmin_row, 0, wx.BOTTOM, 4)

        tmin_row = wx.BoxSizer(wx.HORIZONTAL)
        tmin_label = wx.StaticText(panel, label="Min Transmission:")
        self.tmin_ctrl = wx.TextCtrl(
            panel, value="1e-6", size=(80, -1), style=wx.TE_PROCESS_ENTER
        )
        self.tmin_ctrl.SetToolTip(
            "Prevents log underflow in very dark pixels."
        )
        tmin_row.Add(tmin_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        tmin_row.Add(self.tmin_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        guard_sizer.Add(tmin_row, 0)

        self.saturation_panel = wx.Panel(panel)
        sat_sizer = wx.FlexGridSizer(0, 2, 5, 5)
        sat_sizer.AddGrowableCol(1, 1)

        isat_label = wx.StaticText(
            self.saturation_panel, label="I_sat (mW/cm²):"
        )
        self.sat_isat_ctrl = wx.TextCtrl(self.saturation_panel, size=(120, -1))
        sat_sizer.Add(isat_label, 0, wx.ALIGN_CENTER_VERTICAL)
        sat_sizer.Add(self.sat_isat_ctrl, 1, wx.EXPAND)

        gamma_label = wx.StaticText(self.saturation_panel, label="Γ (MHz):")
        self.sat_gamma_ctrl = wx.TextCtrl(self.saturation_panel, size=(120, -1))
        sat_sizer.Add(gamma_label, 0, wx.ALIGN_CENTER_VERTICAL)
        sat_sizer.Add(self.sat_gamma_ctrl, 1, wx.EXPAND)

        detuning_label = wx.StaticText(
            self.saturation_panel, label="Δ (MHz):"
        )
        self.sat_detuning_ctrl = wx.TextCtrl(
            self.saturation_panel, value="0", size=(120, -1)
        )
        sat_sizer.Add(detuning_label, 0, wx.ALIGN_CENTER_VERTICAL)
        sat_sizer.Add(self.sat_detuning_ctrl, 1, wx.EXPAND)

        self.saturation_panel.SetSizer(sat_sizer)
        options_box.Add(self.saturation_panel, 0, wx.EXPAND | wx.TOP, 4)

        self.inverse_variance_checkbox = wx.CheckBox(
            panel, label="Inverse Variance Weighting"
        )
        self.inverse_variance_checkbox.SetToolTip(
            "Weight per-shot ODs by 1/Var(OD) ≈ 1/(1/A + 1/R) to improve SNR."
        )
        options_box.Add(self.inverse_variance_checkbox, 0, wx.TOP | wx.BOTTOM, 4)

        options_box.Add(guard_sizer, 0, wx.TOP, 4)

        controls_box.Add(options_box, 0, wx.ALL | wx.EXPAND, 5)

        process_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.process_btn = wx.Button(panel, label="Process")
        self.process_btn.SetToolTip("Load and average the selected images")
        process_sizer.Add(self.process_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.progress = wx.Gauge(panel, range=1, size=(140, -1))
        self.progress.SetToolTip("Shows progress while averaging images")
        process_sizer.Add(self.progress, 1, wx.ALIGN_CENTER_VERTICAL)
        controls_box.Add(process_sizer, 0, wx.ALL | wx.EXPAND, 5)

        vbox.Add(controls_box, 0, wx.EXPAND)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        accept_btn = wx.Button(panel, label="Accept")
        cancel_btn = wx.Button(panel, label="Cancel")
        btn_sizer.AddStretchSpacer()
        btn_sizer.Add(accept_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        vbox.Add(btn_sizer, 0, wx.EXPAND)

        panel.SetSizer(vbox)

        self.accept_btn = accept_btn
        self.accept_btn.Enable(False)

        accept_btn.Bind(wx.EVT_BUTTON, self.on_accept)
        cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.process_btn.Bind(wx.EVT_BUTTON, self.on_process)
        self.mode_choice.Bind(wx.EVT_CHOICE, self.on_parameters_changed)
        self.saturation_checkbox.Bind(wx.EVT_CHECKBOX, self.on_toggle_saturation)
        self.inverse_variance_checkbox.Bind(wx.EVT_CHECKBOX, self.on_parameters_changed)
        self.rmin_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_parameters_changed)
        self.tmin_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_parameters_changed)

        self._used_files: List[str] = []
        self.avg: Optional[np.ndarray] = None
        self.avg_uncertainty: Optional[np.ndarray] = None
        self.pooled_atoms: Optional[np.ndarray] = None
        self.pooled_reference: Optional[np.ndarray] = None
        self.selected_mode = 0
        self.r_min_value = 0.0
        self.t_min_value = DEFAULT_T_MIN

        self._gauge_default_colour = self.progress.GetForegroundColour()
        self._processing = False
        self._dirty = True

        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.draw()
        self._show_idle_message()
        self._update_saturation_visibility()
        self.count_ctrl.SetFocus()
        self._reset_progress()

    def on_parameters_changed(self, event: Optional[wx.Event]) -> None:
        selection = self.mode_choice.GetSelection()
        if selection in self._mode_tooltips:
            self.mode_choice.SetToolTip(self._mode_tooltips[selection])
        self._mark_dirty()
        if event:
            event.Skip()

    def on_toggle_saturation(self, event: Optional[wx.Event]) -> None:
        self._update_saturation_visibility()
        self._mark_dirty()
        if event:
            event.Skip()

    def _update_saturation_visibility(self) -> None:
        show_controls = self.saturation_checkbox.GetValue()
        self.saturation_panel.Show(show_controls)
        self.saturation_panel.Enable(show_controls)
        parent = self.saturation_panel.GetParent()
        if parent:
            parent.Layout()

    def _reset_progress(self) -> None:
        if hasattr(self, "progress"):
            self.progress.SetForegroundColour(self._gauge_default_colour)
            self.progress.SetRange(1)
            self.progress.SetValue(0)

    def _prepare_progress(self, total: int) -> None:
        if not hasattr(self, "progress"):
            return
        steps = max(int(total), 1)
        self.progress.SetForegroundColour(self._gauge_default_colour)
        self.progress.SetRange(steps)
        self.progress.SetValue(0)

    def _update_progress(self, step: int, total: int) -> None:
        if not hasattr(self, "progress"):
            return
        if total <= 0:
            self.progress.Pulse()
        else:
            value = min(max(int(step), 0), self.progress.GetRange())
            self.progress.SetValue(value)
        wx.YieldIfNeeded()

    def _show_idle_message(self, message: Optional[str] = None) -> None:
        if not hasattr(self, "axes"):
            return
        self.axes.clear()
        self.axes.set_title(message or "Press Process to compute average")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.draw()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.avg = None
        self.avg_uncertainty = None
        self.pooled_atoms = None
        self.pooled_reference = None
        self._used_files = []
        self.accept_btn.Enable(False)
        self._reset_progress()
        self._show_idle_message()

    def on_process(self, event: Optional[wx.Event]) -> None:
        if self._processing:
            if event:
                event.Skip()
            return
        self._processing = True
        self.process_btn.Enable(False)
        self.accept_btn.Enable(False)
        self._reset_progress()
        self._show_idle_message("Processing…")
        wx.CallAfter(self.load_and_average)
        if event:
            event.Skip()

    def _get_float_from_ctrl(
        self, ctrl: Optional[wx.TextCtrl], default: Optional[float] = None, allow_negative: bool = False
    ) -> Optional[float]:
        if ctrl is None:
            return default
        text = ctrl.GetValue().strip()
        if not text:
            return default
        try:
            value = float(text)
        except ValueError:
            return default
        if not allow_negative and value < 0:
            return default
        return value

    def _effective_saturation_intensity(self) -> Optional[float]:
        """Return the effective saturation intensity in camera units if enabled."""

        if hasattr(self, "saturation_checkbox") and self.saturation_checkbox.GetValue():
            isat = self._get_float_from_ctrl(self.sat_isat_ctrl, default=None)
            if isat in (None, 0):
                return None

            gamma = self._get_float_from_ctrl(self.sat_gamma_ctrl, default=None)
            detuning = self._get_float_from_ctrl(
                self.sat_detuning_ctrl, default=0.0, allow_negative=True
            )

            if gamma in (None, 0):
                return isat

            return isat * (1.0 + 4.0 * (detuning ** 2) / (gamma ** 2))

        if not getattr(self.parent, "avg_apply_saturation_correction", False):
            return None

        isat = getattr(self.parent, "avg_saturation_intensity", None)
        if isat in (None, 0):
            return None

        try:
            isat = float(isat)
        except (TypeError, ValueError):
            return None

        gamma = getattr(self.parent, "avg_transition_gamma", None)
        detuning = getattr(self.parent, "avg_probe_detuning", 0.0)

        try:
            gamma = None if gamma is None else float(gamma)
            detuning = float(detuning)
        except (TypeError, ValueError):
            return isat

        if not gamma:
            return isat

        alpha = 1.0 + 4.0 * (detuning ** 2) / (gamma ** 2)
        return isat * alpha

    def _get_corrected_reference(self, image_data: Sequence[np.ndarray]) -> np.ndarray:
        """Return the dark-subtracted reference image, honoring defringing."""

        dark = np.asarray(image_data[2], dtype=np.float64)
        # Defringing has been removed from the processing pipeline, so always
        # use the raw probe-without-atoms reference.
        return np.asarray(image_data[1], dtype=np.float64) - dark

    def _apply_intensity_calibration(
        self,
        file_path: str,
        atoms_minus_dark: np.ndarray,
        reference_minus_dark: np.ndarray,
        image_data: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply an optional user-provided calibration to convert counts to intensity."""

        calibrator = getattr(self.parent, "avg_intensity_calibration", None)
        if not callable(calibrator):
            return atoms_minus_dark, reference_minus_dark

        try:
            result = calibrator(
                file_path,
                atoms_minus_dark,
                reference_minus_dark,
                image_data,
            )
        except Exception as exc:  # pragma: no cover - defensive, calibration is user supplied
            print(f"Average intensity calibration failed for {file_path}: {exc}")
            return atoms_minus_dark, reference_minus_dark

        if result is None:
            return atoms_minus_dark, reference_minus_dark

        if np.isscalar(result):
            scale = float(result)
            return atoms_minus_dark * scale, reference_minus_dark * scale

        if isinstance(result, (list, tuple)) and len(result) == 2:
            return result[0], result[1]

        return atoms_minus_dark, reference_minus_dark

    def _resolve_weights(
        self, files: Sequence[str], stack: np.ndarray
    ) -> Optional[np.ndarray]:
        """Obtain optional averaging weights supplied by the main window."""

        provider = getattr(self.parent, "avg_weight_provider", None)
        if not callable(provider):
            return None

        try:
            weights = provider(files, stack)
        except Exception as exc:  # pragma: no cover - defensive, provider is user supplied
            print(f"Average weight provider failed: {exc}")
            return None

        if weights is None:
            return None

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape == stack.shape:
            return weights

        if weights.ndim == 1 and weights.shape[0] == stack.shape[0]:
            return weights

        print("Average weight provider returned unexpected shape; ignoring weights")
        return None

    def load_and_average(self) -> None:
        """Load the most recent images and update the average preview."""
        if not self._processing:
            self._processing = True
            self.process_btn.Enable(False)
            self.accept_btn.Enable(False)
            self._reset_progress()
            self._show_idle_message("Processing…")

        total_files = 0

        try:
            self.parent.updateImageListBox()

            try:
                requested = int(self.count_ctrl.GetValue())
            except ValueError:
                requested = self.n
            self.n = max(1, requested)
            self.count_ctrl.ChangeValue(str(self.n))

            files = list(self.parent.fileList[-self.n:])
            total_files = len(files)
            self._prepare_progress(total_files)

            mode_index = self.mode_choice.GetSelection()
            if mode_index == wx.NOT_FOUND:
                mode_index = 0
                self.mode_choice.SetSelection(0)
            self.selected_mode = mode_index

            parsed_r_min = self._get_float_from_ctrl(self.rmin_ctrl, default=self.r_min_value)
            if parsed_r_min is None:
                r_min = self.r_min_value
            else:
                r_min = max(0.0, parsed_r_min)
                self.r_min_value = r_min
            self.rmin_ctrl.ChangeValue(f"{r_min:g}")

            parsed_t_min = self._get_float_from_ctrl(self.tmin_ctrl, default=self.t_min_value)
            if parsed_t_min is None:
                t_min = self.t_min_value
            else:
                t_min = min(max(parsed_t_min, 1e-12), 1.0)
                self.t_min_value = t_min
            self.tmin_ctrl.ChangeValue(f"{t_min:g}")

            eff_isat = self._effective_saturation_intensity()
            apply_saturation = eff_isat is not None
            if self.saturation_checkbox.GetValue() and not apply_saturation:
                print(
                    "Saturation correction enabled but parameters are incomplete; "
                    "skipping correction."
                )

            use_inverse_variance = self.inverse_variance_checkbox.GetValue()

            self.avg = None
            self.avg_uncertainty = None
            self.pooled_atoms = None
            self.pooled_reference = None

            od_images: List[np.ndarray] = []
            atoms_images: List[np.ndarray] = []
            reference_images: List[np.ndarray] = []
            valid_masks: List[np.ndarray] = []
            correction_terms: List[np.ndarray] = []
            used_files: List[str] = []

            for index, file_path in enumerate(reversed(files), start=1):
                self._update_progress(index, total_files)
                try:
                    hdulist = fits.open(file_path, memmap=False)
                except Exception as exc:
                    print(f"Failed to open {file_path}: {exc}")
                    continue

                with hdulist:
                    data = hdulist[0].data
                    if data is None or data.ndim != 3:
                        continue
                    atoms = np.asarray(data[0], dtype=np.float64)
                    reference = np.asarray(data[1], dtype=np.float64)
                    dark = np.asarray(data[2], dtype=np.float64)

                atoms_minus_dark = atoms - dark
                reference_minus_dark = self._get_corrected_reference(data)

                atoms_minus_dark, reference_minus_dark = self._apply_intensity_calibration(
                    file_path,
                    atoms_minus_dark,
                    reference_minus_dark,
                    data,
                )

                valid = reference_minus_dark > max(r_min, 0)
                transmission = np.full_like(atoms_minus_dark, np.nan)
                transmission[valid] = atoms_minus_dark[valid] / reference_minus_dark[valid]
                transmission[valid] = np.clip(transmission[valid], 1e-12, 1.0)

                od = np.full_like(transmission, np.nan)
                od[valid] = -np.log(transmission[valid])

                if apply_saturation:
                    correction = np.zeros_like(od, dtype=np.float64)
                    with np.errstate(invalid="ignore"):
                        correction[valid] = (
                            (atoms_minus_dark[valid] - reference_minus_dark[valid])
                            / eff_isat
                        )
                    correction_terms.append(correction)

                od_images.append(od)
                valid_masks.append(valid.astype(np.float64))
                atoms_images.append(atoms_minus_dark)
                reference_images.append(reference_minus_dark)
                used_files.append(file_path)

            self._used_files = used_files

            if not used_files:
                self._show_idle_message("No valid images to average")
                return

            od_stack = np.stack(
                [
                    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    for arr in od_images
                ],
                axis=0,
            )
            mask_stack = np.stack(valid_masks, axis=0)
            atoms_stack = np.stack(
                [
                    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    for arr in atoms_images
                ],
                axis=0,
            )
            reference_stack = np.stack(
                [
                    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    for arr in reference_images
                ],
                axis=0,
            )

            weights = None
            if use_inverse_variance:
                positive = mask_stack > 0
                denom = np.zeros_like(od_stack, dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom[positive] = (
                        np.reciprocal(np.maximum(atoms_stack[positive], 1e-12))
                        + np.reciprocal(np.maximum(reference_stack[positive], 1e-12))
                    )
                weights = np.zeros_like(od_stack, dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    weights[positive] = np.where(
                        denom[positive] > 0,
                        np.reciprocal(denom[positive]),
                        0.0,
                    )
            elif getattr(self.parent, "avg_weight_provider", None):
                weights = self._resolve_weights(used_files, od_stack)

            if weights is not None:
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                if weights.ndim == 3:
                    weighted_od = od_stack * weights
                    sum_weights = np.sum(weights * mask_stack, axis=0)
                    sum_od = np.sum(weighted_od, axis=0)
                else:
                    weight_array = weights.reshape((-1, 1, 1))
                    weighted_od = od_stack * weight_array
                    sum_weights = np.sum(weight_array * mask_stack, axis=0)
                    sum_od = np.sum(weighted_od, axis=0)

                avg = np.full(sum_od.shape, np.nan)
                valid = sum_weights > 0
                avg[valid] = sum_od[valid] / sum_weights[valid]

                uncertainty = np.full(sum_weights.shape, np.nan)
                uncertainty[valid] = 1.0 / np.sqrt(sum_weights[valid])
                self.avg_uncertainty = uncertainty
                self.avg = avg
                self.pooled_atoms = None
                self.pooled_reference = None
            elif self.selected_mode == 0:
                counts = np.sum(mask_stack, axis=0)
                sum_od = np.sum(od_stack, axis=0)
                avg = np.full(sum_od.shape, np.nan)
                positive = counts > 0
                avg[positive] = sum_od[positive] / counts[positive]
                self.avg_uncertainty = None

                self.avg = avg
                self.pooled_atoms = None
                self.pooled_reference = None
            else:
                atoms_sum = np.sum(atoms_stack, axis=0)
                reference_sum = np.sum(reference_stack, axis=0)
                valid_counts = np.sum(mask_stack, axis=0)
                valid = (reference_sum > 0) & (valid_counts > 0)

                transmission = np.full_like(atoms_sum, np.nan)
                transmission[valid] = atoms_sum[valid] / reference_sum[valid]
                transmission[valid] = np.clip(transmission[valid], t_min, 1.0)

                od_eff = np.full_like(transmission, np.nan)
                od_eff[valid] = -np.log(transmission[valid])

                if apply_saturation and correction_terms:
                    correction_total = np.zeros_like(od_eff, dtype=np.float64)
                    for corr in correction_terms:
                        correction_total += np.nan_to_num(
                            corr, nan=0.0, posinf=0.0, neginf=0.0
                        )
                    correction_total = np.where(valid, correction_total, np.nan)
                    od_eff = np.where(valid, od_eff + correction_total, np.nan)

                self.avg = od_eff
                self.avg_uncertainty = None
                self.pooled_atoms = np.where(valid, atoms_sum, np.nan)
                self.pooled_reference = np.where(valid, reference_sum, np.nan)

            display = np.ma.masked_invalid(self.avg)
            self.axes.clear()
            self.axes.imshow(display, cmap=cm.jet, origin="lower")
            titles = {
                0: "Average per-shot OD (Mean column density)",
                1: "Pooled transmission (Effective OD)",
            }
            self.axes.set_title(titles.get(self.selected_mode, "Average image"))
            self.axes.set_xticks([])
            self.axes.set_yticks([])
            self.canvas.draw()
            self._dirty = False

        finally:
            if total_files > 0:
                self.progress.SetValue(min(self.progress.GetRange(), total_files))
            else:
                self.progress.SetValue(0)
            self._processing = False
            self.process_btn.Enable(True)
            self.accept_btn.Enable(self.avg is not None)
            if self.avg is None:
                self._dirty = True

    def on_count_enter(self, event: Optional[wx.Event]) -> None:
        try:
            n_val = int(self.count_ctrl.GetValue())
        except ValueError:
            n_val = 1
        self.n = max(1, n_val)
        self.count_ctrl.ChangeValue(str(self.n))
        self._mark_dirty()
        if event:
            event.Skip()

    def on_accept(self, event: Optional[wx.Event]) -> None:
        if self.avg is not None:
            folder = self.parent.imageFolderPath.GetValue()
            if self._used_files:
                first = os.path.splitext(os.path.basename(self._used_files[0]))[0]
                last = os.path.splitext(os.path.basename(self._used_files[-1]))[0]
                filename = f"{first}-{last}.fits"
            else:
                filename = f"avg_{datetime.datetime.now():%Y%m%d_%H%M%S}.fits"
            absorb = np.exp(-self.avg)
            noatom = np.ones_like(self.avg)
            dark = np.zeros_like(self.avg)
            data = np.stack([absorb, noatom, dark])
            fits.PrimaryHDU(data).writeto(
                os.path.join(folder, filename), overwrite=True
            )
            self.parent.atomImage = self.avg
            self.parent.avg_preview_count = self.n
            self.parent.initializeAOI()
            self.parent.updateAOIImageAndProfiles()
            self.parent.updateFittingResults()
            self.parent.setAtomNumber()
        self.Close()

    def on_cancel(self, event: Optional[wx.Event]) -> None:
        self.Close()

    def on_close(self, event: Optional[wx.Event]) -> None:
        self.parent.avgFrame = None
        if hasattr(self.parent, "avgPreviewActive"):
            self.parent.avgPreviewActive = False
        if hasattr(self.parent, "avgPreviewBtn"):
            self.parent.avgPreviewBtn.Enable()
        if event:
            event.Skip()


__all__ = ["AvgPreviewFrame", "DEFAULT_T_MIN"]

