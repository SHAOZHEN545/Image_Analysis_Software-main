"""Utilities for building and applying an SVD-based background model."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from astropy.io import fits
from numpy.lib.stride_tricks import sliding_window_view

ProgressCallback = Callable[[int, int, str], None]


class SVDBasisError(RuntimeError):
    """Raised when the SVD basis cannot be constructed."""


@dataclass
class SVDLoadResult:
    """Statistics reported after loading a basis."""

    folder: str
    n_refs: int
    k: int
    singular_values: Sequence[float]
    normalization_factors: Sequence[float]
    basis_path: Optional[str] = None


_SVD_SERIAL_VERSION = 1


class SVDBasis:
    """Background synthesizer constructed from a stack of reference FITS files."""

    ATOM_MASK_THRESHOLD = 0.08
    MASK_DILATION_RADIUS = 2
    MIN_UNMASKED_FRACTION = 0.1

    def __init__(self) -> None:
        self.loaded: bool = False
        self.reference_folder: str = ""
        self.shape: Optional[tuple[int, int]] = None
        self.k: int = 0
        self.n_refs: int = 0
        self.U_aug: Optional[np.ndarray] = None
        self.mean_vec: Optional[np.ndarray] = None
        self.template_image: Optional[np.ndarray] = None
        self.singular_values: List[float] = []
        self.normalization_factors: List[float] = []
        self.last_synthesis_stats: Optional[dict[str, float]] = None
        self.basis_path: str = ""
        self.uses_dark_reference: bool = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.loaded = False
        self.reference_folder = ""
        self.shape = None
        self.k = 0
        self.n_refs = 0
        self.U_aug = None
        self.mean_vec = None
        self.template_image = None
        self.singular_values = []
        self.normalization_factors = []
        self.last_synthesis_stats = None
        self.basis_path = ""
        self.uses_dark_reference = False

    # ------------------------------------------------------------------
    def load_from_folder(
        self,
        folder: str,
        n_refs: int,
        k: int,
        *,
        cancel_event: Optional["threading.Event"] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SVDLoadResult:
        """Build the SVD basis from FITS files located in ``folder``.

        Parameters
        ----------
        folder:
            Directory containing 3-shot FITS stacks where HDU[1] is the
            reference (no-atom) image and HDU[2] is the dark image.
        n_refs:
            Maximum number of reference files to use.
        k:
            Number of SVD modes to retain.
        cancel_event:
            Optional ``threading.Event`` used to request cancellation.
        progress_callback:
            Optional callable invoked with ``(processed, target, path)`` as
            reference files are consumed.
        """

        if n_refs < 1:
            raise SVDBasisError("Number of reference FITS must be positive")
        if k < 1:
            raise SVDBasisError("Number of SVD modes must be positive")
        if not os.path.isdir(folder):
            raise SVDBasisError(f"Reference folder does not exist: {folder}")

        import threading  # late import to avoid cost when unused

        if cancel_event is None:
            cancel_event = threading.Event()

        candidates = sorted(
            [
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if name.lower().endswith((".fits", ".fit"))
            ]
        )
        if not candidates:
            raise SVDBasisError("No FITS files found in reference folder")

        eps = 1e-6
        refs: List[np.ndarray] = []
        shape: Optional[tuple[int, int]] = None
        normalization_factors: List[float] = []

        target = min(n_refs, len(candidates))

        dark_frames_used = 0
        skipped_missing_hdu = 0
        skipped_shape_mismatch = 0
        skipped_dark_mismatch = 0
        skipped_invalid_stack = 0

        for path in candidates:
            if cancel_event.is_set():
                raise SVDBasisError("SVD basis loading cancelled")
            if len(refs) >= n_refs:
                break
            try:
                with fits.open(path, memmap=False) as hdul:
                    no_atom: Optional[np.ndarray] = None
                    dark: Optional[np.ndarray] = None

                    if len(hdul) >= 2 and hdul[1].data is not None:
                        no_atom = np.asarray(hdul[1].data, dtype=np.float64)
                        if len(hdul) >= 3 and hdul[2].data is not None:
                            dark = np.asarray(hdul[2].data, dtype=np.float64)
                    else:
                        primary = hdul[0].data
                        if primary is None:
                            skipped_missing_hdu += 1
                            continue
                        stack = np.asarray(primary, dtype=np.float64)
                        if stack.ndim < 3 or stack.shape[0] < 2:
                            skipped_invalid_stack += 1
                            continue
                        no_atom = np.asarray(stack[1], dtype=np.float64)
                        if stack.shape[0] >= 3:
                            dark = np.asarray(stack[2], dtype=np.float64)
            except Exception as exc:  # pragma: no cover - corrupted files
                raise SVDBasisError(f"Failed to read '{path}': {exc}") from exc

            if dark is not None and no_atom.shape != dark.shape:
                skipped_dark_mismatch += 1
                continue

            if shape is None:
                shape = no_atom.shape
            elif shape != no_atom.shape:
                skipped_shape_mismatch += 1
                continue

            if dark is not None:
                dark_frames_used += 1

            if dark is None:
                ref = np.clip(no_atom, 1.0, None)
                dark = np.zeros_like(no_atom)
            else:
                ref = np.clip(no_atom - dark, 1.0, None)
            refs.append(ref)
            if progress_callback is not None:
                progress_callback(len(refs), target, path)

        if len(refs) < 5:
            details: List[str] = []
            if skipped_missing_hdu:
                details.append(
                    f"{skipped_missing_hdu} file(s) missing a usable no-atom image"
                )
            if skipped_shape_mismatch:
                details.append(
                    f"{skipped_shape_mismatch} file(s) with inconsistent image shape"
                )
            if skipped_dark_mismatch:
                details.append(
                    f"{skipped_dark_mismatch} file(s) with dark-frame shape mismatch"
                )
            if skipped_invalid_stack:
                details.append(
                    f"{skipped_invalid_stack} file(s) with legacy stacks lacking required planes"
                )
            suffix = f" ({'; '.join(details)})" if details else ""
            raise SVDBasisError(
                "Fewer than 5 valid reference FITS files were found in the selected folder"
                + suffix
            )

        data = np.stack(refs, axis=0)
        template = data.mean(axis=0)
        template_safe = np.clip(template, 1.0, None)

        for idx in range(data.shape[0]):
            ref = data[idx]
            ratios = ref / template_safe
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size == 0:
                alpha = 1.0
            else:
                alpha = float(np.median(ratios))
            alpha = max(alpha, eps)
            data[idx] = ref / alpha
            normalization_factors.append(alpha)

        template_norm = data.mean(axis=0)

        flat = data.reshape(data.shape[0], -1).T  # (M, N)
        mean_vec = flat.mean(axis=1, keepdims=True)
        centered = flat - mean_vec

        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        usable_modes = min(k, u.shape[1])
        u_k = u[:, :usable_modes]
        ones = np.ones((u_k.shape[0], 1), dtype=u_k.dtype)
        U_aug = np.hstack([u_k, ones])

        self.loaded = True
        self.reference_folder = folder
        self.shape = shape
        self.k = usable_modes
        self.n_refs = data.shape[0]
        self.U_aug = U_aug
        self.mean_vec = mean_vec.ravel()
        self.template_image = template_norm
        self.singular_values = s[:usable_modes].tolist()
        self.normalization_factors = normalization_factors
        self.last_synthesis_stats = None
        self.basis_path = ""
        self.uses_dark_reference = dark_frames_used > 0

        return SVDLoadResult(
            folder=folder,
            n_refs=self.n_refs,
            k=self.k,
            singular_values=self.singular_values,
            normalization_factors=tuple(self.normalization_factors),
            basis_path=None,
        )

    # ------------------------------------------------------------------
    def save_to_file(self, path: str) -> None:
        """Persist the loaded basis to ``path`` for later reuse."""

        if not self.loaded or self.U_aug is None or self.mean_vec is None:
            raise SVDBasisError("SVD basis has not been loaded")
        if self.shape is None:
            raise SVDBasisError("SVD basis shape is unknown")

        template = self.template_image
        if template is None:
            raise SVDBasisError("Template image missing from basis")

        np.savez_compressed(
            path,
            version=np.int64(_SVD_SERIAL_VERSION),
            shape=np.asarray(self.shape, dtype=np.int64),
            k=np.int64(self.k),
            n_refs=np.int64(self.n_refs),
            U_aug=self.U_aug,
            mean_vec=self.mean_vec,
            template_image=template,
            singular_values=np.asarray(self.singular_values, dtype=np.float64),
            normalization_factors=np.asarray(self.normalization_factors, dtype=np.float64),
            reference_folder=self.reference_folder,
            uses_dark_reference=np.array(self.uses_dark_reference, dtype=np.bool_),
        )
        self.basis_path = path

    # ------------------------------------------------------------------
    def load_from_file(self, path: str) -> SVDLoadResult:
        """Load a previously saved basis from ``path``."""

        if not os.path.isfile(path):
            raise SVDBasisError(f"SVD basis file does not exist: {path}")

        try:
            with np.load(path, allow_pickle=True) as data:
                version = int(data.get("version", 1))
                if version != _SVD_SERIAL_VERSION:
                    raise SVDBasisError(
                        f"Unsupported SVD basis version {version}; expected {_SVD_SERIAL_VERSION}"
                    )

                shape_arr = np.asarray(data["shape"], dtype=np.int64)
                if shape_arr.size != 2:
                    raise SVDBasisError("Serialized SVD basis has invalid shape metadata")
                shape = (int(shape_arr[0]), int(shape_arr[1]))

                U_aug = np.asarray(data["U_aug"], dtype=np.float64)
                mean_vec = np.asarray(data["mean_vec"], dtype=np.float64).ravel()
                template = np.asarray(data["template_image"], dtype=np.float64)
                singular_values = np.asarray(data.get("singular_values", []), dtype=np.float64)
                normalization = np.asarray(data.get("normalization_factors", []), dtype=np.float64)

                k_val = int(data.get("k", U_aug.shape[1] - 1))
                n_refs_val = int(data.get("n_refs", 0))
                ref_folder = ""
                if "reference_folder" in data:
                    ref_data = data["reference_folder"]
                    if isinstance(ref_data, np.ndarray) and ref_data.shape == ():
                        ref_folder = str(ref_data.item())
                    else:
                        ref_folder = str(ref_data)
                uses_dark = bool(data.get("uses_dark_reference", False))
        except SVDBasisError:
            raise
        except Exception as exc:  # pragma: no cover - corrupted files
            raise SVDBasisError(f"Failed to load SVD basis file: {exc}") from exc

        if U_aug.ndim != 2:
            raise SVDBasisError("Serialized SVD basis has invalid matrix dimensions")
        if mean_vec.size != U_aug.shape[0]:
            raise SVDBasisError("Serialized SVD basis has mismatched mean vector size")
        if template.shape != shape:
            raise SVDBasisError("Serialized SVD basis template has wrong shape")

        self.loaded = True
        self.reference_folder = ref_folder
        self.shape = shape
        self.k = int(k_val)
        self.n_refs = int(n_refs_val)
        self.U_aug = U_aug
        self.mean_vec = mean_vec
        self.template_image = template
        self.singular_values = singular_values.tolist()
        self.normalization_factors = normalization.tolist()
        self.last_synthesis_stats = None
        self.basis_path = path
        self.uses_dark_reference = uses_dark

        folder_meta = ref_folder or os.path.dirname(path)

        return SVDLoadResult(
            folder=folder_meta,
            n_refs=self.n_refs,
            k=self.k,
            singular_values=self.singular_values,
            normalization_factors=tuple(self.normalization_factors),
            basis_path=path,
        )

    # ------------------------------------------------------------------
    def synthesize_background(
        self,
        atom_img: np.ndarray,
        dark_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Synthesize a background image for the provided atom shot."""

        if not self.loaded or self.U_aug is None or self.mean_vec is None:
            raise SVDBasisError("SVD basis has not been loaded")
        if self.shape is None:
            raise SVDBasisError("SVD basis shape is unknown")

        atom_arr = np.asarray(atom_img, dtype=np.float64)
        if atom_arr.shape != self.shape:
            raise SVDBasisError(
                f"Atom image shape {atom_arr.shape} does not match basis {self.shape}"
            )

        subtract_dark = dark_img is not None and self.uses_dark_reference

        if subtract_dark:
            dark_arr = np.asarray(dark_img, dtype=np.float64)
            if dark_arr.shape != self.shape:
                raise SVDBasisError(
                    f"Dark image shape {dark_arr.shape} does not match basis {self.shape}"
                )
            working = atom_arr - dark_arr
        else:
            dark_arr = None
            working = atom_arr

        template = self.template_image
        if template is None:
            raise SVDBasisError("Template image missing from basis")
        template_safe = np.clip(template, 1.0, None)
        ratios = working / template_safe
        ratios = ratios[np.isfinite(ratios)]
        alpha = float(np.median(ratios)) if ratios.size else 1.0
        eps = 1e-6
        alpha = max(alpha, eps)
        normalized = working / alpha

        y_vec = normalized.reshape(-1)
        if y_vec.size != self.U_aug.shape[0]:
            raise SVDBasisError("Basis dimension does not match flattened image size")

        mean_vec = self.mean_vec
        centered = y_vec - mean_vec

        mask = self._compute_atom_mask(normalized)
        mask_fraction = 0.0

        if mask is not None:
            keep_mask = (~mask).reshape(-1)
            kept = int(np.count_nonzero(keep_mask))
            total = keep_mask.size
            min_required = max(
                int(self.MIN_UNMASKED_FRACTION * total), self.U_aug.shape[1]
            )
            if kept >= min_required:
                U_reduced = self.U_aug[keep_mask, :]
                centered_reduced = centered[keep_mask]
                coeffs, residuals, _, _ = np.linalg.lstsq(
                    U_reduced, centered_reduced, rcond=None
                )
                recon_centered = self.U_aug @ coeffs
                mask_fraction = 1.0 - (kept / total)
                residual_den = max(kept, 1)
                if residuals.size:
                    residual_rms = math.sqrt(float(residuals[0]) / residual_den)
                else:
                    diff_masked = centered_reduced - U_reduced @ coeffs
                    residual_rms = float(np.sqrt(np.mean(diff_masked**2)))
            else:
                mask = None

        if mask is None:
            coeffs, residuals, _, _ = np.linalg.lstsq(self.U_aug, centered, rcond=None)
            recon_centered = self.U_aug @ coeffs
            if residuals.size:
                residual_rms = math.sqrt(float(residuals[0]) / y_vec.size)
            else:
                diff = centered - recon_centered
                residual_rms = float(np.sqrt(np.mean(diff**2)))
            mask_fraction = 0.0

        recon = recon_centered + mean_vec
        recon = recon.reshape(self.shape)
        if subtract_dark and dark_arr is not None:
            recon = recon + dark_arr

        recon = np.clip(recon, 0.0, None)

        self.last_synthesis_stats = {
            "alpha": alpha,
            "residual_rms": residual_rms,
            "mask_fraction": mask_fraction,
        }

        return recon

    # ------------------------------------------------------------------
    def _compute_atom_mask(self, normalized: np.ndarray) -> Optional[np.ndarray]:
        """Estimate atom-affected pixels using an OD-style threshold."""

        template = self.template_image
        if template is None or template.shape != normalized.shape:
            return None

        template_safe = np.clip(template, 1.0, None)
        ratio = normalized / template_safe
        ratio = np.clip(ratio, 1e-3, None)
        od = -np.log(ratio)
        mask = od > self.ATOM_MASK_THRESHOLD
        if not np.any(mask):
            return None

        return self._dilate_mask(mask, self.MASK_DILATION_RADIUS)

    @staticmethod
    def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask

        pad = int(radius)
        padded = np.pad(mask, pad_width=pad, mode="constant", constant_values=False)
        window_shape = (2 * pad + 1, 2 * pad + 1)
        views = sliding_window_view(padded, window_shape)
        return views.any(axis=(-2, -1))


__all__ = ["SVDBasis", "SVDBasisError", "SVDLoadResult"]
