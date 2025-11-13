import math
import os
from datetime import date

import pytest

np = pytest.importorskip("numpy")
fits = pytest.importorskip("astropy.io.fits")
wx = pytest.importorskip("wx")

from svd_basis import SVDBasis
from thorcam_window import CameraPanel


def _write_reference_stack(folder, idx, atom, no_atom, dark=None, *, stacked=False):
    path = os.path.join(folder, f"ref_{idx:03d}.fits")
    if stacked:
        planes = [atom.astype(np.uint16), no_atom.astype(np.uint16)]
        if dark is not None:
            planes.append(dark.astype(np.uint16))
        stack = np.stack(planes, axis=0)
        fits.PrimaryHDU(stack).writeto(path, overwrite=True)
    else:
        hdus = [
            fits.PrimaryHDU(atom.astype(np.uint16)),
            fits.ImageHDU(no_atom.astype(np.uint16)),
        ]
        if dark is not None:
            hdus.append(fits.ImageHDU(dark.astype(np.uint16)))
        hdul = fits.HDUList(hdus)
        hdul.writeto(path, overwrite=True)
    return path


def _generate_reference_dataset(tmp_path, count=12, shape=(16, 16), *, stacked=False):
    folder = tmp_path / "refs"
    folder.mkdir()
    y = np.linspace(0.0, 1.0, shape[0]).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1)
    base = 1200.0 + 60.0 * x + 45.0 * y
    feature1 = 25.0 * x
    feature2 = 18.0 * y
    feature3 = 12.0 * np.sin(2 * math.pi * x)
    dark = np.full(shape, 80.0)
    for i in range(count):
        no_atom = (
            base
            + (10.0 * math.sin(i / 3.0)) * feature1
            + (8.0 * math.cos(i / 4.0)) * feature2
            + (4.0 + i % 4) * feature3
        )
        atom = no_atom - 5.0  # placeholder; atom channel unused for training
        _write_reference_stack(folder, i, atom, no_atom, dark, stacked=stacked)
    return folder


def _generate_reference_dataset_no_dark(tmp_path, count=8, shape=(12, 12)):
    folder = tmp_path / "refs_nodark"
    folder.mkdir()
    base = np.full(shape, 1500.0)
    gradient = np.linspace(-20.0, 20.0, shape[0]).reshape(-1, 1)
    for i in range(count):
        offset = 15.0 * math.sin(i / 2.0)
        no_atom = base + offset + gradient
        atom = no_atom - 3.0
        _write_reference_stack(folder, i, atom, no_atom, dark=None, stacked=False)
    return folder


def test_svd_basis_reconstructs_background(tmp_path):
    folder = _generate_reference_dataset(tmp_path)
    basis = SVDBasis()
    result = basis.load_from_folder(str(folder), n_refs=10, k=4)
    assert basis.loaded
    assert result.n_refs == 10
    assert basis.shape == (16, 16)

    y = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, 16).reshape(1, -1)
    base = 1200.0 + 60.0 * x + 45.0 * y
    feature1 = 25.0 * x
    feature2 = 18.0 * y
    feature3 = 12.0 * np.sin(2 * math.pi * x)
    true_bg = base + 12.0 * feature1 - 6.5 * feature2 + 5.0 * feature3
    dark = np.full((16, 16), 80.0)
    gaussian = np.exp(-(((x - 0.45) ** 2) + ((y - 0.55) ** 2)) / 0.01)
    atom = true_bg - 90.0 * gaussian

    synth = basis.synthesize_background(atom, dark)
    assert synth.shape == true_bg.shape
    diff = np.abs(synth - true_bg)
    assert float(np.mean(diff)) < 5.0
    assert float(np.max(diff)) < 35.0
    stats = basis.last_synthesis_stats
    assert stats is not None
    assert stats["alpha"] > 0
    assert stats["residual_rms"] >= 0
    assert 0.0 < stats["mask_fraction"] < 0.5


def test_svd_basis_serialization_roundtrip(tmp_path):
    folder = _generate_reference_dataset(tmp_path)
    basis = SVDBasis()
    basis.load_from_folder(str(folder), n_refs=6, k=3)

    basis_dir = tmp_path / "basis"
    basis_dir.mkdir()
    basis_path = basis_dir / "saved_basis.npz"
    basis.save_to_file(str(basis_path))
    assert os.path.exists(basis_path)
    assert basis.basis_path == str(basis_path)

    loaded = SVDBasis()
    result = loaded.load_from_file(str(basis_path))
    assert loaded.loaded
    assert result.basis_path == str(basis_path)
    assert loaded.shape == basis.shape
    assert loaded.k == basis.k
    assert loaded.n_refs == basis.n_refs
    assert loaded.reference_folder == basis.reference_folder


def test_svd_basis_handles_missing_dark_frames(tmp_path):
    folder = _generate_reference_dataset_no_dark(tmp_path)
    basis = SVDBasis()
    result = basis.load_from_folder(str(folder), n_refs=10, k=3)
    assert basis.loaded
    assert result.n_refs == 8
    assert not basis.uses_dark_reference

    atom = np.full((12, 12), 1500.0) + np.linspace(-20.0, 20.0, 12).reshape(-1, 1)
    dark = np.full((12, 12), 25.0)
    synth = basis.synthesize_background(atom, dark)
    assert synth.shape == atom.shape
    assert np.isfinite(synth).all()
    stats = basis.last_synthesis_stats
    assert stats is not None
    assert 0.0 <= stats["mask_fraction"] < 1.0


def test_save_fits_single_svd(tmp_path):
    folder = _generate_reference_dataset(tmp_path)
    basis = SVDBasis()
    basis.load_from_folder(str(folder), n_refs=8, k=3)

    basis_dir = tmp_path / "basis"
    basis_dir.mkdir()
    basis_file = basis_dir / "test_svd.npz"
    basis.save_to_file(str(basis_file))

    y = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, 16).reshape(1, -1)
    base = 1200.0 + 60.0 * x + 45.0 * y
    feature1 = 25.0 * x
    feature2 = 18.0 * y
    feature3 = 12.0 * np.sin(2 * math.pi * x)
    true_bg = base + 9.0 * feature1 + 3.5 * feature2 - 4.0 * feature3
    dark = np.full((16, 16), 80.0)
    atom = true_bg - 50.0 * np.exp(-(((x - 0.4) ** 2) + ((y - 0.6) ** 2)) / 0.02)
    bg = basis.synthesize_background(atom, dark)

    app = wx.App(False)
    frame = wx.Frame(None)
    panel = CameraPanel(frame, sdk=None, idx=1)
    try:
        panel.run_mode = "Hardware Trigger"
        panel.capture_mode = "Single SVD"
        panel.svd_basis = basis
        panel.svd_reference_folder = str(folder)
        panel.svd_basis_path = basis.basis_path
        panel.save_folder = str(tmp_path / "output")
        panel.today = date.today()
        os.makedirs(panel.save_folder, exist_ok=True)
        path = panel.save_fits([atom, bg, dark], error=False)
    finally:
        panel.Destroy()
        frame.Destroy()
        app.Destroy()

    assert path is not None
    with fits.open(path) as hdul:
        assert len(hdul) == 1
        data = hdul[0].data
        assert data.shape == (3, 16, 16)
        assert data.dtype == np.float32
        header = hdul[0].header
        assert header["IMGMODE"] == "SingleSVD"
        assert header["SVDK"] == basis.k
        assert header["SVDNREF"] == basis.n_refs
        assert header["SVDNORM"] == "median_ratio_to_mean"
        assert header["SVDBFILE"] == str(basis_file)
        assert "SVDALPHA" in header
        assert "SVDRMS" in header
        assert "SVDMASK" in header
        stats = basis.last_synthesis_stats
        assert stats is not None
        assert 0.0 <= stats["mask_fraction"] < 1.0


def test_svd_basis_supports_legacy_primary_stack(tmp_path):
    folder = _generate_reference_dataset(tmp_path, stacked=True)
    basis = SVDBasis()
    result = basis.load_from_folder(str(folder), n_refs=9, k=4)
    assert basis.loaded
    assert result.n_refs == 9
    assert basis.uses_dark_reference

    y = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, 16).reshape(1, -1)
    base = 1200.0 + 60.0 * x + 45.0 * y
    feature1 = 25.0 * x
    feature2 = 18.0 * y
    feature3 = 12.0 * np.sin(2 * math.pi * x)
    true_bg = base + 6.0 * feature1 - 4.0 * feature2 + 7.0 * feature3
    dark = np.full((16, 16), 80.0)
    gaussian = np.exp(-(((x - 0.5) ** 2) + ((y - 0.4) ** 2)) / 0.015)
    atom = true_bg - 70.0 * gaussian

    synth = basis.synthesize_background(atom, dark)
    diff = np.abs(synth - true_bg)
    assert float(np.mean(diff)) < 6.0
    assert float(np.max(diff)) < 40.0
    stats = basis.last_synthesis_stats
    assert stats is not None
    assert 0.0 <= stats["mask_fraction"] < 0.6
