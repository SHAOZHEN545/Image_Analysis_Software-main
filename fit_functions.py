import numpy as np
from scipy.optimize import curve_fit
from constant_v6 import kB
from imgFunc_v7 import calculate_phase_space_density, calculate_peak_density


def linear(x, m, b):
    """Simple linear fit."""
    return m * x + b


def quadratic(x, a, b, c):
    """Quadratic fit."""
    return a * x ** 2 + b * x + c


def exponential(x, A, B, C):
    """Generic exponential fit with decaying exponent."""
    return A * np.exp(-B * x) + C


def damped_ho(t, A, gamma, omega, phi, C):
    """Generic damped harmonic oscillator."""
    return C + A * np.exp(-gamma * t) * np.cos(omega * t + phi)


def mot_com_y(t, z_eq, B, gamma, omega, phi):
    """Damped harmonic oscillator for MOT COM_y."""
    return z_eq + B * np.exp(-gamma * t) * np.cos(omega * t + phi)


def gaussian(x, A, mu, sigma, C):
    """Gaussian profile."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C


def lorentzian(x, A, x0, gamma, C):
    """Lorentzian profile with \(\gamma\) equal to the full width at half maximum."""
    half_gamma = gamma / 2.0 if gamma != 0 else np.finfo(float).tiny
    return C + A / (1.0 + ((x - x0) / half_gamma) ** 2)


def inverse(x, A, h, B):
    """Shifted inverse relation."""
    return A / (x - h) + B


def _nan_lifetime_metrics():
    """Return placeholder values when the lifetime fit fails."""

    nan = float("nan")
    return {
        "Lifetime (s)": nan,
        "Decay rate (1/s)": nan,
        "Lifetime (s)_err": nan,
        "N0": nan,
        "N0_err": nan,
        "Offset": nan,
        "Offset_err": nan,
    }


def _derive_mot_lifetime(popt, pcov):
    """Return lifetime metrics and uncertainty."""

    tau = popt[1]
    lifetime_err = float(np.sqrt(pcov[1, 1])) if pcov is not None else float("nan")
    decay_rate = 1 / tau if tau != 0 else np.inf
    return {
        "Lifetime (s)": tau,
        "Decay rate (1/s)": decay_rate,
        "Lifetime (s)_err": lifetime_err,
    }


def _mot_lifetime_model(t, N0, tau, C):
    """Return the MOT lifetime model evaluated at ``t``."""

    return N0 * np.exp(-t / tau) + C


def fit_multi_run_mot_lifetime(var_values, results):
    """Return fitted parameters for a multi-run MOT lifetime dataset.

    Parameters
    ----------
    var_values:
        Sequence of time values associated with the MOT lifetime measurement.
    results:
        Mapping containing at least the ``"Atom Number"`` column produced by the
        fitting pipeline.

    Returns
    -------
    dict | None
        ``None`` if the fit cannot be performed. Otherwise a dictionary with
        the sorted ``times`` and ``atom_counts`` arrays, along with the fitted
        parameters ``popt`` and covariance matrix ``pcov``.
    """

    if not var_values or not results:
        return None

    try:
        times = np.asarray(var_values, dtype=float)
    except Exception:
        return None
    try:
        atom_counts = np.asarray(results.get("Atom Number", []), dtype=float)
    except Exception:
        return None

    mask = np.isfinite(times) & np.isfinite(atom_counts)
    times = times[mask]
    atom_counts = atom_counts[mask]

    if len(times) < 3 or len(atom_counts) != len(times):
        return None

    order = np.argsort(times)
    times = times[order]
    atom_counts = atom_counts[order]

    offset_guess = float(np.min(atom_counts)) if atom_counts.size else 0.0
    amplitude_guess = float(np.max(atom_counts) - offset_guess)
    if not np.isfinite(amplitude_guess) or amplitude_guess <= 0:
        amplitude_guess = float(np.max(atom_counts)) if atom_counts.size else 1.0
    if amplitude_guess <= 0 or not np.isfinite(amplitude_guess):
        amplitude_guess = 1.0

    if len(times) >= 2 and np.any(atom_counts > offset_guess):
        span = times[-1] - times[0]
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        y1 = atom_counts[0]
        y2 = atom_counts[-1]
        if y2 <= offset_guess:
            y2 = offset_guess + 1e-9
        decay_guess = np.log((y1 - offset_guess) / (y2 - offset_guess)) / span
        if not np.isfinite(decay_guess) or decay_guess <= 0:
            decay_guess = 1.0
    else:
        decay_guess = 1.0
    tau_guess = 1.0 / decay_guess if decay_guess > 0 else 1.0
    if not np.isfinite(tau_guess) or tau_guess <= 0:
        tau_guess = 1.0

    try:
        popt, pcov = curve_fit(
            _mot_lifetime_model,
            times,
            atom_counts,
            p0=(amplitude_guess, tau_guess, offset_guess),
            bounds=((0.0, 1e-9, -np.inf), (np.inf, np.inf, np.inf)),
            maxfev=20000,
        )
    except Exception:
        return None

    return {
        "times": times,
        "atom_counts": atom_counts,
        "popt": popt,
        "pcov": pcov,
    }


def _multi_run_mot_lifetime(var_values, results):
    """Fit lifetime data across a multi-run sequence."""

    fit = fit_multi_run_mot_lifetime(var_values, results)
    if not fit:
        return _nan_lifetime_metrics()

    popt = fit["popt"]
    pcov = fit["pcov"]

    derived = _derive_mot_lifetime(popt, pcov)
    derived.update(
        {
            "N0": float(popt[0]),
            "N0_err": float(np.sqrt(pcov[0, 0])) if pcov is not None else float("nan"),
            "Offset": float(popt[2]),
            "Offset_err": float(np.sqrt(pcov[2, 2])) if pcov is not None else float("nan"),
        }
    )
    return derived


def _mot_lifetime_analysis(popt, _data, var_values, results, _parent, pcov=None):
    """Dispatch lifetime analysis for single and multi-run modes."""

    if popt is not None:
        return _derive_mot_lifetime(popt, pcov)
    return _multi_run_mot_lifetime(var_values, results)


FIT_FUNCTIONS = {
    "Linear": {
        "func": linear,
        "p0": (1.0, 0.0),
        "title": "Linear Fit",
        "param_names": ["slope", "intercept"],
        "formula": r"$y = m x + b$",
        "x_label": None,
        "x_unit": None,
        "y_label": None,
        "y_unit": None,
        "derived": lambda *a, **k: {},
    },
    "Quadratic": {
        "func": quadratic,
        "p0": (1.0, 0.0, 0.0),
        "title": "Quadratic Fit",
        "param_names": ["a", "b", "c"],
        "formula": r"$y = a x^2 + b x + c$",
        "x_label": None,
        "x_unit": None,
        "y_label": None,
        "y_unit": None,
        "derived": lambda *a, **k: {},
    },
    "Exponential": {
        "func": exponential,
        "p0": (1.0, 1.0, 0.0),
        "title": "Exponential Fit",
        "param_names": ["A", "B", "C"],
        "formula": r"$y = A e^{-B x} + C$",
        "x_label": None,
        "x_unit": None,
        "y_label": None,
        "y_unit": None,
        "derived": lambda *a, **k: {},
    },
    "Damped H.O.": {
        "func": damped_ho,
        "p0": (1.0, 0.1, 1.0, 0.0, 0.0),
        "title": "Damped Harmonic Oscillator",
        "param_names": ["A", "gamma", "omega", "phi", "C"],
        "formula": r"$y = C + A e^{-\gamma x} \cos(\omega x + \phi)$",
        "x_label": None,
        "x_unit": None,
        "y_label": None,
        "y_unit": None,
        "derived": lambda *a, **k: {},
    },
    "Gaussian": {
        "func": gaussian,
        "p0": (1.0, 0.0, 1.0, 0.0),
        "title": "Gaussian Fit",
        "param_names": ["A", "mu", "sigma", "C"],
        "formula": r"$y = A e^{-(x-\mu)^2/(2\sigma^2)} + C$",
        "x_label": None,
        "x_unit": None,
        "y_label": None,
        "y_unit": None,
        "derived": lambda popt, *a, **k: {"FWHM": 2.354820045 * popt[2]},
    },
    "Lorentzian": {
        "func": lorentzian,
        "p0": (1.0, 0.0, 1.0, 0.0),
        "title": "Lorentzian Fit",
        "param_names": ["A", "x0", "gamma (FWHM)", "C"],
        "formula": r"$y = C + \frac{A}{1 + \left(\frac{x-x_0}{\gamma/2}\right)^2}$",
        "x_label": None,
        "x_unit": None,
        "y_label": None,
        "y_unit": None,
        "derived": lambda popt, *a, **k: {"FWHM": popt[2]},
    },
    "Inverse": {
        "func": inverse,
        "p0": (1.0, 0.0, 0.0),
        "title": "Inverse Fit",
        "param_names": ["A", "h", "B"],
        "formula": r"$y = \frac{A}{x-h} + B$",
        "x_label": None,
        "x_unit": None,
        "y_label": None,
        "y_unit": None,
        "derived": lambda *a, **k: {},
    },
    "Temperature, Density, and PSD": {
        "func": linear,  # Fit performed on \(t^2\) vs \(\sigma^2\)
        "p0": (1.0, 0.0),
        "title": "Temperature, Density, and PSD",
        "param_names": ["slope", "intercept"],
        "formula": r"$\sigma^2 = \beta t^2 + \sigma_0^2$",
        "x_label": r"$t^2$",
        "x_unit": None,
        "y_label": r"$\sigma^2$",
        "y_unit": r"$\mu$m$^2$",
        "derived": lambda popt, data, vars, res, parent, pcov=None: _derive_temp_psd(vars, res, parent),
    },
    "MOT Lifetime": {
        "func": lambda x, N0, tau, C: N0 * np.exp(-x / tau) + C,
        "p0": (1.0, 1.0, 0.0),
        "title": "MOT Lifetime",
        "param_names": ["N0", "tau", "offset"],
        "formula": r"$N = N_0 e^{-t/\tau} + C$",
        "derived": _mot_lifetime_analysis,
    },
    "MOT Ringdown": {
        "func": mot_com_y,
        "p0": (0.0, 0.0, 1.0, 1.0, 0.0),
        "title": "MOT Ringdown",
        "param_names": ["z_eq", "B", "gamma", "omega", "phi"],
        "formula": r"$z = z_{eq} + B e^{-\gamma t} \cos(\omega t + \phi)$",
        "x_label": "t",
        "x_unit": "s",
        "y_label": "z",
        "y_unit": "m",
        "derived": lambda popt, *a, pcov=None, **k: {
            "Damping time (s)": 1 / popt[2] if popt[2] != 0 else np.inf,
            "Frequency (Hz)": popt[3] / (2 * np.pi),
            "Q": popt[3] / (2 * popt[2]) if popt[2] != 0 else np.inf,
        },
    },
}

def get_temp_psd_analysis(var_values, results, parent):
    """Return detailed analysis for the temperature/PSD multi-fit."""

    if var_values is None or len(var_values) == 0 or not results:
        return {}

    t_s = np.square(np.asarray(var_values))
    sx = np.asarray(results.get("x-True Width", []))
    sy = np.asarray(results.get("y-True Width", []))
    sx_sq = np.square(sx)
    sy_sq = np.square(sy)

    if len(t_s) < 2 or len(sx_sq) != len(t_s) or len(sy_sq) != len(t_s):
        return {}

    sx_err = np.asarray(results.get("x-True Width Std", []))
    sy_err = np.asarray(results.get("y-True Width Std", []))

    w_x = w_y = None
    sig_sq_err_x = sig_sq_err_y = None
    if len(sx_err) == len(sx_sq):
        sig_sq_err_x = 2 * sx * sx_err
        w_x = np.where(sig_sq_err_x > 0, 1 / sig_sq_err_x, 1)
    if len(sy_err) == len(sy_sq):
        sig_sq_err_y = 2 * sy * sy_err
        w_y = np.where(sig_sq_err_y > 0, 1 / sig_sq_err_y, 1)

    if w_x is not None:
        px, cov_x = np.polyfit(t_s, sx_sq, 1, w=w_x, cov=True)
    else:
        px, cov_x = np.polyfit(t_s, sx_sq, 1, cov=True)
    if w_y is not None:
        py, cov_y = np.polyfit(t_s, sy_sq, 1, w=w_y, cov=True)
    else:
        py, cov_y = np.polyfit(t_s, sy_sq, 1, cov=True)

    slope_x, intercept_x = px
    slope_y, intercept_y = py
    intercept_x = float(intercept_x)
    intercept_y = float(intercept_y)

    temp_x = slope_x * parent.mass / kB * 1e6  # microKelvin
    temp_y = slope_y * parent.mass / kB * 1e6
    temp_x_err = np.sqrt(cov_x[0, 0]) * parent.mass / kB * 1e6
    temp_y_err = np.sqrt(cov_y[0, 0]) * parent.mass / kB * 1e6
    atom_means = []

    x_atom_values = np.asarray(results.get("x-Atom Number", []), dtype=float)
    if x_atom_values.size:
        finite_x = x_atom_values[np.isfinite(x_atom_values)]
        if finite_x.size:
            atom_means.append(float(np.mean(finite_x)))

    y_atom_values = np.asarray(results.get("y-Atom Number", []), dtype=float)
    if y_atom_values.size:
        finite_y = y_atom_values[np.isfinite(y_atom_values)]
        if finite_y.size:
            atom_means.append(float(np.mean(finite_y)))

    if atom_means:
        avg_atom = float(np.mean(atom_means))
    else:
        total_atoms = np.asarray(results.get("Atom Number", [0]), dtype=float)
        finite_total = total_atoms[np.isfinite(total_atoms)]
        avg_atom = float(np.mean(finite_total)) if finite_total.size else 0.0

    intercept_x_clamped = max(intercept_x, 0.0)
    intercept_y_clamped = max(intercept_y, 0.0)
    sigma_x0 = np.sqrt(intercept_x_clamped)
    sigma_y0 = np.sqrt(intercept_y_clamped)
    intercept_x_err = np.sqrt(cov_x[1, 1])
    intercept_y_err = np.sqrt(cov_y[1, 1])
    sigma_x0_err = 0.5 * intercept_x_err / sigma_x0 if sigma_x0 > 0 else 0.0
    sigma_y0_err = 0.5 * intercept_y_err / sigma_y0 if sigma_y0 > 0 else 0.0

    psd, psd_err = calculate_phase_space_density(
        avg_atom,
        temp_x * 1e-6,
        temp_y * 1e-6,
        parent.mass,
        sigma_x0,
        sigma_y0,
        sigma_x_err=sigma_x0_err,
        sigma_y_err=sigma_y0_err,
        temp_x_err_K=temp_x_err * 1e-6,
        temp_y_err_K=temp_y_err * 1e-6,
    )

    sigma_z0 = sigma_x0  # assume symmetry along z
    density, density_err = calculate_peak_density(
        avg_atom,
        sigma_x0,
        sigma_y0,
        sigma_z0,
        sigma_x_err=sigma_x0_err,
        sigma_y_err=sigma_y0_err,
        sigma_z_err=sigma_x0_err,
    )

    return {
        "t_squared": t_s,
        "sigma_x_squared": sx_sq,
        "sigma_y_squared": sy_sq,
        "sigma_x_squared_err": sig_sq_err_x,
        "sigma_y_squared_err": sig_sq_err_y,
        "slope_x": float(slope_x),
        "slope_y": float(slope_y),
        "intercept_x": intercept_x,
        "intercept_y": intercept_y,
        "cov_x": cov_x,
        "cov_y": cov_y,
        "derived": {
            "Temp_x (µK)": temp_x,
            "Temp_x_err (µK)": temp_x_err,
            "Temp_y (µK)": temp_y,
            "Temp_y_err (µK)": temp_y_err,
            "Intercept_x (µm^2)": intercept_x,
            "Intercept_y (µm^2)": intercept_y,
            "PSD": psd,
            "PSD_err": psd_err,
            "Density (cm^-3)": density,
            "Density (cm^-3)_err": density_err,
        },
    }


def _derive_temp_psd(var_values, results, parent):
    """Compute temperatures and PSD from TOF data."""

    analysis = get_temp_psd_analysis(var_values, results, parent)
    if not analysis:
        return {}
    return analysis["derived"]
