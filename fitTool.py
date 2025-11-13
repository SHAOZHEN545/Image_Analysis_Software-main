'''Common fitting routines.'''

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import operator
from polylog import *
import time
import warnings
from constant_v6 import *


def condensate1DDist(data, x0, thermal_sigma, thermal_amp, TF_radius, BEC_amp, offset, slope):
    thermal = thermal_amp * np.exp(-(data - x0)**2/(2. * thermal_sigma**2)) 
    condensate = BEC_amp * np.maximum((1 -  ((data - x0)/TF_radius)**2), 0)
    return thermal + condensate + offset + data * slope
    
def pureCondensate1D(data, x0, TF_radius, BEC_amp, offset, slope):
    return BEC_amp * np.maximum((1 - ((data-x0)/TF_radius)**2), 0)

    
def fermion1DDist(data, x0, sigma, amplitude, offset, slope, q):
    
    tmp = q - ((data - x0)**2/sigma**2)* f(np.exp(q))
    numerator = fermi_poly5half(tmp)
    denumerator = fermi_poly5half(q)

    dist = offset + amplitude * numerator/denumerator + data * slope     
    return dist

def pureDegenerateFermion1D(data, x0, sigma, amp, offset, slope):
    return offset + amp * (1 - (data - x0)**2/sigma**2)**(5./2) + slope * data


def single_G(x, A, center, sigma, offset):
    return A*np.exp(-1*((x-center)**2)/(2*sigma**2)) + offset

def single_G_plus_slope(x, A, center, sigma, offset, slope):
    return A*np.exp(-1*((x-center)**2)/(2*sigma**2)) + offset + slope * x

def jac_single_G_plus_slope(x, A, center, sigma, offset, slope):
    """Analytic Jacobian of :func:`single_G_plus_slope`."""

    x = np.asarray(x)
    # Prevent divide-by-zero warnings when sigma is extremely small.
    sigma = np.asarray(sigma)
    safe_sigma = np.where(sigma == 0, np.finfo(float).tiny, sigma)

    gaussian = np.exp(-((x - center) ** 2) / (2 * safe_sigma ** 2))
    dA = gaussian
    dcenter = A * gaussian * (x - center) / (safe_sigma ** 2)
    dsigma = A * gaussian * ((x - center) ** 2) / (safe_sigma ** 3)
    doffset = np.ones_like(x)
    dslope = x

    return np.column_stack((dA, dcenter, dsigma, doffset, dslope))

def gaussianFit(
    xBasis,
    x_summed,
    aoi,
    axis='x',
    fitBounds=None,
    max_od=None,
    use_analytic_jacobian=True,
    window_sigma_multiplier=5.0,
    max_fit_points=None,
    fit_tails_only=False,
    tail_fraction=0.5,
    tail_side=None,
    tail_sigma_multiplier=2.0,
    min_tail_points=10,
):
    """Fit a 1D Gaussian to the provided data.

    Parameters
    ----------
    xBasis : ndarray
        Basis over which the data was measured.
    x_summed : ndarray
        Summed data along the chosen axis.
    aoi : tuple
        ((x_start, y_start), (x_end, y_end)) region of interest.
    axis : {'x', 'y'}, optional
        Axis along which the fit is performed.
    fitBounds : tuple, optional
        Optional bounds for the fit.
    max_od : float, optional
        Unused. Retained for backward compatibility.

    window_sigma_multiplier : float, optional
        Number of Gaussian sigmas (in pixel units) to keep on either side of the
        detected peak when windowing the data before fitting. Defaults to ``5``.
    max_fit_points : int, optional
        If provided, uniformly subsample the windowed data down to at most this
        many points before calling :func:`curve_fit`.
    fit_tails_only : bool, optional
        When ``True``, perform the Gaussian fit using only the low-density tails
        of the distribution. The fitter first solves for the best-fit Gaussian
        using the full dataset, then derives a mask from that solution. Samples
        are retained if they fall below ``baseline + tail_fraction * amplitude``
        or lie beyond ``tail_sigma_multiplier`` times the fitted width from the
        peak center. The optimizer is re-run on the masked data (falling back to
        the full profile if too few samples remain) while holding the previously
        fitted offset and slope fixed so the baseline stays anchored, and the
        fitted curve is still evaluated over the full ``xBasis``.
    tail_fraction : float, optional
        Fraction of the peak amplitude above the baseline used to determine the
        cutoff for tail selection. Defaults to ``0.5``.
    tail_sigma_multiplier : float, optional
        Number of Gaussian sigmas from the estimated center that must be
        exceeded for a point to be considered part of the tail. Defaults to
        ``2``.
    min_tail_points : int, optional
        Minimum number of samples required for the tail-only mask. If the mask
        produces fewer points the full dataset is used instead. Defaults to
        ``10``.
    tail_side : {'left', 'right', 'top', 'bottom', None}, optional
        Restrict the tail selection to a single side of the profile. When a
        side is specified the retained samples are mirrored about the fitted
        center to preserve the assumption of symmetry. If the resulting mask
        does not contain enough samples the original two-sided behaviour is
        used instead.

    Returns
    -------
    tuple
        Optimized parameters if the fit succeeds, otherwise ``None``.
    """

    isFitSuccessful = False

    mask = np.isfinite(x_summed)
    xBasis_fit = xBasis[mask]
    x_summed_fit = x_summed[mask]
    if xBasis_fit.size == 0:
        warnings.warn("All data points are NaN; fit aborted")
        return 0., 0., 0., 0., [], False, 0., np.zeros(5)

    N_smoothing = 10
    min_sigma_pixels = 300.0

    x_summed_smooth = np.convolve(x_summed_fit, np.ones((N_smoothing,)) / N_smoothing, mode='same')

    def widthFinder(half_max_value, center, data):
        """Estimate the Gaussian sigma from the half-maximum crossings."""

        mask = data >= half_max_value
        if not np.any(mask):
            return min_sigma_pixels

        indices = np.nonzero(mask)[0]
        left_candidates = indices[indices <= center]
        right_candidates = indices[indices >= center]

        if left_candidates.size == 0 or right_candidates.size == 0:
            left_idx = int(indices[0])
            right_idx = int(indices[-1])
        else:
            left_idx = int(left_candidates[-1])
            right_idx = int(right_candidates[0])

        if left_idx == right_idx:
            return min_sigma_pixels

        def interpolate_cross(lower_idx, upper_idx):
            y_low = data[lower_idx]
            y_high = data[upper_idx]
            if y_high == y_low:
                return float(upper_idx)
            frac = (half_max_value - y_low) / (y_high - y_low)
            return float(lower_idx) + np.clip(frac, 0.0, 1.0)

        left_cross = interpolate_cross(max(left_idx - 1, 0), left_idx) if left_idx > 0 else float(left_idx)
        right_cross = (
            interpolate_cross(right_idx, min(right_idx + 1, len(data) - 1))
            if right_idx < len(data) - 1
            else float(right_idx)
        )

        fwhm = max(right_cross - left_cross, 0.0)
        if fwhm == 0.0:
            return min_sigma_pixels

        sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2)))
        return float(max(sigma, min_sigma_pixels))

    peak_idx = int(np.argmax(x_summed_smooth))
    x_max = x_summed_smooth[peak_idx]

    num = len(x_summed_smooth)
    num /= 10
    num = int(np.maximum(np.minimum(num, 5), 1))
    edge_x = np.concatenate((xBasis_fit[:num], xBasis_fit[-num:]))
    edge_y = np.concatenate((x_summed_smooth[:num], x_summed_smooth[-num:]))

    if edge_x.size >= 2 and np.ptp(edge_x) > 0:
        try:
            slope, offset = np.polyfit(edge_x, edge_y, 1)
            slope = float(slope)
            xOffset = float(offset)
        except np.linalg.LinAlgError:
            slope = 0.0
            xOffset = float(np.nanmean(edge_y))
    else:
        slope = 0.0
        xOffset = float(np.nanmean(edge_y))

    baseline_at_peak = xOffset + slope * xBasis_fit[peak_idx]
    amp_guess = max(x_max - baseline_at_peak, 0.0)
    half_max_value = baseline_at_peak + amp_guess / 2.0

    width_sigma_pixels = max(
        widthFinder(half_max_value, peak_idx, x_summed_smooth),
        min_sigma_pixels,
    )
    if len(xBasis_fit) > 1:
        spacing = np.mean(np.diff(xBasis_fit))
    else:
        spacing = 1.0
    xWidth = width_sigma_pixels * spacing

    if window_sigma_multiplier is not None and np.isfinite(width_sigma_pixels):
        window_pixels = int(
            np.ceil(window_sigma_multiplier * max(width_sigma_pixels, min_sigma_pixels))
        )
        start_idx = max(peak_idx - window_pixels, 0)
        end_idx = min(peak_idx + window_pixels + 1, len(xBasis_fit))
        if start_idx > 0 or end_idx < len(xBasis_fit):
            xBasis_fit = xBasis_fit[start_idx:end_idx]
            x_summed_fit = x_summed_fit[start_idx:end_idx]
            x_summed_smooth = x_summed_smooth[start_idx:end_idx]
            peak_idx -= start_idx
            peak_idx = int(np.clip(peak_idx, 0, len(x_summed_smooth) - 1))

    x_center_guess = xBasis_fit[peak_idx]
    x_max = x_summed_smooth[peak_idx]

    b = ([0.0, xBasis_fit.min(), 1e-3, -np.inf, -np.inf], [25 * x_max, xBasis_fit.max(), np.inf, np.inf, np.inf])
    if fitBounds is not None:
        b[0][1] = fitBounds[0][0]
        b[0][2] = fitBounds[0][1]
        b[1][1] = fitBounds[1][0]
        b[1][2] = fitBounds[1][1]

    baseline_at_peak = xOffset + slope * x_center_guess
    amp_guess = max(x_max - baseline_at_peak, 0.0)

    if max_fit_points is not None and len(xBasis_fit) > max_fit_points:
        max_fit_points = int(max_fit_points)
        if max_fit_points < 2:
            raise ValueError("max_fit_points must be at least 2 when specified")
        sample_indices = np.linspace(
            0, len(xBasis_fit) - 1, max_fit_points, dtype=int
        )
        xBasis_fit = xBasis_fit[sample_indices]
        x_summed_fit = x_summed_fit[sample_indices]
        x_summed_smooth = x_summed_smooth[sample_indices]
        peak_idx = int(np.argmax(x_summed_smooth))
        x_center_guess = xBasis_fit[peak_idx]
        x_max = x_summed_smooth[peak_idx]
        baseline_at_peak = xOffset + slope * x_center_guess
        amp_guess = max(x_max - baseline_at_peak, 0.0)

    initialGuess = (amp_guess, x_center_guess, xWidth, xOffset, slope)

    def perform_gaussian_fit(
        fit_xBasis,
        fit_x_summed,
        starting_guess,
        *,
        fixed_baseline=None,
    ):
        num_trial = 0
        max_num_trial = 3
        isFitSuccessful = False
        if fixed_baseline is None:
            current_guess = starting_guess
        else:
            current_guess = (
                starting_guess[0],
                starting_guess[1],
                starting_guess[2],
            )
        current_width = max(starting_guess[2], np.finfo(float).tiny)
        x_center = 0.0
        x_width = 0.0
        x_offset = starting_guess[3]
        x_peakHeight = 0.0
        x_slope = starting_guess[4]
        x_fitted = []
        err = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        poptx = None
        poptx_full = None

        while num_trial <= max_num_trial and isFitSuccessful is False:
            try:
                if fixed_baseline is None:
                    jacobian = jac_single_G_plus_slope if use_analytic_jacobian else None
                    fit_func = single_G_plus_slope
                    bounds_to_use = b
                    p0 = current_guess
                else:
                    jacobian = None
                    fixed_offset, fixed_slope = fixed_baseline

                    def gaussian_with_fixed_baseline(x, A, center, sigma):
                        return single_G_plus_slope(
                            x, A, center, sigma, fixed_offset, fixed_slope
                        )

                    fit_func = gaussian_with_fixed_baseline
                    bounds_to_use = (
                        [b[0][0], b[0][1], b[0][2]],
                        [b[1][0], b[1][1], b[1][2]],
                    )
                    p0 = current_guess

                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    poptx, pcovx = curve_fit(
                        fit_func,
                        fit_xBasis,
                        fit_x_summed,
                        p0=p0,
                        bounds=bounds_to_use,
                        jac=jacobian,
                        check_finite=False,
                    )
                x_peakHeight = float(poptx[0])
                x_center = float(poptx[1])
                x_width = float(poptx[2])
                if fixed_baseline is None:
                    x_offset = float(poptx[3])
                    x_slope = float(poptx[4])
                    poptx_full = np.asarray(poptx, dtype=float)
                    err_vals = np.sqrt(np.diag(pcovx))
                    if err_vals.size < 5:
                        err_vals = np.pad(err_vals, (0, 5 - err_vals.size), mode="constant")
                else:
                    x_offset, x_slope = fixed_baseline
                    poptx_full = np.array(
                        [x_peakHeight, x_center, x_width, x_offset, x_slope],
                        dtype=float,
                    )
                    err_vals_partial = np.sqrt(np.diag(pcovx))
                    err_vals = np.zeros(5)
                    err_vals[: err_vals_partial.size] = err_vals_partial
                x_fitted = single_G_plus_slope(
                    xBasis, x_peakHeight, x_center, x_width, x_offset, x_slope
                )
                err = err_vals
                isFitSuccessful = True
                print("")
                print(
                    " ================ Gaussian FIT "
                    + str(axis)
                    + "-axis ==============="
                )
                print("")
                print("------ Guassian Parameters ------")
                print("")
                print(poptx)
                print("")
                print("------ Gaussian fitting SUCCESS ------")
            except Exception as exc:
                print("------ Gaussian fitting FAILED ------")
                if isinstance(exc, OptimizeWarning):
                    print(
                        "curve_fit raised OptimizeWarning; retrying with adjusted parameters"
                    )

                if num_trial == max_num_trial:
                    x_center = 0.0
                    x_width = 0.0
                    x_offset = starting_guess[3]
                    x_peakHeight = 0.0
                    x_fitted = []
                    x_slope = 0.0
                    err = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    isFitSuccessful = False
                else:
                    if num_trial % 2 == 0:
                        current_width = current_width / 2 ** (1 + num_trial)
                    else:
                        current_width = current_width * 2 ** (num_trial)
                    current_width = max(current_width, np.finfo(float).tiny)
                    if fixed_baseline is None:
                        current_guess = (
                            current_guess[0],
                            current_guess[1],
                            current_width,
                            current_guess[3],
                            current_guess[4],
                        )
                    else:
                        current_guess = (
                            current_guess[0],
                            current_guess[1],
                            current_width,
                        )

                print("")
                print(" ================ END FIT ===============")
                print("")

                num_trial += 1
                continue

            print("")
            print(" ================ END FIT ===============")
            print("")
            num_trial += 1

        return (
            x_center,
            x_width,
            x_offset,
            x_peakHeight,
            x_fitted,
            isFitSuccessful,
            x_slope,
            err,
            poptx_full if isFitSuccessful else poptx,
        )

    full_fit_result = perform_gaussian_fit(xBasis_fit, x_summed_fit, initialGuess)

    final_result = full_fit_result

    if fit_tails_only and full_fit_result[5]:
        poptx = full_fit_result[8]
        if poptx is not None:
            fitted_amplitude = float(poptx[0])
            fitted_center = float(poptx[1])
            fitted_sigma = float(np.abs(poptx[2]))
            fitted_offset = float(poptx[3])
            fitted_slope = float(poptx[4])

            baseline_values = fitted_offset + fitted_slope * xBasis_fit
            amplitude_threshold = baseline_values + tail_fraction * fitted_amplitude
            sigma_cutoff = np.abs(xBasis_fit - fitted_center) >= (
                tail_sigma_multiplier * max(fitted_sigma, np.finfo(float).tiny)
            )
            fitted_profile = single_G_plus_slope(
                xBasis_fit, fitted_amplitude, fitted_center, fitted_sigma, fitted_offset, fitted_slope
            )
            base_tail_mask = np.logical_or(
                fitted_profile <= amplitude_threshold,
                sigma_cutoff,
            )

            min_points_required = max(int(min_tail_points), 2)
            base_count = np.count_nonzero(base_tail_mask)

            if base_count >= min_points_required:
                tail_x = xBasis_fit[base_tail_mask]
                tail_y = x_summed_fit[base_tail_mask]

                normalized_side = (tail_side or "").strip().lower()
                if normalized_side in {"left", "right", "top", "bottom"}:
                    if normalized_side in {"left", "top"}:
                        side_condition = xBasis_fit <= fitted_center
                    else:
                        side_condition = xBasis_fit >= fitted_center

                    side_mask = base_tail_mask & side_condition
                    min_side_points = max(2, int(np.ceil(min_tail_points / 2.0)))

                    if np.count_nonzero(side_mask) >= min_side_points:
                        tail_x = xBasis_fit[side_mask]
                        tail_y = x_summed_fit[side_mask]

                        mirrored_x = (2.0 * fitted_center) - tail_x
                        mirrored_y = tail_y

                        tail_x = np.concatenate((tail_x, mirrored_x))
                        tail_y = np.concatenate((tail_y, mirrored_y))

                        order = np.argsort(tail_x)
                        tail_x = tail_x[order]
                        tail_y = tail_y[order]

                if tail_x.size >= min_points_required:
                    tail_initial_guess = (
                        fitted_amplitude,
                        fitted_center,
                        fitted_sigma,
                        fitted_offset,
                        fitted_slope,
                    )
                    tail_fit_result = perform_gaussian_fit(
                        tail_x,
                        tail_y,
                        tail_initial_guess,
                        fixed_baseline=(fitted_offset, fitted_slope),
                    )

                    if tail_fit_result[5]:
                        final_result = tail_fit_result

    (
        x_center,
        x_width,
        x_offset,
        x_peakHeight,
        x_fitted,
        isFitSuccessful,
        x_slope,
        err,
        _,
    ) = final_result

    return (
        x_center,
        x_width,
        x_offset,
        x_peakHeight,
        x_fitted,
        isFitSuccessful,
        x_slope,
        err,
    )
    
def radialAverage(data, center, boundary):
    r_max = int(
        min(
            abs(center[0] - boundary[0]),
            abs(center[0] - boundary[2]),
            abs(center[1] - boundary[1]),
            abs(center[1] - boundary[3]),
        )
    )
    shifted_center = np.array([int(center[0] - boundary[0]), int(center[1] - boundary[1])])
    y, x = np.indices(data.shape)

    radii = np.sqrt((x - shifted_center[0]) ** 2 + (y - shifted_center[1]) ** 2)
    rounded_radii = np.rint(radii).astype(int)
    mask = rounded_radii < r_max

    if not np.any(mask):
        return np.array([])

    finite_mask = np.isfinite(data)
    combined_mask = mask & finite_mask

    if not np.any(combined_mask):
        return np.array([])

    totals = np.bincount(rounded_radii[combined_mask], weights=data[combined_mask])
    counts = np.bincount(rounded_radii[combined_mask])

    with np.errstate(divide="ignore", invalid="ignore"):
        radial_profile = np.divide(
            totals,
            counts,
            out=np.zeros_like(totals),
            where=counts > 0,
        )

    return radial_profile[:r_max]
    

def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize = 0.2, weights=None, steps=False, interpnan=False, left=None, right=None,
        mask=None ):
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape,dtype='bool')

    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    nr = np.histogram(r, bins, weights=mask.astype('int'))[0]

    if stddev:
        whichbin = np.digitize(r.flat,bins)
        radial_prof = np.array([image.flat[mask.flat*(whichbin==b)].std() for b in range(1, nbins+1)])
    else: 
        radial_prof = np.histogram(r, bins, weights=(image*weights*mask))[0] / np.histogram(r, bins, weights=(mask*weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        
        return xarr, yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return bins[1:], radial_prof

def initialGauss(data):
	size = np.shape(data)

	xSlice = np.sum(data,0)    
	ySlice = np.sum(data,1)
	x0 = np.argmax(xSlice)
	y0 = np.argmax(ySlice)
	offset = np.nanmin(data)
	peak = np.nanmax(data)
	amplitude = peak - offset

	a = 0
	xOff = np.nanmin(xSlice)
	maxX = np.nanmax(xSlice)-xOff
	for i in range(len(xSlice)):
		if xSlice[i] - xOff > 0.5 * maxX:
			a += 1
	b = 0
	yOff = np.nanmin(ySlice)
	maxY = np.nanmax(ySlice)-yOff
	for i in range(len(ySlice)):
		if ySlice[i] - yOff > 0.5 * maxY:
			b += 1  

	return [x0, y0, max(a, 0.1), max(b,0.1), amplitude, offset]


def qguess(tof, sigma, density, mass = 6.):
    """Estimate dimensionless chemical potential q from time-of-flight data.

    Parameters
    ----------
    tof : float
        Time of flight in seconds.
    sigma : float
        RMS width after time of flight. The ratio ``sigma/tof`` corresponds
        to the one-dimensional root-mean-square (RMS) velocity of the cloud.
    density : float
        Peak number density in m^-3.
    mass : float, optional
        Atomic mass in atomic mass units (amu). Defaults to ``6`` amu.

    Returns
    -------
    float
        Dimensionless chemical potential ``q = \beta \mu``.
    """
    mass = mass * 10**-3/(6.02*10**23)
    rms_velocity = sigma / tof

    T = mass * (rms_velocity ** 2) / kB
    beta = 1./(kB*T)

    mu = hbar**2/(2*mass) * (3 * np.pi**2)**(2./3.) * density** (2./3.)
    q = beta * mu
    return q

    
def fitData(data, distribution, option):

    tmp0 =time.time()
    size = np.shape(data)
    
    
    	
    coordinates = np.meshgrid(range(size[1]), range(size[0]))





def f(x):
    return (1+x)/x * np.log(1+x)


def radioDistribution(data, center, sigma):

	size = np.shape(data)
	
	x1 = min(center[0], size[0]-center[0])/float(sigma[0])
	y1 = min(center[1], size[1]-center[1])/float(sigma[1])
	r0 = min(x1, y1)

	lr = int(0.95*r0)
	od_list = []

	for r in np.arange(0, lr, 0.01):
		od = 0
		for theta in range(0, 360, 5):
			x = center[0] + int(r*np.cos(theta) * sigma[0])
			y = center[1] + int(r*np.sin(theta) * sigma[1])
			od += data[y, x]
		od=od/360
		od_list.append(od)

	return od_list



