'''Image processing helper functions.'''

from constant_v6 import *
import numpy as np
from scipy import stats
import os, sys, struct
from math import sqrt, log, atan
from scipy.optimize import curve_fit
from PIL import Image
from polylog import *
import copy
import time
import matplotlib.image as mpimg
from astropy.io import fits
from skimage import io
from scipy.ndimage import gaussian_filter


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def square(values):
    """Return a list of squares of the input values."""
    return [x ** 2 for x in values]

def readAIA(path):
    b = bytearray(2*3*1024*1024+100)
    f = open(path, "rb")
    try: numBytesRead = f.readinto(b) 
    finally: f.close()
    tmp0 = time.time()
    numBytesPerValue = struct.unpack('h',chr(b[3])+chr(b[4]))[0]
    rowTotal = struct.unpack('h',chr(b[5])+chr(b[6]))[0]
    colTotal = struct.unpack('h',chr(b[7])+chr(b[8]))[0]
    layerTotal = struct.unpack('h',chr(b[9])+chr(b[10]))[0]
    imageData = np.zeros((layerTotal,rowTotal,colTotal))
    byteIndex = 11
    for layer in range(layerTotal):
        for row in range(rowTotal):
            for col in range(colTotal):
                imageData[layer][row][col]= struct.unpack('h',chr(b[byteIndex])+chr(b[byteIndex+1]))[0]  
                byteIndex+=2    
    tmp1 = time.time()
    return imageData
    
def readTIF(path):
    start = time.time()
    imageData = []
    print("Opening tif image: " + path)
    im = Image.open(path)
    for i in range(3):
        im.seek(i)
        temp = np.sum(np.array(im).astype(float), axis=2)
        imageData.append(temp)
    end = time.time()
    print(str(end - start) + " seconds taken for TIF reading....")
    return imageData
    
def readFITS(path):
    start = time.time()
    imageData=[]
    try:
        fitsHDUlist = fits.open(path)
    except Exception as e:
        print(str(e))
    fits_data = fitsHDUlist[0].data
    for i in [0,1,2]:
        imageData.append((fits_data[i]).astype(float))
    end = time.time()
    return imageData

def createAbsorbImg(imageData, correctedNoAtom = None):
    if len(imageData) !=3:
        raise Exception("~~~~~~ Given image does not have three layers ~~~~~~~")
    
    if correctedNoAtom is None:
        correctedNoAtom = imageData[1] - imageData[2]

    correctedNoAtom[correctedNoAtom <= 0] = 1.
    absorbImg=(imageData[0]-imageData[2])/correctedNoAtom
    
    temp = np.empty(imageData[0].shape)	
    temp.fill(minT)
        
    absorbImg = np.maximum(absorbImg,temp)

    return absorbImg

def createNormalizedAbsorbImg(imageData, aoi):
    y, x = imageData[0].shape
    xLeft = aoi[0][0]
    yTop = aoi[0][1]
    xRight = aoi[1][0]
    yBottom = aoi[1][1]

    tempAtom = np.copy(imageData[0])
    meanAtom = (np.mean(tempAtom[0:y, 0:xLeft + 1]) + np.mean(tempAtom[0:y, xRight:x]) + np.mean(tempAtom[0:yTop+1, xLeft:xRight+1]) + np.mean(tempAtom[yBottom:y, xLeft:xRight+1]))/4
    stdAtom = np.sqrt((np.std(tempAtom[0:y, 0:xLeft + 1])**2 + np.std(tempAtom[0:y, xRight:x])**2 + np.std(tempAtom[0:yTop+1, xLeft:xRight+1])**2 + np.std(tempAtom[yBottom:y, xLeft:xRight+1])**2)/4)
    
    tempNoAtom = np.copy(imageData[1])
    meanNoAtom = (np.mean(tempNoAtom[0:y, 0:xLeft + 1]) + np.mean(tempNoAtom[0:y, xRight:x]) + np.mean(tempNoAtom[0:yTop+1, xLeft:xRight+1]) + np.mean(tempNoAtom[yBottom:y, xLeft:xRight+1]))/4
    stdNoAtom = np.sqrt((np.std(tempNoAtom[0:y, 0:xLeft + 1])**2 + np.std(tempNoAtom[0:y, xRight:x])**2 + np.std(tempNoAtom[0:yTop+1, xLeft:xRight+1])**2 + np.std(tempNoAtom[yBottom:y, xLeft:xRight+1])**2)/4)

    imageData[1] = meanAtom + (stdAtom/stdNoAtom)*(imageData[1] - meanNoAtom)
    correctedNoAtom = imageData[1] - imageData[2]

    return createAbsorbImg(imageData, correctedNoAtom)


def readData(path, filetype, betterRefOpt = [False, None]):
    num_trial = 0
    max_num_trial = 3
    while (num_trial < max_num_trial):
        try:
            if filetype == "aia":
                imageData = readAIA(path)
            elif filetype == "tif":
                imageData = readTIF(path)
            elif filetype == "fits":
                imageData = readFITS(path)
            else:
                raise Exception('---------Unknown file type-------')
            if betterRefOpt[0] is True:
                correctedNoAtom = betterRefOpt[1]
                if (correctedNoAtom is None) or (correctedNoAtom.shape != imageData[1].shape):
                    correctedNoAtom = imageData[1] - imageData[2]
            else:
                correctedNoAtom = imageData[1] - imageData[2]
            
            absorbImg = createAbsorbImg(imageData, correctedNoAtom)
            
            tmp2 = time.time()
            num_trial = max_num_trial
            return absorbImg, imageData
        except Exception:
            num_trial += 1
            print("")
            print(" ===== READING IMAGE TRIAL ----> " + str(num_trial) + " ====== ")
            print("")
            time.sleep(0.2)
            if (num_trial == max_num_trial):
                raise Exception("Imaging reading failed at imgFunc_v6.py after " + str(num_trial) + " trials...")

def readNoAtomImage(filename):
    imageData = readFITS(filename)
    temp= np.asarray(imageData[1] - imageData[2])
    return np.maximum(temp, .1)

def readNoAtomImageFlattened(fileNameList):
    temp = []
    for fileName in fileNameList:
        temp.append(readNoAtomImage(fileName).flatten())
    return np.array(temp)
    
def readAtomImage(filename):
    imageData = readFITS(filename)
    temp = np.asarray(imageData[0] - imageData[2])
    return np.maximum(temp, .1)

def atomNumber(Img, offset):

    img = np.asarray(Img, dtype=float)
    valid_mask = np.isfinite(img)

    if not np.any(valid_mask):
        return 0.0

    simplicio = np.nansum(img)
    pixel_count = np.count_nonzero(valid_mask)
    sim2 = simplicio - offset * pixel_count
    return sim2

    
def aoiEdge(Img, leftright, updown):
    img = np.asarray(Img, dtype=float)

    def region_mean(region):
        finite = np.isfinite(region)
        if not np.any(finite):
            return 0.0, 0
        return float(np.nansum(region[finite])), int(np.count_nonzero(finite))

    if leftright and updown:
        regions = [
            (slice(0, 3), slice(0, -3)),
            (slice(-3, None), slice(3, None)),
            (slice(3, None), slice(0, 3)),
            (slice(0, -3), slice(-3, None)),
        ]
    elif leftright and not updown:
        regions = [
            (slice(3, None), slice(0, 3)),
            (slice(0, -3), slice(-3, None)),
        ]
    elif updown and not leftright:
        regions = [
            (slice(0, 3), slice(0, -3)),
            (slice(-3, None), slice(3, None)),
        ]
    else:
        return 0.0

    total = 0.0
    count = 0
    for y_slice, x_slice in regions:
        region = img[y_slice, x_slice]
        region_sum, region_count = region_mean(region)
        total += region_sum
        count += region_count

    if count == 0:
        return 0.0

    return total / count
    
def twoDGaussianFit(data):
    """Fits a two-dimensional Gaussian, A*exp(-0.5*(((x-x0)/sigmaX)^2+((y-y0)/sigmaY)**2))+offset, to given data 
    (a numpy array) and returns the fit parameters [[x0,y0],[sigmaX,sigmaY],offset,A]. """

    def oneDGaussian(x, centerX,sDevX,Amp,yOffset):
        return Amp*np.exp(-0.5*((x-centerX)/sDevX)**2)+yOffset
    
    xSlice = np.sum(data,0)    
    ySlice = np.sum(data,1)
        
    xOff = np.nanmin(xSlice)
    AmpX = np.nanmax(xSlice)-xOff
    x0 = np.argmax(xSlice)
    
    yOff = np.nanmin(ySlice)
    AmpY = np.nanmax(ySlice)-yOff
    y0 = np.argmax(ySlice)
    sigmaX =40
    sigmaY =40

    xVals, yCovar = curve_fit(oneDGaussian,range(len(xSlice)),xSlice,p0=(x0,sigmaX,AmpX,xOff))
    x0 = xVals[0]
    sigmaX = xVals[1]

    
    yVals, yCovar = curve_fit(oneDGaussian,range(len(ySlice)),ySlice,p0=(y0,sigmaY,AmpY,yOff))
    y0 = yVals[0]
    sigmaY = yVals[1]
    
    offset = 0.5*(xVals[3]/(np.shape(data)[1]) + yVals[3]/(np.shape(data)[0]))
    A = 0.5 * ( xVals[2]/(sqrt(2.0*np.pi)*sigmaY) + yVals[2]/(sqrt(2.0*np.pi)*sigmaX) ) 
    
    return [[x0,y0],[sigmaX,sigmaY],A,offset]
    
def partlyCondensateFit(data):

    def oneDPartlyCondensate(x, centerX, AmpP, a, sDevX, AmpG, offset):
        return np.maximum(AmpG, 0)*np.exp(-0.5*((x-centerX)/sDevX)**2) + np.maximum(AmpP, 0) *  np.maximum( (1 - a * (x - centerX)**2), 0)  ** 2.5+ offset
       

    xSlice = np.sum(data,0)    
    ySlice = np.sum(data,1)
        
    xOff = np.nanmin(xSlice)
    AmpGX = np.nanmax(xSlice)-xOff
    x0 = np.argmax(xSlice)
    
    yOff = np.nanmin(ySlice)
    AmpGY = np.nanmax(ySlice)-yOff
    y0 = np.argmax(ySlice)
    sigmaX =40
    sigmaY =40

    lengthX = 0
    for i in range(len(xSlice)):
        if xSlice[i] - xOff > 0.5 * AmpGX:
            lengthX += 1
 
    lengthY = 0
    for i in range(len(ySlice)):
        if ySlice[i] - yOff > 0.5 * AmpGY:
            lengthY += 1  

    check = AmpGX*np.exp(-0.5)
    for index in range(x0,len(xSlice)):
        if xSlice[index] < check:
            sigmaX = index-x0
            break
    check = AmpGY*np.exp(-0.5)
    for index in range(y0,len(ySlice)):
        if ySlice[index] < check:
            sigmaY = index-y0
            break
    
    aX = 4./lengthX**2
    aY = 4./lengthY**2
    AmpPX = AmpGX
    AmpPY = AmpGY

    xVals, yCovar = curve_fit(oneDPartlyCondensate,range(len(xSlice)),xSlice,p0=(x0, AmpPX/2, aX, sigmaX, AmpGX/2, xOff))
    x0 = xVals[0]
    widthX = sqrt(1/xVals[2])
    sigmaX = xVals[3]
    
    yVals, yCovar = curve_fit(oneDPartlyCondensate,range(len(ySlice)),ySlice,p0=(y0, AmpPY/2, aY, sigmaY, AmpGY/2, yOff))
    y0 = yVals[0]
    widthY = sqrt(1/yVals[2])
    sigmaY = yVals[3]

    AmpP = sqrt(np.maximum(xVals[1], 0) * np.maximum(yVals[1], 0) * 9./16. * sqrt(xVals[2] * yVals[2]) )
    AmpG = 0.5 * ( np.maximum(xVals[4], 0)/(sqrt(2.0*np.pi)*sigmaY) + np.maximum(yVals[4], 0)/(sqrt(2.0*np.pi)*sigmaX) ) 
    offset = 0.5 * (xVals[5]/(np.shape(data)[1]) + yVals[5]/(np.shape(data)[0]))

    return [[x0,y0], [widthX, widthY], [sigmaX,sigmaY], AmpP, AmpG, offset]

def fermionFit(data):

    def oneDPolylog(x, centerX, Rx, Amp , q, yOffset):
        print([centerX, Rx, Amp, q, yOffset])
        x = np.array(x)

        numerator = fermi_poly5half(q - (x-centerX)**2/Rx**2 * np.exp(q))
        denuminator = fermi_poly5half(q)

        out = numerator/denuminator * Amp + yOffset

        return out

    xSlice = np.sum(data,0)    
    ySlice = np.sum(data,1)
        
    xOff = np.nanmin(xSlice)
    AmpX = np.nanmax(xSlice)-xOff
    x0 = np.argmax(xSlice)
    
    yOff = np.nanmin(ySlice)
    AmpY = np.nanmax(ySlice)-yOff
    y0 = np.argmax(ySlice)
    
    sigmaX = 0
    for i in xSlice:
        if i - xOff > 0.5 * AmpX:
            sigmaX += 1

    sigmaX/=2.
    sigmaY = 0
    for i in ySlice:
        if i - yOff > 0.5 * AmpY:
            sigmaY += 1
    sigmaY/=2.
    print([sigmaX, sigmaY])

    q0 = 1

    xVals, yCovar = curve_fit(oneDPolylog,range(len(xSlice)),xSlice,p0=(x0,sigmaX,AmpX, q0 ,xOff))
    yVals, yCovar = curve_fit(oneDPolylog,range(len(ySlice)),ySlice,p0=(y0,sigmaY,AmpY, q0 ,yOff))
    
    x0 = float(xVals[0])
    RX = float(xVals[1])

    y0 = float(yVals[0])
    RY = float(yVals[1])
    
    qx=float(xVals[3])
    qy=float(yVals[3])
    offset = float(0.5*(xVals[4]/(np.shape(data)[1]) + yVals[4]/(np.shape(data)[0])))
    A = np.array(data).max()
    
    return [[x0,y0],[RX,RY],A,[qx, qy],offset]
    

def temperatureSingleGaussianFit(ToF, gSigmaX, gSigmaY, OmegaAxial, OmegaRadial, atom):
    if atom == 'Li':
        m = mLi
    elif atom == 'Cs':
        m = mCs
    else:
        return False
    
    Tempx = (m * 2 * gSigmaX**2 * OmegaAxial ** 2)/(2*kB*(1 + OmegaAxial ** 2 * ToF ** 2))
    Tempy = (m * 2 * gSigmaY**2 * OmegaRadial ** 2)/(2*kB*(1 + OmegaRadial ** 2 * ToF ** 2))

    return [Tempx, Tempy]

def trapFrequencyRadial(ToF, rho, rho0):
    freq = (sqrt((rho/rho0)**2 - 1))/ToF
    return freq

def trapFrequencyAxial(ToF, z, rho0, omegaRadial):
    tau = ToF * omegaRadial
    temp = tau * atan(tau) - log(sqrt(1+ tau**2))
    a = temp
    b = - z/rho0
    e = (sqrt(b**2-4*a) - b)/(2*a)

    return e*omegaRadial
    
def chemicalPotential(ToF, omegaRadial, rho0):
    m = mLi
    mu = 0.5 * m * ((omegaRadial**2/(1+(omegaRadial**2) * (ToF**2))) * rho0**2)
    return mu

def effectiveInteraction(a):
    m = mLi
    return 4*np.pi*(hbar**2)*a/m
    
def atomNumberFit(mu, omegaRadial, omegaAxial, U0):
    m= mLi
    omegaBar = (omegaRadial**2 * omegaAxial) ** (1./3.)
    N = (8 *np.pi/15) * ((2*mu)/(m*omegaBar**2))**(3./2.) * (mu/U0)
    return N
        
def fillImageAOI(data, AOI, offset):
    out = np.zeros((1024,1024)) + offset
    out[AOI[0][0]:AOI[0][1],AOI[1][0]:AOI[1][1]] 

def dataFit(atom, arr1, arr2, arr3):
    if atom == "Li":
        m = mLi
    elif atom == "Cs":
        m = mCs

    timeOfFlight = np.array(arr1)
    # Widths are provided as the Gaussian \(\sigma\) radius; no extra scaling is
    # required. Previously these values were multiplied by \(\sqrt{2}\), which
    # doubled the temperature extracted from the slope. Removing that scaling
    # ensures the fit operates directly on the physical width.
    RX = np.array(arr2)
    RY = np.array(arr3)

    # Perform regression on \(\sigma^2\) versus \(t^2\)
    slopeX, bX, rX, pX, sX = stats.linregress(np.square(timeOfFlight), np.square(RX))

    # Temperature in Kelvin from the slope of \(\sigma^2\) vs \(t^2\)
    Tempx = slopeX * m / kB
    wx = np.nan
    if bX > 0:
        wx = np.sqrt(bX)

    slopeY, bY, rY, pY, sY = stats.linregress(np.square(timeOfFlight), np.square(RY))

    Tempy = slopeY * m / kB
    wy = np.nan
    if bY > 0:
        wy = np.sqrt(bY)

    # Return temperatures and the trap radii (initial Gaussian \(\sigma\)).
    return [Tempx, Tempy, wx, wy]

def TOverTF(q):
    return (6*fermi_poly3(q))**(-1./3.)


def calculate_phase_space_density(
    atom_number,
    temp_x_K,
    temp_y_K,
    mass,
    sigma_x,
    sigma_y,
    sigma_z=None,
    temp_z_K=None,
    atom_number_err=None,
    sigma_x_err=0.0,
    sigma_y_err=0.0,
    sigma_z_err=None,
    temp_x_err_K=0.0,
    temp_y_err_K=0.0,
    temp_z_err_K=None,
):
    """Compute the phase space density for a thermal cloud.

    Parameters
    ----------
    atom_number : float
        Total number of atoms ``N``.
    temp_x_K, temp_y_K, temp_z_K : float
        Cloud temperatures along each axis in Kelvin. ``temp_z_K`` defaults
        to ``temp_x_K`` if not provided.
    mass : float
        Atomic mass in kilograms.
    sigma_x, sigma_y, sigma_z : float
        Widths of the Gaussian cloud along each axis in meters.
    atom_number_err : float, optional
        Uncertainty on ``atom_number``. If ``None`` a Poisson
        uncertainty of ``sqrt(atom_number)`` is assumed.
    sigma_x_err, sigma_y_err, sigma_z_err : float, optional
        Uncertainties on the corresponding widths.
    temp_x_err_K, temp_y_err_K, temp_z_err_K : float, optional
        Uncertainties of the corresponding temperatures. ``temp_z_err_K``
        defaults to ``temp_x_err_K`` if not provided.

    Returns
    -------
    tuple
        ``(psd, psd_err)`` where ``psd`` is the phase space density and
        ``psd_err`` its propagated uncertainty.
    """

    if sigma_z is None:
        sigma_z = sigma_x
    if temp_z_K is None:
        temp_z_K = temp_x_K
    if atom_number_err is None:
        atom_number_err = np.sqrt(abs(atom_number))
    if sigma_z_err is None:
        sigma_z_err = sigma_x_err
    if temp_z_err_K is None:
        temp_z_err_K = temp_x_err_K

    volume = (2 * np.pi) ** 1.5 * sigma_x * sigma_y * sigma_z
    n0 = atom_number / volume

    h = 2 * np.pi * hbar
    lambda_x = h / np.sqrt(2 * np.pi * mass * kB * temp_x_K)
    lambda_y = h / np.sqrt(2 * np.pi * mass * kB * temp_y_K)
    lambda_z = h / np.sqrt(2 * np.pi * mass * kB * temp_z_K)
    psd = n0 * lambda_x * lambda_y * lambda_z

    rel_err_sq = 0.0
    if atom_number > 0:
        rel_err_sq += (atom_number_err / atom_number) ** 2
    if sigma_x > 0:
        rel_err_sq += (sigma_x_err / sigma_x) ** 2
    if sigma_y > 0:
        rel_err_sq += (sigma_y_err / sigma_y) ** 2
    if sigma_z > 0:
        rel_err_sq += (sigma_z_err / sigma_z) ** 2
    if temp_x_K > 0:
        rel_err_sq += ((0.5 * temp_x_err_K) / temp_x_K) ** 2
    if temp_y_K > 0:
        rel_err_sq += ((0.5 * temp_y_err_K) / temp_y_K) ** 2
    if temp_z_K > 0:
        rel_err_sq += ((0.5 * temp_z_err_K) / temp_z_K) ** 2

    psd_err = abs(psd) * np.sqrt(rel_err_sq)

    return psd, psd_err


def calculate_peak_density(
    atom_number,
    sigma_x,
    sigma_y,
    sigma_z,
    atom_number_err=None,
    sigma_x_err=0.0,
    sigma_y_err=0.0,
    sigma_z_err=0.0,
):
    """Return peak density of a 3D Gaussian cloud and its uncertainty.

    The peak density is evaluated from the Gaussian volume

    .. math::

        n_0 = \frac{N}{(2\pi)^{3/2} \sigma_x \sigma_y \sigma_z},

    where ``N`` is the atom number and ``\sigma_i`` are the 1/e radii
    along each axis in **meters**.  The returned density is expressed in
    atoms per cubic centimetre (cm\ :sup:`-3`).

    Parameters
    ----------
    atom_number : float
        Total atom number.
    sigma_x, sigma_y, sigma_z : float
        $1/\mathrm{e}$ radii of the Gaussian density distribution along
        each axis in metres.
    atom_number_err, sigma_x_err, sigma_y_err, sigma_z_err : float, optional
        Uncertainties associated with the respective parameters.

    Returns
    -------
    tuple
        ``(density, density_err)`` where ``density`` is the peak density
        ``n_0`` in atoms/cm³ and ``density_err`` its propagated
        uncertainty.
    """

    if atom_number_err is None:
        atom_number_err = np.sqrt(abs(atom_number))

    volume = (2 * np.pi) ** 1.5 * sigma_x * sigma_y * sigma_z
    n0_m3 = atom_number / volume

    rel_err_sq = 0.0
    if atom_number > 0:
        rel_err_sq += (atom_number_err / atom_number) ** 2
    if sigma_x > 0:
        rel_err_sq += (sigma_x_err / sigma_x) ** 2
    if sigma_y > 0:
        rel_err_sq += (sigma_y_err / sigma_y) ** 2
    if sigma_z > 0:
        rel_err_sq += (sigma_z_err / sigma_z) ** 2

    n0_err_m3 = abs(n0_m3) * np.sqrt(rel_err_sq)

    # Convert from m^-3 to cm^-3
    n0_cm3 = n0_m3 / 1e6
    n0_err_cm3 = n0_err_m3 / 1e6

    return n0_cm3, n0_err_cm3
