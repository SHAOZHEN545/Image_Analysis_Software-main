'''Tools for fitting degenerate gas data.'''

import numpy as np
import copy
import time
import sys, struct
import os
import scipy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import Image
from constant_v6 import *
from polylog import *
from imgFunc_v7 import *
from fitTool import *
from astropy.io import fits
from skimage import io

class degenerateFitter():
    def __init__(self):
        self.x_center = 0.
        self.x_width = 1.
        self.x_peakHeight = 1.
        self.x_offset = 0.
        self.x_slope = 0.

        self.x_basis = np.array([])
        self.x_summed = np.array([])

        self.x_center_degen = 0.
        self.thermal_width = 1.
        self.thermal_amp = 1.
        self.TF_radius = 1.
        self.BEC_amp = 1.
        self.offset_degen = 0.
        self.slope_degen = 0.

        self.totalPopulation = 1.

        self.fermi_center = 1.
        self.fermi_radius = 1.
        self.fermi_height = 1.
        self.fermi_offset = 0.
        self.fermi_slope = 1.
        self.q = -1.
        self.tof = 1.
        self.ToverTF = 1.
        
    def setTOF(self, tof):
        self.tof = tof * 10**-3
    
    def setInitialCenter(self, x0):
        self.x_center = x0
    
    def setInitialWidth(self, w0):
        self.x_width = w0
    
    def setInitialPeakHeight(self, h0):
        self.x_peakHeight = h0
    
    def setInitialOffset(self, offset):
        self.x_offset = offset
    
    def setInitialSlope(self, slope):
        self.x_slope = slope

    def setData(self, basis, data):
        self.x_basis = basis
        self.x_summed = data    
    
    def getFolded1DProfile(self):
        temp_center_x = int(round(self.x_center))
        below_center = self.x_summed[:temp_center_x]
        above_center = self.x_summed[temp_center_x:]
        profile_length = np.minimum(len(below_center), len(above_center))
        
        x_folded_profile = np.empty(profile_length)
        x_folded_profile[0] = self.x_summed[temp_center_x]
        for i in np.arange(1, profile_length-1):
            x_folded_profile[i] = (above_center[i] + below_center[-i])/2.
        
        self.x_folded_profile = x_folded_profile
        self.x_folded_basis = np.linspace(temp_center_x, above_center[profile_length-1], profile_length)
    
    def doDegenerateFit(self, isBoson = True):
        try:
            if len(self.x_basis) == 0 or len(self.x_summed) == 0:
                raise Exception(" <<<<<< Can't do Degenerate fit since NO DATA >>>>>>>> ")
                        
            def letsFitBoson(scaling_factor):
                g = [self.x_center,  scaling_factor*self.x_width, self.x_peakHeight/100./scaling_factor, self.x_width/scaling_factor, self.x_peakHeight * scaling_factor, self.x_offset, self.x_slope]
                b  = ([0.8 * g[0], 0, 0, 0, 0, -np.inf, -np.inf], [1.2 *g[0], np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
                time1 = time.time()
                p, q = curve_fit(condensate1DDist, self.x_basis, self.x_summed, p0=g, bounds = b, method = 'trf')
                time2 = time.time()
                print("-------- BEC fit took " + str(time2 - time1) + " seconds...")
                print(p)
                self.x_center_degen = p[0]
                self.thermal_width = p[1]
                self.thermal_amp = p[2]
                self.TF_radius = p[3]
                self.BEC_amp = p[4]
                self.offset_degen = p[5]
                self.slope_degen = p[6]
   
            def letsFitFermion(scaling_factor):
                """Fit the fermionic distribution with a scaled density guess."""
                print(" in the letsFitFermion....")
                density = 10**13 * 10**6 * scaling_factor
                init_q = qguess(self.tof, self.x_width, density)
                g = [self.x_center,  self.x_width, self.x_peakHeight, self.x_offset, self.x_slope, init_q]
                b  = ([0.8 * g[0], 0.5 * g[1], 0.5 * g[2], -np.inf, -np.inf, -np.inf], [1.2 *g[0], np.inf, np.inf, np.inf, np.inf, np.inf])
                time1 = time.time()
                p, q = curve_fit(fermion1DDist, self.x_basis, self.x_summed, p0=g, bounds = b, method = 'trf', maxfev=1e10)
                time2 = time.time()
                print("-------- Fermion fit took " + str(time2 - time1) + " seconds...")
                print(p)
                self.fermi_x_center = p[0]
                self.fermi_radius = p[1]
                self.fermi_height = p[2]
                self.fermi_offset = p[3]
                self.fermi_slope = p[4]
                self.q = p[5]
                
            def fitDecider():
                if self.thermal_width > self.TF_radius and self.BEC_amp >= self.thermal_amp:
                    return True
                else:
                    return False
                
            def fermiFitDecider():
                if self.fermi_radius > self.x_width and self.q > 0:
                    return True
                else:
                    return False
           
            if isBoson is True:
                i = 0
                num_trial = 200
                sc = 1.
                while i < num_trial:
                    letsFitBoson(sc)
                    if fitDecider() is True:
                        print(" <<<<<< BEC FIT SUCCEEDED >>>>>> ")
                        break
                    print("======= BEC fit " + str(i+1) + " trial =======")
                    i += 1
                    sc += .25 * 2
                
                if i == num_trial:
                    print(" <<<<<< BOSON BEC FIT FAILED >>>>>> ")
                    return
                
                self.x_fitted_degen = condensate1DDist(self.x_basis, self.x_center_degen, self.thermal_width, self.thermal_amp, self.TF_radius, self.BEC_amp, self.offset_degen, self.slope_degen)
                self.calculateTemp()
            
            if isBoson is False:
                i = 0
                num_trial = 200
                sc = 1.
                reachedLimit = False
                while i < num_trial:
                    letsFitFermion(sc)
                    if fermiFitDecider() is True:
                        print(" <<<<<< Fermion FIT SUCCEEDED >>>>>> ")
                        break
                    
                    print("======= Fermion fit " + str(i+1) + " trial =======")
                    i += 1
                    sc = sc * 2.

                    if i == num_trial:
                        g = [self.x_center,  self.x_width, self.x_peakHeight, self.x_offset, self.x_slope]
                        b  = ([0.8 * g[0], 0.5 * g[1], 0.5 * g[2], -np.inf, -np.inf], [1.2 *g[0], np.inf, np.inf, np.inf, np.inf])
                        time1 = time.time()
                        p, q = curve_fit(pureDegenerateFermion1D, self.x_basis, self.x_summed, p0=g, bounds = b, method = 'trf', maxfev=1e10)
                        reachedLimit = True
                        
                        self.x_fitted_degen = pureDegenerateFermion1D(self.x_basis, p[0], p[1], p[2], p[3], p[4])    

                if reachedLimit is False:
                    self.x_fitted_degen = fermion1DDist(self.x_basis, self.fermi_center, self.fermi_radius, self.fermi_height, self.fermi_offset, self.fermi_slope, self.q)
                
                self.calculateToverTF()
                
        except Exception as e:
            print(" ~~~~~~~~ degenerate fitting failed ~~~~~~~~~")
            print(e)
            return e

    def calculateToverTF(self):
        self.ToverTF = (6 * fermi_poly3(self.q))**(-1/3)
        print("")
        print("")
        print(" ~~~~~~ T/T_F = " + str(self.ToverTF))
        print("")
        print("")
        
    def calculateTemp(self):
        becPopulation = self.BEC_amp  * 4./3. * self.TF_radius
        thermalPopulation = self.thermal_amp *np.sqrt(2 * np.pi) * self.thermal_width
        temp = condensate1DDist(self.x_basis, self.x_center_degen, self.thermal_width, self.thermal_amp, self.TF_radius, self.BEC_amp, 0., 0.)
        totalPopulation = np.trapz(temp, self.x_basis, dx=.1)
        
        self.totalPopulation = totalPopulation
        self.becPopulationRatio = (becPopulation/totalPopulation)
        self.tOverTc = (1 - self.becPopulationRatio)**(1./3.)
        
        print("")
        print("")
        print("BEC Population: " + str(becPopulation))
        
        print("")
        print("Thermal Population: " + str(thermalPopulation))        
        
        print("Sum = " + str(becPopulation + thermalPopulation))
        print("")
        print("Total Population: " + str(totalPopulation))
        
        print("")
        print("")
        print(" ~~~~~~ T/T_C = " + str(self.tOverTc))
        print("")
        print("")
        print("Thermal Width: " + str(self.thermal_width) + " pixels")
        print("")
        print("")
    
    def getFittedProfile(self):
        return self.x_fitted_degen
    
    def getTOverTc(self):
        return self.tOverTc
    
    def getTotalPopulation(self):
        return self.totalPopulation
    
    def getBecPopulationRatio(self):
        return self.becPopulationRatio
    
    def getThomasFermiRadius(self):
        return self.TF_radius
    
    def getThermalWidth(self):
        return self.thermal_width
    
    def getThermalAmp(self):
        return self.thermal_amp
    
    def getFermiCenter(self):
        return self.fermi_center
    
    def getFermiRadius(self):
        return self.fermi_radius
    
    def getFermiHeight(self):
        return self.fermi_height

    def getFermiOffset(self):
        return self.fermi_offset
    
    def getFermiSlope(self):
        return self.fermi_slope
        
    def getQ(self):
        return self.q
    
    def getTOverTF(self):
        return self.ToverTF

