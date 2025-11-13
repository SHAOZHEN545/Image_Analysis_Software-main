'''Utilities to remove interference fringes.'''

import os, sys, glob, gc
from os import listdir
from os.path import isfile, join
from matplotlib import gridspec
from matplotlib import rc

import numpy as np
import matplotlib
import datetime
import time

from sklearn.decomposition import PCA as sklearnPCA
from astropy.io import fits
from matplotlib.mlab import PCA

from scipy import linalg as LA
from scipy import ndimage

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

from PIL import Image

from shutil import copyfile

from imagePlot import *
from imgFunc_v7 import *
from watchforchange import *
from localPath import *
from exp_params import *
from fitTool import *
from Monitor import*

class defringer():
    def __init__(self):
        self.backgroundPath = "D:\\Dropbox (MIT)\\BEC3-CODE\\imageAnalyze\\working branch\\side images_EMCCD\\pca_test\\background\\"
        self.imagePath = "D:\\Dropbox (MIT)\\BEC3-CODE\\imageAnalyze\\working branch\\side images_EMCCD\\pca_test\\atoms\\"
        self.yTop = 0
        self.yBottom = 1        
        self.xLeft = 0
        self.xRight = 1
        self.roiIndex = []
        self.betterRef = []                 
                                                     
    
    
    def images(self, fileNameList):
        temp = []
        for fileName in fileNameList:
            temp.append(-np.log(self.readFits(fileName)))
        return temp
        
    def imagesFlattened(self, fileNameList):
        temp = []
        for fileName in fileNameList:
            temp.append(-np.log(self.readFits(fileName).flatten()))
        return np.array(temp)
    
    def avgImages(self, fileNameList):
        temp = np.asarray(self.images(fileNameList))    
        return np.mean(temp, axis = 0)
        
    def setNoAtomFilePath(self, path):
        self.backgroundPath = path
        
    def pca(self, noAtomList, numOfDim = -1, center = False):
        temp = readNoAtomImage(noAtomList[0])
        self.shape = temp.shape

        if numOfDim == -1:
            print("num of dim:" + str(len(noAtomList)))
            numOfDim = len(noAtomList)
        
        if center is True:        
            data = self.center(noAtomList)
        else:
            data = readNoAtomImageFlattened(noAtomList) 
            
        U, s, Vt = LA.svd(data, full_matrices = False)
        V = Vt.T
        
        return Vt[:numOfDim, :]

    
    def defringedRef(self, absImgFileName, idx, num, setRoiIndex = False):
        pathForBackGround = self.backgroundPath
        os.chdir(pathForBackGround)   
        noAtomFileNameList = sorted(glob.iglob(pathForBackGround + '*.' + 'fits'), key=os.path.getctime, reverse = True)[idx : num]

        
        
        noAtomList= self.pca(noAtomFileNameList)
        atomImage = readAtomImage(absImgFileName)
        if setRoiIndex is False:
            self.roiIndex = np.ones(atomImage.shape).flatten()
        else:
            temp = np.ones(atomImage.shape)      
            temp[self.yTop-3:self.yBottom+4, self.xLeft-3:self.xRight+4] = 0
            self.roiIndex = temp.flatten()
            
        betterNoAtom = self.createBetterRef(self.roiIndex, noAtomList, atomImage).reshape(atomImage.shape)
        
        self.betterRef = betterNoAtom
        
        return betterNoAtom

    def defringedImage(self, absImgFileName, num, setRoiIndex = False):
        atomImage = readAtomImage(absImgFileName)
        betterRef = self.defringedRef(absImgFileName, num, setRoiIndex)        
        absorbImg = -np.log(atomImage/betterRef)
        
        minT = np.exp(-5)
        temp = np.empty(atomImage.shape)	
        temp.fill(minT)
        absorbImg = np.maximum(absorbImg, temp)
        return absorbImg
        
    def BMatrix(self, roiIndex, noAtomImageList):
        numOfRef = noAtomImageList.shape[0]
        BMatrix = np.zeros((numOfRef, numOfRef))
        for i in range(numOfRef):
            for j in range(numOfRef):
                BMatrix[i,j] = np.sum(np.dot(np.dot(noAtomImageList[i], noAtomImageList[j]), roiIndex))
                
        return BMatrix
    
    def DVector(self, roiIndex, noAtomImageList, atomImage):
        numOfRef = noAtomImageList.shape[0]
        DVector = np.zeros(numOfRef)
        for i in range(numOfRef):
            DVector[i] = np.sum(np.dot(np.dot(atomImage, noAtomImageList[i]), roiIndex))
            
        return DVector
    
    def createBetterRef(self, roiIndex, noAtomList, absImage, flag = -1):
        if flag == 1:
            absImage = absImage.flatten()
            B = self.BMatrix(roiIndex, noAtomList)
            DVec = self.DVector(roiIndex, noAtomList, absImage)
            CVec = LA.solve(B, DVec)
            
            ref = np.zeros(absImage.shape)
            for i in range(len(CVec)):
                ref += CVec[i] * noAtomList[i]                    
        else:
            temp = np.copy(absImage)
            temp[self.yTop-3:self.yBottom+4, self.xLeft-3:self.xRight+4] = 0
            coeff = np.dot(temp.flatten(), noAtomList.T)
            ref = np.dot(noAtomList.T, coeff)
        
        return ref
    
    def setRoiIndex(self, roiIndex):
        self.yTop = roiIndex[1]
        self.yBottom = roiIndex[3]
        self.xLeft = roiIndex[0]
        self.xRight = roiIndex[2]

    def meanSquaredDeviation(self, NoAtomImage, atomImage):
        print(self.roiIndex.shape)
        return np.sum(np.dot((NoAtomImage - atomImage)**2, self.roiIndex))
         
if __name__ == '__main__':
    

    tester = defringer()

    pathForBackGround = tester.backgroundPath
    os.chdir(pathForBackGround)   
    noAtomList =  glob.glob(pathForBackGround + '*.' + 'fits')
    
    
    pathForAtomImage = tester.imagePath
    os.chdir(pathForAtomImage)
    atomImage = glob.glob(pathForAtomImage + '*.' + 'fits')[0]    
    original = -np.log(readData(atomImage, "fits")[1])

    tester.setRoiIndex([88, 70, 115, 93])
    corrected = tester.defringedImage(atomImage, setRoiIndex = True)

    msdOriginal = tester.meanSquaredDeviation(readNoAtomImage(atomImage).flatten(), readAtomImage(atomImage).flatten())
    msdCorrected = tester.meanSquaredDeviation(tester.betterRef.flatten(), readAtomImage(atomImage).flatten())
    
    print(msdOriginal)
    print(msdCorrected)
    print((msdOriginal - msdCorrected)*100./msdOriginal)
    
    f, (axes1, axes2) = plt.subplots(1, 2, sharex=True)
    axes1.imshow(original, cmap='gray_r', aspect='auto', vmin=-1, vmax=1)
    axes2.imshow(corrected, cmap='gray_r', aspect='auto', vmin=-1, vmax=1)
    plt.show()
    
    

