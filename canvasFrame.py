'''Frame containing Matplotlib canvas.'''

import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import wx
from fitTool import *

class CanvasFrame(wx.Frame):
    """Frame that holds a Matplotlib canvas for displaying profiles."""

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, size=(1500, 500))
        
        figure, axes = plt.subplots(nrows= 4, ncols= 1)
        figure.tight_layout(h_pad=1.0) 
        self.axes1 = axes[0]
        self.axes1.set_title("x direction:")
        self.axes2 = axes[1]
        self.axes2.set_title("x direction (smoothed):")
        self.axes3 = axes[2]
        self.axes3.set_title("y direction:")
        self.axes4 = axes[3]
        self.axes4.set_title("y direction (smoothed):")        

        
        self.canvas = FigureCanvas(self, -1, figure)
        self.canvas.SetToolTip("Plot of fitted profiles")
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 8, wx.LEFT | wx.TOP | wx.GROW)
        
        self.calcResultsBox = wx.StaticText(self, label = "Calculation Results:")
        self.calcResultsBox.SetToolTip("Displays fit parameters")
        self.sizer.Add(self.calcResultsBox, 1, wx.LEFT | wx.TOP | wx.GROW)
        
        self.SetSizer(self.sizer)
        self.Fit()
        
        self.basis = np.arange(0.0, 3.0, 0.01) 
        
        self.x1 = np.sin(2 * np.pi * self.basis)
        self.x2 = np.cos(2 * np.pi * self.basis)        
        self.y1 = np.sin(2 * np.pi * self.basis)
        self.y2 = np.cos(2 * np.pi * self.basis)
        
    def setData(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2        
        self.y1 = y1
        self.y2 = y2            
    
    def setIndexRange(self, aoi):
        self.xBegin = aoi[0][0]
        self.xEnd= aoi[1][0]
        self.yBegin = aoi[0][1]
        self.yEnd= aoi[1][1]

    def draw(self):
        x1size = self.x1.shape[0]
        x2size = self.x2.shape[0]
        x1basis = np.linspace(self.xBegin, self.xEnd, x1size)
        x2basis = np.linspace(self.xBegin, self.xEnd, x2size)
        self.axes1.plot(x1basis, self.x1)
        self.axes2.plot(x2basis, self.x2)        
    
        y1size = self.y1.shape[0]
        y2size = self.y2.shape[0]
        y1basis = np.linspace(self.yBegin, self.yEnd, y1size)
        y2basis = np.linspace(self.yBegin, self.yEnd, y2size)
        self.axes3.plot(y1basis, self.y1)
        self.axes4.plot(y2basis, self.y2)

        self.calcResultsBox.SetLabel("blah blah")

        self.Show()
