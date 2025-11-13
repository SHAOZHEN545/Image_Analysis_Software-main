'''Panel that displays results figures.'''

import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import wx
from fitTool import *

class FigurePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent,  size=(600, 600))
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.canvas.SetToolTip("Image and fit display")

    def drawFigure(self, data, plotMin, plotMax):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.axes.plot(t, s)
        self.axes.imshow(data, vmin = plotMin, vmax = plotMax)
        self.draw()


