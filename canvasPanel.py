'''Panel that embeds a Matplotlib figure.'''

import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import wx
from fitTool import *

class CanvasPanel(wx.Panel):
    """Simple panel holding a single Matplotlib figure."""

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.canvas.SetToolTip("Plot display")
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def draw(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes.plot(t, s)


if __name__ == "__main__":
    app = wx.PySimpleApp()
    fr = wx.Frame(None, title='test')
    panel = CanvasPanel(fr)
    panel.draw()
    fr.Show()
    app.MainLoop()