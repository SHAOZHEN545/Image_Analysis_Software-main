'''Helper for quick image plotting.'''

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
from fitTool import *

def atomImagePlot(data, caption, gParam, strParam):
    """Plot absorption images with annotations."""

    plt.close('all')
    ldata = len(data)
    fig = plt.figure(figsize=(15, 8))
    
    center = gParam[0:2]
    sigma = gParam[2:4]

    for r in range(ldata):
        positionString = '23' + str(r+1)
        plt.subplot(positionString)
        plt.imshow(data[r], cmap='jet', aspect='auto', vmin=-1, vmax=5)
        plt.grid(True, color='white')
        plt.title(caption[r])

    size = np.shape(data[0])
    
    y = []
    for k in range(ldata):
        l = radioDistribution(data[k], center, sigma)
        y.append(l)
    x= range(len(y[0]))
    positionString = '234' 
    plt.subplot(positionString)
    originalLine,  = plt.plot(x, y[0], 'g:', label='origianl')
    gaussianLine,  = plt.plot(x, y[1], 'r--', label='gaussian')
    fitLine,  = plt.plot(x, y[2], 'b-', label='fit')
    plt.legend([originalLine, gaussianLine, fitLine], caption)
    plt.title('Density Distribution')

    plt.subplot('235')
    plt.text(0.1,0.7,strParam[3],fontsize=20)
    plt.axis('off')

    plt.subplot('236')
    plt.text(0,0.7,strParam[0],fontsize=20)
    plt.text(0,0.4,strParam[1],fontsize=20)
    plt.text(0,0.1,strParam[2],fontsize=20)
    plt.axis('off')

    print("created fiimages")
    plt.show()



def atomNumberPlot(data1):
    n = len(data1)
    x = range(n)
    p1 = plt.plot(x, data1, 'ro')
    maxNumber = max(data1) 
    plt.axis("[-1, n, 0, maxNumber*1.2]")
    plt.grid(True)
    plt.legend(['Data'])
    plt.title('Atom Number')
    plt.show()

if __name__ == '__main__':
  
    data1 =  np.random.random((1024,1024))
    data2 =  np.random.random((1024,1024))
    data3 =  np.random.random((1024,1024))
    data = [data1, data2, data3]
    atomImagePlot(data, ['a','b','c'],  [500,600,154,184], ['asdfasdf','asdfasdf'])





