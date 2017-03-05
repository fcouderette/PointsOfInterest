# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:03:53 2017

@author: frederique
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io as io
import random
import math as m
from cmath import *

import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
from scipy import ndimage
import scipy.signal as ss


def readImage(path):
    """ Reads and returns image """
    plt.figure()
    image=io.imread(path)
    
    # Display result
    io.imshow(image[:,:])
    plt.title('Original Image')
    print('Size of Image : ',image.shape)
    
    return image


def applySobelFilterRGB(image):
    """ Defines and applies Sobel filter to image. Returns Ix and Iy"""
    
    # Sobel horizontal filter
    sobelFilterH=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    #print("\nHorizontal Sobel filter : \n",sobelFilterH)
    # Sobel vertical filter
    sobelFilterV=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    #print("\nVertical Sobel filter : \n",sobelFilterV)
    
    # Horizontal convolution
    IxR = ndimage.filters.convolve(image[:,:,0],sobelFilterH)  # horizontal
    IxG = ndimage.filters.convolve(image[:,:,1],sobelFilterH)  # horizontal
    IxB = ndimage.filters.convolve(image[:,:,2],sobelFilterH)  # horizontal
    
    # Vertical Convolution
    IyR = ndimage.filters.convolve(image[:,:,0],sobelFilterV)  # vertical 
    IyG = ndimage.filters.convolve(image[:,:,1],sobelFilterV)  # vertical 
    IyB = ndimage.filters.convolve(image[:,:,2],sobelFilterV)  # vertical 
    
    return IxR,IyR,IxG,IyG,IxB,IyB


def computeGi(Ix,Iy):
    """Computes Gi"""
    Gi=(Ix*Ix+Iy*Iy)**(1/2)
    
    return Gi
    
    
def computeThetai(Ix,Iy):
    """Computes Thetai"""
    
    #thetai=np.arctan(Ix/Iy)
    thetai=2*np.arctan(Ix/(Iy+np.sqrt(Ix*Ix+Iy*Iy)))
    
    #thetai=phase(complex(Ix,Iy))
    
    return thetai
    
def displayThing(image, title):
    """ """
    plt.figure()
    io.imshow(image)
    plt.title(title)
    
    return 0

# =============================================================================


if __name__=='__main__':
    img=readImage('donnees_pour_tp/aerien_rvb/im1.rvb.tif')
    
    # 1) Compute Ix and Iy for RGB
    IxR,IyR,IxG,IyG,IxB,IyB=applySobelFilterRGB(img)
    displayThing(IxR, 'IxR')
    displayThing(IyR, 'IyR')
    displayThing(IxG, 'IxG')
    displayThing(IyG, 'IyG')
    displayThing(IxB, 'IxB')
    displayThing(IyB, 'IyB')
    
    # 2) Compute Gi and thetai for RGB
    GiR=computeGi(IxR,IyR)
    GiG=computeGi(IxG,IyG)
    GiB=computeGi(IxB,IyB)
    
    displayThing(GiR, 'GiR')
    displayThing(GiG, 'GiG')
    displayThing(GiB, 'GiB')
    
    thetaiR=computeThetai(IxR,IyR)
    thetaiG=computeThetai(IxG,IyG)
    thetaiB=computeThetai(IxB,IyB)
    
    displayThing(thetaiR, 'thetaiR')
    displayThing(thetaiG, 'thetaiG')
    displayThing(thetaiB, 'thetaiB')
    
    # 3) Compute Gamma for RGB
    
    
    
    
    
    
    
    