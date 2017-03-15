# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:07:37 2017

@author: frederique
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import matplotlib.cm as cmx
#import matplotlib.colors as colors
import skimage.io as io
import random
import math as m
from cmath import *

import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
from scipy import ndimage
import scipy.signal as ss
import scipy.ndimage.filters as sf
import skimage.feature as sfe


import time

# =============================================================================

def readImage(path):
    """ Reads and returns image """
    plt.figure()
    image=io.imread(path)
    
    # Display result
    io.imshow(image[:,:])
    plt.title('Original Image')
    print('Size of Image : ',image.shape)
    
    return image

# =============================================================================    

def displayImg(image, title):
    """ """
    io.imshow(image)
    plt.title(title)
    print('Size of Image : ',image.shape)
    
    return 0

# =============================================================================
    
def applySobelFilter(image):
    """ Defines and applies Sobel filter to image. Returns Ix and Iy"""
    
    # Sobel horizontal filter
    sobelFilterH=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    #print("\nHorizontal Sobel filter : \n",sobelFilterH)
    
    # Sobel vertical filter
    sobelFilterV=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    #print("\nVertical Sobel filter : \n",sobelFilterV)
    
    # Horizontal convolution
    Ix = ndimage.filters.convolve(image,sobelFilterH)  # horizontal
#    plt.figure()
#    io.imshow(Ix)
#    plt.title('Ix')
    
    # Vertical Convolution
    Iy = ndimage.filters.convolve(image,sobelFilterV)  # vertical 
#    plt.figure()
#    io.imshow(Iy)
#    plt.title('Iy')
    
    return Ix,Iy

# =============================================================================

def computeGi(Ix,Iy):
    """Computes Gi"""
    Gi=(Ix*Ix+Iy*Iy)**(1/2)
    
    # Display result
#    plt.figure()
#    io.imshow(Gi)
#    plt.title('Gi')
    
    return Gi
    
# =============================================================================

def computeThetai(Ix,Iy):
    """Computes Thetai"""
    
    #thetai=np.arctan(Ix/Iy)
    thetai=2*np.arctan(Ix/(Iy+np.sqrt(Ix*Ix+Iy*Iy)))
    
    #thetai=phase(complex(Ix,Iy))
    
    # Display result
#    plt.figure()
#    io.imshow(thetai)
#    plt.title('thetai')
    
    return thetai

# =============================================================================

def Gamma(x0,y0,r):
    """Computes Gamma"""
    #myList=np.zeros((4*r,4))
    i=0
    myList=[]
    
    
    for iLine in range(x0-r,x0+r):
        # Get through half the image (pairs are symetric)
        for iCol in range(y0-r,y0+1):
            if dist(x0,y0,iLine,iCol)<r:
                # First quarter
                if iLine<x0 and iCol<y0:
                    myList.append([iLine,iCol,np.abs(x0+(x0-iLine)),np.abs(y0+(y0-iCol))])
                    i+=1
                    
                # Second quarter
                elif iLine>x0 and iCol<y0:
                    myList.append([iLine,iCol,np.abs(x0-(iLine-x0)),np.abs(y0+(y0-iCol))])
                    i+=1
                    
                #line x0
                elif iLine==x0 and iCol<y0:
                    myList.append([x0,iCol,x0,y0+dist(x0,y0,iLine,iCol)])
                    i+=1
                    
                # line y0
                elif iLine<x0 and iCol==y0:
                    myList.append([iLine,y0,x0+dist(x0,y0,iLine,iCol),y0])
                    i+=1
                    
    
    return myList 

# =============================================================================

def Gamma2(Gamma,Gi):
    """ Computes pairs of pixels of Gamma space above a gradient threshold"""
    myList=[]
    
# Gamma : list
# Gi : matrix    
    
    for iLine in range(len(Gamma)): #.shape[0]):
        if Gi[Gamma[iLine][0],Gamma[iLine][1]]>(np.max(Gi)*0.7) and Gi[Gamma[iLine][2],Gamma[iLine][3]]>(np.max(Gi)*0.7) :
            myList.append([Gamma[iLine][0],Gamma[iLine][1],Gamma[iLine][2],Gamma[iLine][3]])
    
    #print('Gamma2')      
    return myList

# =============================================================================

def dist(x0,y0,x,y):
    """Computes distance between pixel (0,0) and pixel (x,y)"""
    distance = m.sqrt((x0-x)**2+(y0-y)**2)   
    #print('dist')
    return distance
    
# ============================================================================= 
    
def computeGWF(listOfPairs, Gi, line):
    """Computes gradient weight function"""
    gwf=m.log(1+Gi[listOfPairs[line][0], listOfPairs[line][1]])*m.log(1+Gi[listOfPairs[line][2], listOfPairs[line][3]])
    #print('computeGWF')
    return gwf
    
# =============================================================================    
    
def computePWF(listOfPairs, theta, line):
    """Computes phase weight function"""
    
    # For 2 different pixels, computes gamma_i and gamma_j
    if(listOfPairs[line][0]!=listOfPairs[line][2] and listOfPairs[line][1]!=listOfPairs[line][3]):
          
    
        gamma_i=theta[listOfPairs[line][0],listOfPairs[line][1]]-computeAngleBetweenLines(listOfPairs[line][0],listOfPairs[line][1],listOfPairs[line][2],listOfPairs[line][3]) # pi=Ix/Iy
        gamma_j=theta[listOfPairs[line][2],listOfPairs[line][3]]-computeAngleBetweenLines(listOfPairs[line][0],listOfPairs[line][1],listOfPairs[line][2],listOfPairs[line][3])  
    
    else:  
        gamma_i=0
        gamma_j=0
    
    # Computes pwf
    pwf=(1-m.cos(gamma_i + gamma_j)) * (1-m.cos(gamma_i - gamma_j))
    
    
    if(isnan(pwf)):
        pwf=0
    #print('computePWF')
    
    return pwf
    
# =============================================================================
    
def computeAngleBetweenLines(x1,y1,x2,y2):
    """ Computes angle between line (x1,y1)->(x2,y2) and horizontal"""
    
    # If line is not horizontal
    if(x1!=x2):
        angle=(y2-y1)/(x2-x1)
    # If line is horizontal
    else:
        angle=0
        
    return angle
    
# =============================================================================    
    
def computeSymetryMap(symetricPoints, G, theta):
    """ Computes element of symetry map """
    symetryMapElement=0
    for iLine in range(int(len(symetricPoints))): #303
#        myGwf=computeGWF(symetricPoints, G,iLine)
#        myPwf=computePWF(symetricPoints, theta, iLine)
#        symetryMapElement+=myPwf*myGwf
         symetryMapElement+=computeGWF(symetricPoints, G,iLine)*computePWF(symetricPoints, theta, iLine)
        #print('S = ',S)
    #print('computeSymetryMap')
    return symetryMapElement

# =============================================================================

def convolveSymetryMap(symetryMap, sigma):
    """ Convolve image by gaussian filter """
        
    gauss=sf.gaussian_filter(symetryMap, sigma)
    
    return gauss 

# =============================================================================

def detectAboveThreshold(im, R, threshold):
    """ Detects and displays local maxima """
    
    # Detection of local maxima with a minimum distance between them of 20 pixels
    coordinates = sfe.peak_local_max(im, min_distance=threshold)
    
    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    
    ax = axes.ravel()

    # Displays original image
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    
    # Displays convolved symetry map
    ax[1].imshow(R, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Convolution')
    
    # Displays local maxima on original image
    ax[2].imshow(im, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'ro')
    ax[2].axis('off')
    ax[2].set_title('Local Maxima')
    
    fig.tight_layout()
    
    plt.show()
        
    return coordinates

# =============================================================================

def exportDetectedPoint(indexes, data, filepath):
    """ Export list of point data in text file like following :
        index col line
        .
        .
        .
    """
    
    # Opens file
    fichier = open(filepath, "w")
    fichier.write('index column line\n')
    
    # For each sheet of xslx file
    for index in range(indexes.shape[0]):
        fichier.write(str(index))  
        fichier.write(" ") 
        fichier.write(str(indexes[index,1]))
        fichier.write(" ") 
        fichier.write(str(indexes[index,0]))
        fichier.write("\n") 
        
    fichier.close()
    
    
    return 0

# =============================================================================
# =============================================================================
# ============================= TESTS ==========================================
# =============================================================================
# =============================================================================


if __name__=='__main__':
    
    img=readImage('donnees_pour_tp/terrestre_detail_gris/image027.detail.tif')
    displayImg(img, 'Original Image')
    
    
    
    ## 1) Compute Ix and Iy
    Ix,Iy=applySobelFilter(img)
    displayImg(Ix, 'Ix')
    displayImg(Iy, 'Iy')
    
    
    
    ## 2) Compute Gi and thetai
    Gi=computeGi(Ix,Iy)
    thetai=computeThetai(Ix,Iy)
    displayImg(Gi, 'Gi')
    displayImg(thetai, 'thetai')
    
    
    
    ## 4) Compute symetry map
    # Gamma space radius
    r=3
    print('\nr=',r)
    
    # Initialization of symetry map
    S=np.zeros((img.shape[0],img.shape[1]))
       
    x0=r
    y0=r
    print('coord0 :',x0,y0)
    
    # Compute gamma space (pairs around a pixel of coordinates x0,y0)
    myGamma=Gamma(x0,y0,r)
    myGamma_stock=Gamma(x0,y0,r)
   
    # Time since beginning of script
    tps1=time.clock()
       
    iLine=r
    iCol=r
    while iLine <(img.shape[0]-r):
        #print('**line=',iLine)
        while iCol<(img.shape[1]-r):
            #print('col=',iCol)
        
            newList=Gamma2(myGamma_stock,Gi)
            #print('newList : ',newList)
            
            S[iLine,iCol]=computeSymetryMap(newList, Gi, thetai)
            #print(' S = ',S[iLine,iCol])
            
            # Next column
            iCol+=1
            for s in range(len(myGamma)):
                myGamma_stock[s][1]+=1
                myGamma_stock[s][3]+=1
        
            
        # Next line
        for n in range(len(myGamma)):
            myGamma_stock[n][0]+=1
            myGamma_stock[n][2]+=1
        
        # Starting column
        for k in range(len(myGamma)): #.shape[0]):
            myGamma_stock[k][1]=myGamma[k][1]        
            myGamma_stock[k][3]=myGamma[k][3]

        iLine+=1
        iCol=r
        
    tps2=time.clock()
    print("\nExecution time of S = ",tps2-tps1, " seconds")
    
    
    
    
    ## 5) Convolve Symetry Map
    R=convolveSymetryMap(S, 20)
    print('\n R = ',R)
    
    tps3=time.clock()
    print("\nExecution time of R = ",tps3-tps2, " seconds")
    
    
  
    
    ## 6) Detect maxima above threshold
    coordinates=detectAboveThreshold(img,R, 30)
    
    tps4=time.clock()
    print("\nExecution time of R = ",tps4-tps3, " seconds")
    
    print("\nGlobal execution time = ",tps4-tps1, " seconds")



    ## 7) Export local maxima in texte file
    exportDetectedPoint(coordinates, R, 'coordinates.txt')
   
    
    
    