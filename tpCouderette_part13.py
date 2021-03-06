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


def readImage(path):
    """ Reads and returns image """
    plt.figure()
    image=io.imread(path)
    
    # Display result
#    io.imshow(image[:,:])
#    plt.title('Original Image')
#    print('Size of Image : ',image.shape)
    
    return image
    
    
    
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

def computeGi(Ix,Iy):
    """Computes Gi"""
    Gi=(Ix*Ix+Iy*Iy)**(1/2)
    
    # Display result
#    plt.figure()
#    io.imshow(Gi)
#    plt.title('Gi')
    
    return Gi
    
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

def Gamma(image,x0,y0,r,Gi):
    """Computes Gamma"""
    myList=[]
    #myLista=[]
    #Gamma=np.zeros((image.shape[0],image.shape[1]))
    
    for iLine in range(image.shape[0]):
        # Get through half the image (pairs are symetric)
        for iCol in range(int(image.shape[1]/2)):
            if m.sqrt((x0-iLine)**2+(y0-iCol)**2)<r:
                # First quarter
                if iLine<x0 and Gi[iLine,iCol]>(np.max(Gi)/10) and Gi[np.abs(x0+(x0-iLine)),np.abs(y0+(y0-iCol))]>(np.max(Gi)/10):
                    myList.append([iLine,iCol,np.abs(x0+(x0-iLine)),np.abs(y0+(y0-iCol))])
                # Second quarter
                elif iLine>=x0 and Gi[iLine,iCol]>(np.max(Gi)/10) and Gi[np.abs(x0-(iLine-x0)),np.abs(y0+(y0-iCol))]>(np.max(Gi)/10):
                    myList.append([iLine,iCol,np.abs(x0-(iLine-x0)),np.abs(y0+(y0-iCol))])
     
#    for iLine in range(len(myList)):
#         if dist(myList[iLine][0],myList[iLine][1],myList[iLine][2],myList[iLine][3])<2*r:
#             if Gi[myList[iLine][0],myList[iLine][1]]>(np.max(Gi)/10) and Gi[myList[iLine][2],myList[iLine][3]]>(np.max(Gi)/10) :
#                 myLista.append([myList[iLine][0],myList[iLine][1],myList[iLine][2],myList[iLine][3]])
                  
                
          
    # Display result
    #plt.figure()
    #io.imshow(Gamma)
    #plt.title('Gamma')
    #print('Gamma')
    
    return myList #,Gamma

#def dist(x0,y0,x,y):
#    """Computes distance between pixel (0,0) and pixel (x,y)"""
#    distance = m.sqrt((x0-x)**2+(y0-y)**2)   
#    return distance
    
    
    
def computeGWF(listOfPairs, Gi, line):
    """Computes gradient weight function"""
    gwf=m.log(1+Gi[listOfPairs[line][0], listOfPairs[line][1]])*m.log(1+Gi[listOfPairs[line][2], listOfPairs[line][3]])
    return gwf
    
    
def computePWF(listOfPairs, theta, line):
    """Computes phase weight function"""
    # listOfPairs[line][0],listOfPairs[line][1] are the coordinates of a pixel, not the value
    # image[listOfPairs[line][0],listOfPairs[line][1]] is the value at coordinates listOfPairs[line][0],listOfPairs[line][1]
    
    if(listOfPairs[line][0]!=listOfPairs[line][2] and listOfPairs[line][1]!=listOfPairs[line][3]):
        #alpha_ij= computeAngleBetweenLines(listOfPairs[line][0],listOfPairs[line][1],listOfPairs[line][2],listOfPairs[line][3])
        #print('alpha_ij',alpha_ij)    
    
        gamma_i=theta[listOfPairs[line][0],listOfPairs[line][1]]-computeAngleBetweenLines(listOfPairs[line][0],listOfPairs[line][1],listOfPairs[line][2],listOfPairs[line][3]) # pi=Ix/Iy
        gamma_j=theta[listOfPairs[line][2],listOfPairs[line][3]]-computeAngleBetweenLines(listOfPairs[line][0],listOfPairs[line][1],listOfPairs[line][2],listOfPairs[line][3])  
    
    else:  
        gamma_i=0
        gamma_j=0
    
#    print('\ntheta[listOfPairs[line][0],listOfPairs[line][1]] = ',theta[listOfPairs[line][0],listOfPairs[line][1]])
#    print('theta[listOfPairs[line][2],listOfPairs[line][3]] = ',theta[listOfPairs[line][2],listOfPairs[line][3]])
#    print('[listOfPairs[line][0],listOfPairs[line][1]] = [',listOfPairs[line][0],'],[',listOfPairs[line][1],']')
#    print('[listOfPairs[line][2],listOfPairs[line][3]] = [',listOfPairs[line][2],'],[',listOfPairs[line][3],']')
#    print('gamma_j',gamma_j)
    
#    pwf_pos=1-m.cos(gamma_i + gamma_j)
#    pwf_neg=1-m.cos(gamma_i - gamma_j)
    
    pwf=(1-m.cos(gamma_i + gamma_j)) * (1-m.cos(gamma_i - gamma_j))
    
    
    if(isnan(pwf)):
        pwf=0
    
    return pwf
    
    
def computeAngleBetweenLines(x1,y1,x2,y2):
    """ Computes angle between line (x1,y1)->(x2,y2) and horizontal"""
    
    # If line is not horizontal
    if(x1!=x2):
        angle=(y2-y1)/(x2-x1)
    # If line is horizontal
    else:
        angle=0
        
#    print('y2 = ',y2)
#    print('y1 = ',y1)
#    print('x2 = ',x2)
#    print('x1 = ',x1)
    # beware modulo pi (in this case coordinates are all positive)
    
    return angle
    
def computeSymetryMap(symetricPoints, G, theta):
    """ Computes element of symetry map """
    symetryMapElement=0
    for iLine in range(int(len(symetricPoints))): #303
#        myGwf=computeGWF(symetricPoints, G,iLine)
#        myPwf=computePWF(symetricPoints, theta, iLine)
#        symetryMapElement+=myPwf*myGwf
         symetryMapElement+=computeGWF(symetricPoints, G,iLine)*computePWF(symetricPoints, theta, iLine)
        #print('S = ',S)
    
    return symetryMapElement


#def convolveSymetryMap(symetryMap, sigma):
#    """ Convolve image by gaussian filter"""
#        
#    gauss=sf.gaussian_filter(symetryMap, sigma)
#    
#    return gauss 
#
#
#def detectAboveThreshold(data, threshold):
#    """ """
#    detection = []
#    indexes=[]
#    
#    for i in range(data.shape[0]):
#        if (data[i]>threshold):
#            detection.append(data[i])
#            indexes.append(i)
#        else:
#            detection.append(0)
#        
#    return detection, indexes
#
#
#def exportDetectedPoint(indexes, listOfPairs, data, filepath):
#    """ Export list of point data in text file like following :
#        index col line
#        .
#        .
#        .
#    """
#    
#    # Opens file
#    fichier = open(filepath, "w")
#    fichier.write('index column line\n')
#    
#    # For each sheet of xslx file
#    for index in indexes:
#        fichier.write(str(index))  
#        fichier.write(" ") 
#        fichier.write(str(listOfPairs[index][1]))
#        fichier.write(" ") 
#        fichier.write(str(listOfPairs[index][0]))
#        fichier.write("\n") 
#        
#    fichier.close()
#    
#    
#    return 0

# =============================================================================


if __name__=='__main__':
    
    img=readImage('donnees_pour_tp/aerien_gris/im1.tif')
    
    # 1) Compute Ix and Iy
    Ix,Iy=applySobelFilter(img)
    
    # 2) Compute Gi and thetai
    Gi=computeGi(Ix,Iy)
    thetai=computeThetai(Ix,Iy)
    
    #print('\nthetai 158 334 :',thetai[158,334])
    #print('theta shape : ', thetai.shape)
    
    r=int(img.shape[0]/50)
    print('\nRayon r=',r)
    S=np.zeros((img.shape[0],img.shape[1]))
    j=0
    
    
    for iLine in range(r+1,img.shape[0]-r-1):
        print('iteration ',j)
        for iCol in range(r+1,img.shape[1]-r-1):
            
            newList=Gamma(img,iLine,iCol,r,Gi)
            #print('\nmyList :\n',newList)

            
            S[iLine,iCol]=computeSymetryMap(newList, Gi, thetai)
            #print(' S = ',S[iLine,iCol])
            j+=1
            

    print('\n S = ',S)
    print('\n S[145,145] = ',S[145,145])
    
    # 5) Convolve Symetry Map
#    R=convolveSymetryMap(S, 20)
#    print('\n R = ',R)
#    
#    plt.figure()
#    plt.plot(R,'or', label='R')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    
#    
#    # 6) Detect maxima above threshold
#    test, indexes=detectAboveThreshold(R, 7)
#    
#    plt.plot(test,'xg', label='maxima')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    plt.show()
#    
#    print('\nmyList size : ',len(myList))
#    print('\nS size : ',len(S))
#    print('\nR size : ',len(R))
    

#    yoyo=[2,5,7,6,9,2,56]
#    indexes=[2,4,6]
#    listOfPairs=[[0,0,0,0],[5,5,6,6],[7,7,8,8],[6,6,7,7],[9,9,10,10],[2,2,3,3],[56,56,57,57]]
#    
#    exportDetectedPoint(indexes, listOfPairs, yoyo, 'test.txt')

#    exportDetectedPoint(indexes, myList, R, 'test2.txt')
#    
    
    
    