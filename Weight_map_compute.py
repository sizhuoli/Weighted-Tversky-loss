#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from scipy.ndimage.measurements import label
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_weight(filename):
    """Compute weight map from a raster image"""
    #load image and transform into numpy array
    annotation_im = Image.open(filename)
    annotation = np.array(annotation_im)
    #change pixel intensities to 0 (background) and 1 (object)
    annotation[annotation<0.5]=0
    annotation[annotation>=0.5]=1
    #assign a unique label to a single object in the image. 
    #b denotes the number of unique objects and a denotes the generated mask.
    a,b = label(annotation)
    print('num trees:',b)
    #matrix for storing distances
    mapp = np.zeros((a.shape[0],a.shape[1],b))
    #iterate through all individual objects
    for i in range(1,b+1):
        aa = a.copy()
        #select only pixels of object i, set other pixels to 0 (background)
        aa[aa!=i]=0
        not_zeros = np.argwhere(aa != 0)
        zeros = np.argwhere(aa == 0)
        #compute the Euclidean distance of every background pixel to every object pixel.  
        dist_matrix = distance_matrix(zeros, not_zeros, p=2)
        #compute the min distance of a background pixel to each object (corresponds to the distance of the pixel to the border of each object)
        dist = np.min(dist_matrix, axis=1)
        nu = 0
        #iterate through all pixels in the image
        for j in range(mapp.shape[0]):
            for k in range(mapp.shape[1]):
                if aa[j,k] == 0:
                    #save distances in the matrix  
                    mapp[j,k,i-1] = dist[nu]
                    nu +=1
    #for each pixel position, sort the distances to all objects
    maps = np.sort(mapp,axis = 2)
    #distance to the closest object and the 2nd closest object
    d1 = maps[:,:,0]
    d2 = maps[:,:,1]
    #sum of distance
    d = d1+d2
    #save data
    name1 = filename.replace('annotation','weight')
    name1 = name1.replace('.png','')
    #save sum of distance
    np.save(name1, d)
    #weighed map with w0 = 10 and sigma = 5
    weii = 10 * np.exp(-(d)**2/50)
    name2 = filename.replace('annotation','weight_map')
    name2 = name2.replace('.png','')
    #save weight map
    np.save(name2, weii)
    return None

def main():
    #generate weight map from a single raster mask and show the overlay of mask with weight map
    create_weight('annotation_0.png')
    #visualize
    im = np.load('weight_map_0.npy')
    img = Image.open('annotation_0.png')
    img = np.array(img)
    img[img<0.5] = 0
    img[img>=0.5] =1
    #overlay
    com = img + im
    plt.figure(figsize = (8,8))
    ax = plt.gca()
    im = ax.imshow(com)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Weight map overlaied with the mask', fontsize = 16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

if __name__ == "__main__":
    main()