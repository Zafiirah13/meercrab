#!usr/bin/env python
"""
Authors : Zafiirah Hosenie
Email : zafiirah.hosenie@gmail.com or zafiirah.hosenie@postgrad.manchester.ac.uk
Affiliation : The University of Manchester, UK.
License : MIT
Status : Under Development
Description :
Python implementation for Classification of Bogus and Real using Deep Learning for MeerLICHT facility.
(paper can be found here : https://arxiv.org/abs/---).
This code is tested in Python 3 version 3.5.3  
"""

import os
import numpy as np
from astropy.io import fits
from os import listdir
from os.path import isfile, join
from scipy import misc
import pandas as pd
from PIL import Image

def _get_data_path():
    path = "../data/"
    return path
def normalisation(X_):
    X_min =X_.min()
    X_max=X_.max()
    image = (X_-X_min)/(X_max-X_min)
    return image

def shuffle_all(L, n, seed=0):
    '''INPUT:
    L: List [X,y] for e.g [X_train, y_train]
    n: len of y for e.g len(y_train)

    OUTPUT:
    L: X, y are shuffled for e.g X_train, y_train
    '''
    np.random.seed(seed)
    perm = np.random.permutation(n)
    for i in range(len(L)):
        L[i] = L[i][perm]

    return L

def load_image(ID,full_data,n_images):
    '''
    This function reads a csv file containing images in BLOBS 
    and stack them in the order of [new,ref,diff,scorr] 

    INPUT
    ID: The column that contains the transientid
    full_data: a csv file with columns - transientid, scorr_img, diff_img, ref_img, new_img, label
    n_images: Integer value either 4, 3, 2. The number of images we want to train the CNN.

    OUTPUT:
    X_img: The numpy array of images stacked as [new,ref,diff] if n_images=3, or 
           [new,ref,diff,scorr] if n_images=4, or [new,ref] if n_images=2    
    ID: An array of transientid
    '''

    sci_img  = []
    temp_img = []
    diff_img = []
    scor_img = []

    for j in range(ID.shape[0]):

        image_dat = full_data.loc[full_data['transientid'] == ID[j]]

        i = 1
        nparray = image_dat.iloc[:,i].values
        vhex    = bytearray.fromhex(nparray[0])
        varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
        #varray  = normalisation(varray)
        scor_img.append(varray)

        i = 2
        nparray = image_dat.iloc[:,i].values
        vhex    = bytearray.fromhex(nparray[0])
        varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
        #varray  = normalisation(varray)
        diff_img.append(varray)

        i = 3
        nparray = image_dat.iloc[:,i].values
        vhex    = bytearray.fromhex(nparray[0])
        varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
        #varray  = normalisation(varray)
        temp_img.append(varray)

        i = 4
        nparray = image_dat.iloc[:,i].values
        vhex    = bytearray.fromhex(nparray[0])
        varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
        #varray  = normalisation(varray)
        sci_img.append(varray)


    sci_img  = np.expand_dims(np.array(sci_img),1)
    temp_img = np.expand_dims(np.array(temp_img),1)
    diff_img = np.expand_dims(np.array(diff_img),1)
    scor_img = np.expand_dims(np.array(scor_img),1)

    # new, reference, difference and significance image
    if n_images == 'NRDS':
        X_img= np.stack((temp_img,sci_img,diff_img,scor_img),axis=-1)
        X_img= X_img.reshape(X_img.shape[0], 100, 100, 4)

    # new, reference, difference and significance image
    elif n_images == 'NRD':
        X_img= np.stack((sci_img,temp_img,diff_img),axis=-1)
        X_img= X_img.reshape(X_img.shape[0], 100, 100, 3)

    # new, reference and significance image
    elif n_images == 'NRS':
        X_img= np.stack((sci_img,temp_img,scor_img),axis=-1)
        X_img= X_img.reshape(X_img.shape[0], 100, 100, 3)

    # new, reference image
    elif n_images == 'NR':
        X_img= np.stack((sci_img,temp_img),axis=-1)
        X_img= X_img.reshape(X_img.shape[0], 100, 100, 2)

    # scignificance image
    elif n_images == 'S':
        X_img= scor_img
        X_img= X_img.reshape(X_img.shape[0], 100, 100, 1)

    elif n_images == 'D':
        X_img= diff_img
        X_img= X_img.reshape(X_img.shape[0], 100, 100, 1)

    return X_img, ID

def _parse_transient_fold(fold,n_images,remove_edge_cases):
    '''
    This function will read the csv file with columns:transientid,scorr_img,diff_img,ref_img_new_img,label.

    INPUT
    fold: The filename of the csv file for e.g 'training_set' or 'test_set'
    n_images: Number of images to consider: 4, or 3, or 2
    remove_edge_cases: True or False - Do we want to remove edge cases.

    OUTPUT
    If remove_edge_cases is applied it will delete those candidate files and output a fresh dataframe
    X: The images after deleting edge cases examples
    Y: The labels after deleting edge cases examples
    ID: The transient id after deleting edge cases examples
    '''
    image_data = pd.read_csv(fold)
   
    #image_data = pd.read_csv('./data/'+fold+'.csv')
    transId    = image_data.iloc[:,0].values
    Y          = image_data['label'].values

    X, ID = load_image(ID=transId,full_data=image_data,n_images=n_images)


    if remove_edge_cases:
        del_list = []
        for i in range(len(X)):
            avgs = []
            avgs.append(np.mean(X[i,0], axis=0))
            avgs.append(np.mean(X[i,0], axis=1))
            avgs.append(np.mean(X[i,1], axis=0))
            avgs.append(np.mean(X[i,1], axis=1))
            
            for a in avgs:
                if next((True for j in range(len(a)) if np.abs(a[j])<1), -1) > -1:
                    del_list.append(i)
                    break

        print("Number of images deleted: ", len(del_list))
        X = np.delete(X, del_list, axis=0)
        Y = np.delete(Y, del_list, axis=0)
        ID = np.delete(ID, del_list, axis=0)
       
    return X, Y, ID
    
def get_transient(minPix, maxPix, n_images, fold, cropped=False, shuffle=False, seed=0,  remove_edge_cases=False):
    '''
    This function will shuffle and cropped the images if TRUE

    INPUT
    minPix: Integer value varies from 0 to <100. for e.g 35
    maxPix: Integer value varies from 1 to 100. for e.g 65
    n_images: The number of images to consider when training the CNN- 4, or 3 or 2
    fold: The filename of the csv file for e.g 'training_set' or 'test_set'
    cropped: Either True or False
    shuffle: Either True or False
    remove_edge_cases: Either True or False

    OUTPUT
    If the above are True, it will output new dataframe X, Y, ID

    '''
    X, Y, ID = _parse_transient_fold(fold,n_images,remove_edge_cases)

    assert len(X) == len(Y)
   
    if shuffle:
        np.random.seed(seed)
        perm = np.random.permutation(len(Y))   
        X  = X[perm,:,:,:]
        Y  = Y[perm]
        ID = ID[perm]

    if cropped:
        X = X[:,minPix:maxPix, minPix:maxPix,:]/255.
    
    X  = X.astype(np.float32)
    Y  = Y.astype(np.int32)
    ID = ID.astype(np.int32)


    return  X, Y, ID    

def get_data(dataset, n_images, minPix, maxPix, cropped=False, shuffle=False, seed=0, **kwargs):
    '''
    Use all the above functions to fetch the data: for example the training and test set
    '''
    
    if dataset == "meerlicht":
        return get_transient(minPix=minPix, maxPix=maxPix, n_images=n_images, cropped=cropped, shuffle=shuffle, seed=seed, **kwargs)
        
    else:
        raise Exception("Unknown dataset: %s" % str(dataset))
