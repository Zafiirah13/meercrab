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

This code will be used as an integration to BLACKBOX
"""

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import model_from_json
__version__='2.0.0'
#from keras.models import model_from_json

#-------------------------------------------------------
#----Load Candidate images and Stacked them for CNN-----
#-------------------------------------------------------


def load_new_candidate(ID,full_data,n_images,minPix,maxPix,cropped=False):
	'''
	INPUT:
	ID: The first column of the csv file is 'transientid' --> data.iloc[:,0]
	full_data : Here we will provide the whole csv file as above, the code will atomatically select the last 4 or 3 for extract the images
	            # Four last columns is the candidate file in the order of (20)New Image, (21)Ref Image, (22)Diff Image, (23)Scorr image

	n_images: The number of images to consider 4, 3, or 2
	min_pix: value range from 0 to 100 (applied when cropped = True)
	max_pix: value range from 0 to 100 (applied when cropped = True)
	cropped: True - cropping is done from the centre. If we want 30X30 pixels image, then min_pix= 35, max_pix=65
	
	RETURN: 
	X : images with shape (1,100,100,3) if one image is consider or
	 	(1,30,30,3) when cropped or (1,30, 30, 4) four image consider
	ID: The ID of the image - Important to  keep track of the images

	'''
	sci_img  = []
	temp_img = []
	diff_img = []
	scor_img = []

	for j in range(ID.shape[0]):

	    image_dat = full_data.loc[full_data.iloc[:,0] == ID[j]]

	    i = 1
	    nparray = image_dat.iloc[:,-i].values
	    vhex    = bytearray.fromhex(nparray[0])
	    varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
	    scor_img.append(varray)

	    i = 2
	    nparray = image_dat.iloc[:,-i].values
	    vhex    = bytearray.fromhex(nparray[0])
	    varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
	    diff_img.append(varray)

	    i = 3
	    nparray = image_dat.iloc[:,-i].values
	    vhex    = bytearray.fromhex(nparray[0])
	    varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
	    temp_img.append(varray)

	    i = 4
	    nparray = image_dat.iloc[:,-i].values
	    vhex    = bytearray.fromhex(nparray[0])
	    varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
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
	    

	if cropped:
	    X_img = X_img[:,minPix:maxPix, minPix:maxPix,:]/255.

	X  = X_img.astype(np.float32)
	ID = ID.astype(np.int32)

	return X, ID

def load_prediction_candidate(ID,full_data,n_images,minPix,maxPix,cropped=False):
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
        scor_img.append(varray)

        i = 2
        nparray = image_dat.iloc[:,i].values
        vhex    = bytearray.fromhex(nparray[0])
        varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
        diff_img.append(varray)

        i = 3
        nparray = image_dat.iloc[:,i].values
        vhex    = bytearray.fromhex(nparray[0])
        varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
        temp_img.append(varray)

        i = 4
        nparray = image_dat.iloc[:,i].values
        vhex    = bytearray.fromhex(nparray[0])
        varray  = np.frombuffer(vhex, dtype='<f4').reshape(100, 100)
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

    if cropped:
    	X_img = X_img[:,minPix:maxPix, minPix:maxPix,:]/255.

    X  = X_img.astype(np.float32)
    ID = ID.astype(np.int32)
    return X, ID

#---------------------------------------
#-------Prediction of a candidate-------
#---------------------------------------

def realbogus_prediction(model_name, X_test, ID, probability_threshold,model_path="./meerCRAB_model/"):
	'''
	The code will load the pre-trained network and it will perform prediction on new candidate file.

	INPUT:
	model_name: 'NET1', 'NET2', 'NET3'
	X_test : Image data should have shape (Nimages,100,100,3), (Nimages,30,30,3), (Nimages,30,30,4). This will vary depending on the criteria one use for min_pix, max_pix and num_images.
	ID: The transient ID extracted from the csv file ID=data.iloc[:,0]

	OUTPUT:
	overall_real_prob: An array of probability that each source is real. Value will range between [0 to 1.0]
	overall_dataframe: A table with column transientid of all sources and its associated probability that it is a real source
	'''
	# load json and create model
	json_file = open(model_path+model_name+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	fit_model  = model_from_json(loaded_model_json)

	# load weights into new model
	fit_model.load_weights(model_path+model_name+".h5")
	print("Loaded model:"+ model_name +" from disk")

	# Overall prediction for the whole sample
	overall_ypred  = np.argmax(fit_model.predict(X_test),axis=1)
	overall_probability = fit_model.predict(X_test)

	# For all the candidate, output the probability that it is a real source
	overall_real_prob = overall_probability[:,1]
	overall_dataframe = pd.DataFrame(ID, columns=['transientid'])
	overall_dataframe['ML_PROB_REAL'] = overall_real_prob
	overall_dataframe['label'] = np.round(overall_real_prob>=probability_threshold)

	# Select prediction only for bogus
	bogus_pred_index = np.where(overall_ypred==0)
	bogus_transientID = ID[bogus_pred_index]
	bogus_probability = overall_probability[bogus_pred_index,0][0]

	# Select prediction only for real
	real_pred_index = np.where(overall_ypred==1)
	real_transientID = ID[real_pred_index]
	real_probability = overall_probability[real_pred_index,1][0]
	real_dataframe = pd.DataFrame(real_transientID,columns=['transientid'])
	real_dataframe['ML_PROB_REAL'] = real_probability

	return overall_real_prob, overall_dataframe
