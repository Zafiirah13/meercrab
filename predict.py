
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

# # MeerCRAB prediction phase on new candidate files
# This script will be integrated in BlackBOX to make prediction on new candidate files.
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('PS')
import os
import numpy as np
import pandas as pd
from meerCRAB_code.model import compile_model,model_save 
import matplotlib.pylab as plt
from keras.utils import np_utils
from time import gmtime, strftime
from meerCRAB_code.util import makedirs, ensure_dir
from meerCRAB_code.prediction_phase import load_new_candidate, realbogus_prediction

import argparse

__version__='2.0.0'



parser = argparse.ArgumentParser()
parser.add_argument('-dd','--data_file', help='The directory where the csv candidates are located',type=str,default='./data/dumpformachinelearning_20200114161507.csv')
parser.add_argument('-m','--model_cnn_name',help='The network name choose from: NET1 NET2  NET3', type=str,default='NET1')
parser.add_argument('-n','--num_images',help='The images to consider and can take str as either NRDS NRD NR D S', type=str, default='NRD')
parser.add_argument('-minP','--minPix',help='The minimum pixel to be used from the image ', type=int, default=35)
parser.add_argument('-maxP','--maxPix',help='The maximum pixel to be used from the image ', type=int, default=65)
parser.add_argument('-t','--threshold',help='threshold atleast 9 people vetted a source as either real or bogus - threshold_9, can also use threshold_8', type=str, default='threshold_9')
parser.add_argument('-p', '--probability_threshold', help='Detection threshold', default=0.5, type=float)
parser.add_argument('-mp', '--model_path', help='The directory where the model should be saved and can be loaded from',type=str, default='./meerCRAB_model/')

args = parser.parse_args()
data_file, model_cnn_name, num_images,probability_threshold, minPix, maxPix, threshold, model_path = args.data_file, args.model_cnn_name, args.num_images, args.probability_threshold, args.minPix, args.maxPix, args.threshold, args.model_path


#-----------------------------------------------------------------------------------------------------------------------------------#
															# ## Load csv file
#-----------------------------------------------------------------------------------------------------------------------------------#

# 
# Csv file having this format - Each row has 24 columns separated by semi-colons.
# 
# In order they are:
# 
# - transientid: ID of souce in DB
# - username:
# - vettingdate:
# - vetclas: can be either real, bogus, bogus_cosmicray, bogus_subtract, 
# - bogus_spike or bogus_ghost
# - number: number of source in orig. FITS file
# - image: ID of image/FITS file in DB
# - date-obs:
# - filter:
# - object: the MeerLICHT/BlackGEM tile of observation
# - psf-fwhm:
# - s-seeing:
# - s-seestd:
# - x_peak: integer x position (no python index) of peak in Scorr image
# - y_peak: idem y
# - ra_peak: corresponding ra [degrees]
# - dec_peak: corresponding dec [degrees]
# - flux_peak: corresponding calibrated flux [microJy]
# - fluxerr_peak: flux uncertainty [microJy]
# - mag_peak: corresponding calibrated magnitude [AB magn.]
# - magerr_peak: magn. uncertainty [AB magn.]
# - thumbnail_red: 100x100 thumbnail
# - thumbnail_ref:
# - thumbnail_d:
# - thumbnail_scorr:
# 
# Notice that the thumbnails are 2D numpy arrays of 32bit floats, and are 
# written as binary large objects (BLOBs). 
# 


data = pd.read_csv(data_file,sep=';',header=None)
data = data.drop_duplicates(subset=0, keep="first")
data.head()

#-----------------------------------------------------------------------------------------------------------------------------------#
														# # Load the new candidates
# - ID: The first column of the csv file is 'transientid' --> data.iloc[:,0]
# - full_data : Here we will provide the whole csv file as above, the code will atomatically select the last 4 or 3 column to extract the images. Four last columns is the candidate file in the order of (20)New Image, (21)Ref Image, (22)Diff Image, (23)Scorr image
# 
# - n_images: The number of images to consider 4, 3, 2. Note that here we should be careful. If the network that we will select below has been trained on 3 images, therefore we               will need to use n_images=3
# - min_pix: value range from 0 to 100 (applied when cropped = True). Note that here we should be careful. If the network that we will select below has been trained on 30X30 images, therefore we will need to use min_pix=35.
# - max_pix: value range from 0 to 100 (applied when cropped = True). Note that here we should be careful. If the network that we will select below has been trained on 30X30 images, therefore we will need to use max_pix=65.
# - cropped: True - cropping is done from the centre. If we want 30X30 pixels image, then min_pix= 35, max_pix=65
#-----------------------------------------------------------------------------------------------------------------------------------#



test, ID_test = load_new_candidate(ID=data.iloc[:,0].values,full_data=data,n_images=num_images,minPix=minPix,maxPix=maxPix,cropped=True)
print("Total number of training instances: {}".format(str(len(ID_test))))
print("The Shape of the test set is {}".format(test.shape))

#-----------------------------------------------------------------------------------------------------------------------------------#
												# # Prediction on new candidate files
# Here we will load the pre-existing train model using the parameter 
# 
# INPUTS:
# - model_name: model_cnn_name = 'NET3'
# - X_test : should have shape (Nimages,100,100,3), (Nimages,30,30,3), (Nimages,30,30,4). This will vary depending on the criteria one use for min_pix, max_pix and num_images.
# - ID: The transient ID extracted from the csv file ID=data.iloc[:,0]
# 
# OUTPUTS:
# - overall_real_prob: An array of probability that each candidate is real
# - overall_dataframe: A table with column transientid and ML_PROB_REAL
#-----------------------------------------------------------------------------------------------------------------------------------#


overall_real_prob, overall_dataframe = realbogus_prediction(model_name=model_cnn_name+'_'+threshold+'_'+num_images, X_test=test,ID=ID_test, probability_threshold=probability_threshold, model_path=model_path)


# The transient ID for each candidate
print(ID_test)

# The probability that each source is a real source: It varies from 0 to 1
print(overall_real_prob)

# A dataframe that contains the transient ID and its probability that it is a Real source
print(overall_dataframe)




