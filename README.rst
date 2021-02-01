========
meerCRAB
========
MeerLICHT Classification of Real And Bogus using deep learning.
Author: Zafiirah Hosenie
Email: zafiirah.hosenie@gmail.com or zafiirah.hosenie@postgrad.manchester.ac.uk


=======================
How to train the model?
=======================
1. cd to meerCRAB_code folder
2. use 'MeerCRAB - DEMO.ipynb' notebook.
3. Select the appropriate parameters to be used for training the network.
4. if training = True, run all cells, the code will train and test automatically.
5. if training = False, only prediction will be done on the test set in folder '../data'

======================================================================
How to use the trained model on new candidate images without training?
======================================================================
1. cd to meerCRAB_code folder
2. use MeerCRAB-prediction-phase.ipynb
3. Assuming we have a csv file similar to the data base that Bart sent, we need to include the directory
4. The code use the last 4 columns of the csv files to extract the new, ref, diff, scorr images.
5. Note that all save model have been trained on 3 images (30X30pixels), therefore we need to feed 3 images of 30X30. 
6. Input to the function code: 'realbogus_prediction' should be of the shape (Nimages, 30, 30, 3), select which model we want to load, for e.g 'NET1_32_64','NET1_64_128','NET1_128_256','NET2','NET3', give the ID of the images.
7. The function will output the probability that each candidate is a real source, values varying from [0,1]
