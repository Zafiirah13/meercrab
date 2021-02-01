[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4049943.svg)](https://doi.org/10.5281/zenodo.4049943)
[![issues](https://img.shields.io/github/issues/Zafiirah13/meercrab)](https://github.com/Zafiirah13/meercrab/issues)
[![forks](https://img.shields.io/github/forks/Zafiirah13/meercrab)](https://github.com/Zafiirah13/meercrab/network/members)
[![stars](https://img.shields.io/github/stars/Zafiirah13/meercrab)](https://github.com/Zafiirah13/meercrab/stargazers)

# MeerCRAB
MeerLICHT Classification of Real And Bogus using deep learning

Author: Zafiirah Hosenie

Email: zafiirah.hosenie@gmail.com or zafiirah.hosenie@postgrad.manchester.ac.uk

Create virtual environment
---
    conda create -n meercrab
    conda install python==3.6
    
Install the required packages
---

Ensure python 3.6 has been set-up first

    git clone https://github.com/Zafiirah13/meercrab.git    
    cd meercrab
    Install requirements: python setup.py install
    Version: python setup.py --version


How to train the model using python script train.py?
---
To specify specific parameters, run:

    python train.py -m NET3 -n NRDS -minP 35 -maxP 65 -t theshold_9 -train True -mp ./meerCRAB_model/
    
- Select the appropriate parameters to be used for training the network.
- model name: -m can take the following parameters: NET1 , NET2, NET3, NET1_32_64,  NET1_64_128,  NET1_128_256
- number of images: -n can take NRDS, NRD, NRS, NR, D, S
- minimum pixel to crop from images: -minP can take integer values from 0 to 40
- Maximum pixel to crop: -maxP can take integer values from 60 to 100
- threshold: -t can take threshold_8, threshold_9, threshold_10
- training: -train can be boolen True or False
- the directory to save the model: -mp is a string "./meerCRAB_model/"

or train with default parameters as:

    python train.py

How to train the model using Jupyter notebook?
---
    open 'MeerCRAB - DEMO.ipynb' notebook in your browser.
- Select the appropriate parameters to be used for training the network.
- If training = True, run all cells, the code will train and test automatically.
- If training = False, only prediction will be done on the test set found in folder './data'

How to perform prediction on new candidate images without training using python script predict.py?
---
Using specific parameters:

    python predict.py -dd./data/dumpformachinelearning_20200114161507.csv -m NET3 -n NRDS -minP 35 -maxP 65 -t theshold_9 -train True -p 0.5 -mp ./meerCRAB_model/

- data path: -dd is a string that indicates the data directory
- model name: -m can take the following parameters: NET1 , NET2, NET3, NET1_32_64,  NET1_64_128,  NET1_128_256
- number of images: -n can take NRDS, NRD, NRS, NR, D, S
- minimum pixel to crop from images: -minP can take integer values from 0 to 40
- Maximum pixel to crop: -maxP can take integer values from 60 to 100
- threshold: -t can take threshold_8, threshold_9, threshold_10
- probability threshold: -p float varies betwwen 0 to 1. The threshold probability to assign a real candidate
- the directory to load the model: -mp is a string "./meerCRAB_model/"

or run with default parameters:

    python predict.py
    
How to perform prediction on new candidate images without training using Jupyter notebook?
---
    open MeerCRAB-prediction-phase.ipynb in a browser and run all cells
- Assuming we have a csv file similar to the data base,the code use the last 4 columns of the csv files to extract the new, ref, diff, scorr images. Please ensure that the csv file or any database is of this order [New, Ref, Diff, Scorr] and found at the end columns.
- Note that saved models have been trained on either NRDS (4 images), NRD (3 images), NR (2 Images), D (1 image) of 30X30pixels, therefore we need to feed the appropriate number images of 30X30. 
- Input to the function code: 'realbogus_prediction' should be of the shape (Nimages, 30, 30, 4), select which model we want to load, for e.g 'NET1_32_64','NET1_64_128','NET1_128_256','NET1','NET2','NET3' and give the ID of the images.
- The function will output the probability that each candidate is a real source with values varying from [0,1]
