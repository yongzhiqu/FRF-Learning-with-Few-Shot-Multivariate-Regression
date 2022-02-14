# FRF-Learning-with-Few-Shot-Multivariate-Regression

This repository contains the code for the paper "Run-time Cutting Force Estimation Based on Learned Nonlinear Frequency Response Function ", to appear on Journal of Manufacturing Science and Engineering, 2022. Codes are developed based on https://github.com/fewshotreg/Few-Shot-Regression

Model Explanation: 
Model should be executed in the command window in an environment with TensorFlow version 1 installed. All files should be in the same folder when running the code. 

## Data:

Matlab preprocessed spindle spectra data are provided as training and testing data. The data is saved in .pkl format for direct read using the code.

"spindledata_complex_5_3.pkl" contains dataset A, the data for training.

"spindledata_complex_5_5.pkl" contains dataset B, the data for testing under same machine condition.

"spindledata_complex_1_15.pkl" contains dataset C, the data for testing under different machine condition with few shot learning.


## Model consists of four scripts: 
ffn_main_spindle.py,  ffn_model_2D.py,  reg_data_generator.py,  Plot_Results.py

## ffn_main_spindle.py : 
This code initializes the model, model parameters and hyperparameters, and loads in the data. The model’s name for the training loop is specified in this part, this name is what the model is saved under after training occurs and is used to load the model in for few-shot training and testing. 

After the model is initialized, there are three operations within the main script:
Training loop, Few-shot training loop, testing loop

The main script is written to only run one part at a time. The script must be run once to perform the training, then again to perform few-shot training, then again to run the testing. To specify which operation that is to be run set the corresponding flag to True, and the rest to False in the first part of the code.

** Defines which operation within the model to perform **

** Only one should be set to True at a time **

flags.DEFINE_bool('Training', True, '--')

flags.DEFINE_bool('Few_Shot_Train', False, '--')

flags.DEFINE_bool('Testing', False, '--')

## Operation 1: Training Loop
The model is trained on the data from the training portion (Spindle_Train_Generator) from reg_data_generator.py, then saved after the specified number of epochs. 
## Operation 2: Few-Shot Training Loop
Few-Shot training is not mandatory. It is a means of further updating the model on a few samples from the testing data to improve the model performance. Perform Few-Shot training based on needs.

The previously trained model’s name needs to be specified within this loop for it to be loaded in correctly. Previously trained model is trained further on a specified number of samples from the testing dataset, which is loaded from reg_data_generator (Spindle_Test_Generator), then saved. Models that are trained via few-shot are saved with “_few_shot” added to the model’s name.

## Operation 3: Testing
Load in the previously trained or few-shot trained model for testing. Previously trained model’s name will need to be specified within this loop. The testing dataset is loaded in within this operation and the testing is performed with this data. The results are saved in tensor form to a .pkl file which can be extracted and plotted in the ‘Plot_Results.py” script. 

## Results are saved in dictionary that has two parts:

{vib} which is the ground truth, has three parts [real, imaginary, frequency]
{force} which is the prediction, has two parts [real, imaginary]

## Function description:

ffn_model_2D.py:
Defines the model structure for the three main portions:
Feature extractor module:
	Define the depth and width of the fully connected neural network layers
Weights generator module:
	Define the number of attention layers and the width of the attention layers. 
Dot product module:
	How the results from the feature extractor module and weights generator module are used to calculate a prediction and the associated model losses. 

reg_data_generator:
Loads in and creates batches for the training and testing data. One dataset should be used for training and another for few-shot training and testing. 

Spindle_Train_Generator:
Import and load in the training data to the appropriate variables
Create batches with the training data
Data for training comes from here

Spindle_Test_Generator:
Import and load in the testing data to the appropriate variables
Create batches with the testing data
Both the data for testing and few-shot training come from here

Plot_Results.py:
Load in predicted results, saved in .pkl file, from previously tested model. Results are then split data from tensor form into each respective portion (ground truth real, imaginary, and prediction real and imaginary). Results are plotted and can be saved to .csv files. 
