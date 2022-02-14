"""
Code for generating and loading data
"""

import numpy as np
import os
import random
import tensorflow as tf
import pickle
from scipy import signal
# from utils import point_source
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

## Spindle Data for training ##
class Spindle_Train_Generator(object):
    def __init__(self, num_shot, num_val, train=True, bmaml=False):
        self.num_shot = num_shot
        self.bmaml = bmaml
        self.dim_input = 3
        self.dim_output = 2
        import pickle

        ## All data (Vibration Data) takes the structure: [Number of tasks, Length of task, [Real, Imaginary, frequency]]
        ## All labels (Force Data) take the structure: [Number of tasks, Length of task, [Real, Imaginary]]

        ## Three data files are :        
        ## spindledata_complex_1_15.pkl  # Size: Data [1160,100,3], Labels [1160,100,2]
        ## spindledata_complex_5_3.pkl   # Size: Data [5200,100,3], Labels [5200,100,2]
        ## spindledata_complex_5_5.pkl   # Size: Data [2600,100,3], Labels [2600,100,2]
        
        f = open('spindledata_complex_5_5.pkl', 'rb')
        data = pickle.load(f)
        f.close()
        
        self.data = dict()
        self.data['train_x'] = data['vib'][:, :, :]
        self.data['train_y'] = data['force'][:, :, :]
        self.data['test_x'] = data['vib'][:, :, :]
        self.data['test_y'] = data['force'][:, :, :]
        
        print('load data: train_x', self.data['train_x'].shape, 'test_x', self.data['test_x'].shape, 'train_y',
              self.data['train_y'].shape, 'test_y', self.data['test_y'].shape)

        ## Create batches for model training ##
    def generate_batch(self, indx):
        context_x = self.data['train_x'][indx]
        context_y = self.data['train_y'][indx]
        target_x = self.data['test_x'][indx]
        target_y = self.data['test_y'][indx]
        #if self.bmaml:
        #    leader_x = np.concatenate([context_x, target_x], 1)
        #    leader_y = np.concatenate([context_y, target_y], 1)
        #    return context_x, context_y, leader_x, leader_y, target_x, target_y
        return context_x, context_y, target_x, target_y
    

## Spindle Data for Testing and Few-Shot Training ##
class Spindle_Test_Generator (object):
    def __init__(self, num_shot, num_val, train=True, bmaml=False):
        self.num_shot = num_shot
        self.bmaml = bmaml
        self.dim_input = 3
        self.dim_output = 2
        import pickle

        ## All data (Vibration Data) takes the structure: [Number of tasks, Length of task, [Real, Imaginary, frequency]]
        ## All labels (Force Data) take the structure: [Number of tasks, Length of task, [Real, Imaginary]]

        ## Three data files are :        
        ## spindledata_complex_1_15.pkl  # Size: Data(vibration) [1160,100,3], Labels(force) [1160,100,2]
        ## spindledata_complex_5_3.pkl   # Size: Data(vibration) [5200,100,3], Labels(force) [5200,100,2]
        ## spindledata_complex_5_5.pkl   # Size: Data(vibration) [2600,100,3], Labels(force) [2600,100,2]
        
        f = open('spindledata_complex_5_3.pkl', 'rb')
        data = pickle.load(f)
        f.close()
        
        self.data = dict()
        self.data['train_x'] = data['vib'][:, :, :]
        self.data['train_y'] = data['force'][:, :, :]
        self.data['test_x'] = data['vib'][:, :, :]
        self.data['test_y'] = data['force'][:, :, :]
        
        print('load data: train_x', self.data['train_x'].shape, 'test_x', self.data['test_x'].shape, 'train_y',
              self.data['train_y'].shape, 'test_y', self.data['test_y'].shape)
        
    def generate_batch(self, indx):
        context_x = self.data['train_x'][indx]
        context_y = self.data['train_y'][indx]
        target_x = self.data['test_x'][indx]
        target_y = self.data['test_y'][indx]
        #if self.bmaml:
        #    leader_x = np.concatenate([context_x, target_x], 1)
        #    leader_y = np.concatenate([context_y, target_y], 1)
        #    return context_x, context_y, leader_x, leader_y, target_x, target_y
        return context_x, context_y, target_x, target_y
