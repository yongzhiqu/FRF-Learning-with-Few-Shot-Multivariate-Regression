import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt
from numpy import genfromtxt

#from Successes import Equations
import time
import pickle
np.set_printoptions(suppress = True)

#Load model output and convert to library

# Load in the results from the testing of the model
string = 'Model_1_Prediction'

f = open(string + '.pkl', 'rb')
data = pickle.load(f)
print(np.shape(data['Prediction']))
#print(np.shape(data['Prediction']), np.shape(data['Ground_Truth']))

#Initialize the lists
prediction_amplitude = []
prediction_phase = []

ground_truth_amplitude = []
ground_truth_phase = []

# Size should match the number of samples, should be either 1160, 2600, or 5200
size = 2600

## Section of code to separate out each portion of the saved results from the
## model prediction
a = []
b = []
c = []
d = []
e = []
f = []
g = []
h = []

for i in range(0,size):
    for j in range(0,100):
        a.append(data['Prediction'][i][j][0])
        b.append(data['Prediction'][i][j][1])
        e.append(data['Ground_Truth'][i][j][0])
        f.append(data['Ground_Truth'][i][j][1])
    
    a = np.array(a)
    a = a.reshape(100,1)
    b = np.array(b)
    b = b.reshape(100,1)
    c.append(a)
    d.append(b)
    a = []
    b =[]
    
    e = np.array(e)
    e = e.reshape(100,1)
    f = np.array(f)
    f = f.reshape(100,1)
    g.append(e)
    h.append(f)
    e = []
    f =[]

c = np.array(c)
d = np.array(d)
c = c.reshape(size,100)
d = d.reshape(size,100)

print(np.shape(c))
print(np.shape(d))

g = np.array(g)
h = np.array(h)
g = g.reshape(size,100)
h = h.reshape(size,100)

Ground_Truth_Real = g
Ground_Truth_Imaginary = h
print(np.shape(Ground_Truth_Real))
print(np.shape(Ground_Truth_Imaginary))


## Save results as .csv files ##
#np.savetxt(string + '_Real.csv', c, delimiter = ",")
#np.savetxt(string + '_Imag.csv', d, delimiter = ",")
#np.savetxt(string + '_Ground_Truth_Real.csv', g, delimiter = ",")
#np.savetxt(string + '_Ground_Truth_Imag.csv', h, delimiter = ",")


## Plot the ground truth real spectrum vs. the predicted real spectrum ##
fig = plt.figure()
plt.plot(Ground_Truth_Real[0,:])
plt.plot(c[0,:])
plt.title('Ground Truth vs. Prediction Real Result')
plt.xlabel('Frequency (hz)')
plt.ylabel('Amplitude')
plt.legend(['Ground Truth','Prediction'])
#plt.show()


## Plot the ground truth imaginary spectrum vs. the predicted imaginary spectrum ##
fig = plt.figure()
plt.plot(Ground_Truth_Imaginary[0,:])
plt.plot(d[0,:])
plt.title('Ground Truth vs. Predicted Imaginary Result')
plt.xlabel('Frequency (hz)')
plt.ylabel('Amplitude')
plt.legend(['Ground Truth','Prediction'])
#plt.show()

## Additionally, results can be converted back to time series via IFFT to view results in the time domain ##
