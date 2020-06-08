#!/usr/bin/env python
# coding: utf-8
'''
This is an example of anytime POS classification by using Hierarchical Reservoir Task (HRT) model
Reference paper for HRT model:
L. Pedrelli, H. Xavier, "Hierarchical-Task Reservoir for Anytime POS Tagging from Continuous Speech", IJCNN, 2020

----

Luca Pedrelli
luca.pedrelli@inria.fr
lucapedrelli@gmail.com
INRIA Bordeaux Sud-Ouest (France)
Mnemosyne Research Group
https://team.inria.fr/mnemosyne/

----
'''

from HRT import HRT
from reservoirpy import ESN
import numpy as np
import scipy as sp
import scipy.io as spio
import joblib

seed = 42
np.random.seed(seed)

trainIndexes = 4620
testIndexes = 6300

Nu = [39,52,51]
Nr = 1000
Nl = 3
rhos = [0.93681, 1.1492, 0.8405]
lis = [0.18782, 0.12199, 0.7756]
iss = [0.48649, 7.2763, 0.1620]
sparsity = 0.01
regs =  [0.01,0.0001,0.1]

dataset = spio.loadmat('./dataset/inputs.mat', squeeze_me=True)
inputs = [inp.T for inp in dataset['inputs']]

dataset = spio.loadmat('./dataset/targets_ph.mat', squeeze_me=True)
targets = [inp for inp in dataset['targets_PH']]

# one-hot encoding
Target_indexes = np.array(list(map(int, set(np.concatenate(targets)))))-1
I = np.eye(Target_indexes.shape[0])
targets_2 = []
for target in targets:
    targets_2.append(np.transpose(I[np.array(list(map(int, target)))-1]))
targets = targets_2

hrt = HRT(Nu,Nr,Nl, sparsity, rhos, lis, iss, regs)

# Phoneme Prediction Task
inputs = hrt.trainTest(0,inputs,targets, trainIndexes, testIndexes)
inputs = [inp.T for inp in inputs]
dataset = spio.loadmat('./dataset/targets_wd.mat', squeeze_me=True)
targets = [inp for inp in dataset['targets_WD']]

# Word Prediction Task
inputs = hrt.trainTest(1,inputs,targets, trainIndexes, testIndexes)
inputs = [inp.T for inp in inputs]
dataset = spio.loadmat('./dataset/targets_pos.mat', squeeze_me=True)
targets = [inp for inp in dataset['targets']]

# POS Prediction Task
outputs = hrt.trainTest(2,inputs,targets, trainIndexes, testIndexes)
print('Test Error on POS Task: ', np.mean(np.argmax(np.concatenate(outputs[trainIndexes:trainIndexes+testIndexes],axis=1),axis=0) != np.argmax(np.concatenate(targets[trainIndexes:trainIndexes+testIndexes],axis=1),axis=0)))


# In[ ]:




