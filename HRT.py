#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
from reservoirpy import ESN

class HRT(object):
    '''
    Hierarchical Reservoir Task (HRT) class:
    this class implement the HRT model suitable for sequence modeling.
    Reference paper for HRT model:
    L. Pedrelli, H. Xavier, "Hierarchical-Task Reservoir for Anytime POS Tagging from Continuous Speech", IJCNN, 2020
    
    ----
    
    Luca Pedrelli
    luca.pedrelli@inria.fr
    lucapedrelli@gmail.com
    INRIA Bordeaux Sud-Ouest (France)
    Mnemosyne Research group
    https://team.inria.fr/mnemosyne/
    
    ----
    '''
    
    def __init__(self, Nu,Nr,Nl, sparsity, rhos, lis, iss, regs):
        self.layers = []
        
        for layer in range(Nl):

            Win = np.random.rand(Nr,Nu[layer]+1) - 0.5

            W = np.random.rand(Nr,Nr) - 0.5
            mask = np.random.rand(Nr,Nr)
            W[mask > sparsity] = 0

            Win = (Win / np.linalg.norm(Win,2)) * iss[layer]
            original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
            W = W * (rhos[layer] / original_spectral_radius)
            W = sp.sparse.csr_matrix(W)

            reservoir = ESN(lr=lis[layer], W=W, Win=Win, input_bias=True, ridge=regs[layer], Wfb=None, fbfunc=None, typefloat=np.float32) 
            self.layers.append(reservoir)
            
    def trainTest(self, layer,inputs,targets, trainIndexes, testIndexes):
        print('state computation layer '+str(layer)+'...')
        all_states = self.layers[layer].compute_all_states(inputs[0:trainIndexes], workers=1, verbose=0)
        all_states2 = self.layers[layer].compute_all_states(inputs[trainIndexes:trainIndexes+testIndexes], workers=1, verbose=0)
        print('done.')
        
        print('readout training layer '+str(layer)+'...')
        self.layers[layer].Wout = self.layers[layer].fit_readout(all_states, targets[0:trainIndexes], verbose = 0)
        print('done.')

        print('outputs computation layer '+str(layer)+'...')
        output_pred_train = self.layers[layer].compute_outputs(all_states)
        output_pred = self.layers[layer].compute_outputs(all_states2)
        print('done.')
        
        return output_pred_train+output_pred
