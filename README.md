# Hierarchical Reservoir Task (HRT) model

## Installation
Please, follow installation instruction from ReservoirPy library https://github.com/neuronalX/reservoirpy/tree/parallelization

## Quick Example
#### Anytime POS Tagging from Continuous Speech
An example of Anytime POS Tagging from Continuous Speech by using the HRT model.
- HRT_Anytime_POS_Task.py

    ```bash
    python HRT_Anytime_POS_Task.py
    ```
## How to use the HRT class
1. Define the input dimensions (one for each layer), the number of recurrent units, the number of layers, the spectral radii (one for each layer), the input scales (one for each layer), the sparsity and the regularization terms (one for each layer)
    ```python
    Nu = [39,52,51]
    Nr = 1000
    Nl = 3
    rhos = [0.93681, 1.1492, 0.8405]
    lis = [0.18782, 0.12199, 0.7756]
    iss = [0.48649, 7.2763, 0.1620]
    sparsity = 0.01
    regs =  [0.01,0.0001,0.1]
    ```
2. Define the HRT model
    ```python
    hrt = HRT(Nu,Nr,Nl, sparsity, rhos, lis, iss, regs)
    ```
3. Train the layer l and compute the outputs
    ```python
    inputs = hrt.trainTest(l,inputs,targets, trainIndexes, testIndexes)
    ```
