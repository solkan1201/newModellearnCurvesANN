#-*- coding utf-8 -*-

import numpy as np
import math


#######################################################################
# Require:                                                            #
# net[i,j] ← matrix with the weights of each neural synapse,          #
# @ in_j ← input data for learning,                                   #
# @ out_j ← output data for learning,                                 #
# @ n_inputs ← number of input neurons in the first layer,            # 
# @ n_outputs ← number of output neurons in the last layer,           #
# @ dimension_i ← number of rows in the matrix of neural synapses,    #
# @ dimension_j ← number of columns in the matrix of neural synapses  #
#######################################################################

n_inputs = 4
n_outputs = 3


in_ = np.empty((n_inputs))
out_ = np.empty((n_outputs))

dimension_i = n_inputs + n_outputs
dimension_j = n_inputs + n_outputs

net = np.empty((dimension_i, dimension_j))

# criar a matriz de saida
out_noutputs = np.zeros((n_outputs))

# asignar as sinais de saida 
for jj in range(1, n_inputs - 1):
    net[0, jj] = in_[jj]

# calcula as neuronas de saida 
for jj in range(n_inputs + 1, dimension_j - 2):
    net[0, jj] = 0

    for ii in range(1, dimension_i - 2 - n_outputs):
        if ii != jj:
            break
        elif net[ii, jj] != 0:
            net[0, jj] = net[0, jj] + net[ii, jj] * net[ii, 0]

    # calcular f(x)
    net[jj, 0] = 1 / (1 + math.exp(-1 * net[0, jj]))

# return out_noutputs