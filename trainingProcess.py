#-*- coding utf-8 -*-

import numpy as np
import math


#######################################################################
# Require:                                                            #
# net(i,j) ← matrix with the weights of each neural synapse,          #
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

# asignar sinais de entrada
for j in range(1, n_inputs):
    net[0, j] = in_[j]

# calcular as neuronas de saidas
for jj in range(n_inputs + 1, dimension_j - 2):
    
    net[0, jj] = 0
    
    for ii in range(1, dimension_i - 2 - n_outputs):
        if ii == j:
            break
        elif net[ii, jj] != 0:
            net[0, jj] = net[0, jj] + net[ii, jj] * net[ii, 0]

    # calcular f(x)
    net[jj, 0] =  1 / (1 + math.exp(-1 * net[0, jj]))

    # calcular derivada de f(x)

    net[jj, dimension_j - 1] = net[jj, 0] * (1 - net[jj, 0])

# calculando delta para as saidas das neuronas
primeiro = dimension_j - 1 - n_outputs

for ii in range(0, n_outputs - 1):
    net[ dimension_i - 1, primeiro + ii] = out_ - net[primeiro + 1, 0] 

# calculando delta para as saidas das neuronas
for jj in range(n_inputs + 1, dimension_j - 2, -1):
    
    for ii in range(1, dimension_j - 2 - n_outputs):
        if ii == jj:
            break
        elif net[ii, jj] != 0:
            net[dimension_i - 1, ii] = net[ dimension_i - 1, ii] + net[ii, jj] * net[dimension_i - 1 , jj]

# Ajuste de pesos 
for jj in range(n_outputs + 1, dimension_j - 2, -1):

    for ii in range(1, dimension_i - 2 - n_outputs):

        if ii != jj:
            break
        elif net[ii, jj] != 0:
            net[ii, jj] = net[ii, jj] + 0.45 * net[dimension_i - 1, jj]
            * net[jj, dimension_j - 1] * net[ii, 0]


# return net
 