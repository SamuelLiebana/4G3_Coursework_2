import numpy as np
import matplotlib.pyplot as plt
import random
from bitstring import BitArray
from scipy.stats import norm
from tqdm import tqdm

""" Implementation of Hopfield Network to calculate the simulated error probability 
    for a network initialised at corrupted memory."""

# CONSTANTS
N = 100 # number of neurons in network
# number of memories for simulated values of p_e
M_sim = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    15, 20, 25, 30, 35, 40,45, 50, 
    55, 60, 65, 70, 75, 80, 85, 90, 
    95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 
    375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 
    725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
]   

# input noises (i.e. probability of flipping a bit 
# of original memory in initial state)
input_noise = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 1]
# placeholder for simulated error probabilities
error_probs = np.zeros([len(input_noise), len(M_sim)])

# SIMULATION AND ERROR PROBABILITY
for noise in tqdm(range(len(input_noise))):
    # counting variable for indexing error_probs
    index = 0
    # loop over number of memories
    for m in M_sim:
        n_errors=0
        index += 1 # increase counting variable
        # for 50 different collections of memories
        for n in range(50):
            # initialise empty memory array
            mem_array = np.zeros([m, N])
            # generate random memories (i.e. bit patterns)
            for i in range(m):
                mem = random.randint(0, 2**N - 1)
                mem_bin = BitArray(uint=mem, length=N)
                mem_array[i] = mem_bin
            # calculate weight matrix
            W = np.matmul(np.transpose(mem_array)-0.5, mem_array-0.5)
            np.fill_diagonal(W, 0)
            # choose 50 initial memory states (with replacement)
            orig_mems = mem_array[np.random.choice(range(m), 50)]
            # average over chosen memory states
            for orig_mem in orig_mems:
                # average over 50 memory state corruptions for each 
                # selected memory
                for i in range(50):
                    # flip the bits according to the input noise
                    mask = np.random.binomial(1, input_noise[noise], N)
                    r = np.logical_xor(orig_mem, mask) 
                    # asynchronous update from Equation 1.1.2
                    r_new = np.matmul(W, r)
                    # implementation of step function applied 
                    # to the input for the first neuron
                    new_val = 0 if r_new[0]<0 else r[0] if r_new[0]==0 else 1
                    n_errors += 1 if new_val != orig_mem[0] else 0
        # calculate probability as proportion of changes
        prop_errors = n_errors/(50*50*len(orig_mems))
        # populate error probability array
        error_probs[noise, index-1] = prop_errors
   
# PLOTTING
# plot of simulated values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(M_sim, error_probs[0], linestyle = '--', label = '$p_{noise} = 0$', color = 'k')
ax.plot(M_sim, error_probs[1], linestyle = '--', label = '$p_{noise} = 0.01$', color = 'b')
ax.plot(M_sim, error_probs[2], linestyle = '--', label = '$p_{noise} = 0.02$', color = 'm')
ax.plot(M_sim, error_probs[3], linestyle = '--', label = '$p_{noise} = 0.05$', color = 'g')
ax.plot(M_sim, error_probs[4], linestyle = '--', label = '$p_{noise} = 0.1$', color = 'c')
ax.plot(M_sim, error_probs[5], linestyle = '--', label = '$p_{noise} = 0.2$', color = 'r')
ax.plot(M_sim, error_probs[6], linestyle = '--', label = '$p_{noise} = 0.5$', color = 'brown')
ax.plot(M_sim, error_probs[7], linestyle = '--', label = '$p_{noise} = 0.8$', color = 'olive')
ax.plot(M_sim, error_probs[8], linestyle = '--', label = '$p_{noise} = 0.9$', color = 'orange')
ax.plot(M_sim, error_probs[9], linestyle = '--',  label = '$p_{noise} = 1.0$', color = 'darkgray')

# settings for x and y labels and ticks and legend
ax.set_xlabel('Number of Memories (M)', size = 15)
ax.set_ylabel('Probability of Error ($p_e$)', size = 15)
ax.tick_params(labelsize=15)
leg1 = ax.legend(prop={'size': 15})
# ensuring that the x-axis is logarithmic
plt.xscale('log')


plt.show()