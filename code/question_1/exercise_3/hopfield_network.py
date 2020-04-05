import numpy as np
import matplotlib.pyplot as plt
import random
from bitstring import BitArray
from scipy.stats import norm

""" Implementation of Hopfield Network to calculate the simulated error probability 
    for a network initialised at an encoded memory. The result is compared to the 
    analytical result obtained from Equation 1.2.26."""

# CONSTANTS
N = 100 # number of neurons in network
M_an = np.array(range(2,1001)) # number of memories for analytical values of p_e
# number of memories for simulated values of p_e
M_sim = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    15, 20, 25, 30, 35, 40,45, 50, 
    55, 60, 65, 70, 75, 80, 85, 90, 
    95, 100, 150, 200, 250, 300, 350, 
    400, 450, 500, 550, 600, 650, 700, 
    750, 800, 850, 900, 950, 1000
]
# placeholder for simulated error probabilities
error_probs = np.zeros([len(M_sim)])

# SIMULATION AND ERROR PROBABILITY
# counting variable for indexing error_probs
count = 0
# loop over number of memories
for m in M_sim:
    count += 1 # increase counting variable
    # variable to count number of errors
    p_e = 0
    # for 50 different collections of memories
    for n in range(50):
        # initialise empty memory array
        mem_array = np.zeros([m, N])
        # generate random memories (i.e. bit patterns)
        # following properties of balance and uncorrelatedness
        for i in range(m):
            mem = random.randint(0, 2**N - 1)
            mem_bin = BitArray(uint=mem, length=N)
            mem_array[i] = mem_bin
        # calculate weight matrix
        W = np.matmul(np.transpose(mem_array)-0.5, mem_array-0.5)
        np.fill_diagonal(W, 0)
        # choose (with replacement) 50 random initial rate vectors
        rand_mems = np.random.choice(range(m), 50)
        for mems in rand_mems:
            r = mem_array[mems]
            # asynchronous update from Equation 1.1.2
            r_new = np.matmul(W, r)
            # implementation of step function applied 
            # to the input for the first neuron
            new_val = 0 if r_new[0]<0 else r[0] if r_new[0]==0 else 1
            # increment error counter if state has changed
            p_e += 1 if new_val != r[0] else 0
    # average the number of errors over trials
    error_probs[count-1] = p_e/(50*len(rand_mems))
   
# PLOTTING
# plot of simulated values 
plt.scatter(M_sim, error_probs, marker= 'x', color = 'k', label='Simulated')
# plot of analytical values
p_e_1 = np.ones(999) - norm.cdf(np.sqrt((N-1)/(2*(M_an-1))))
plt.semilogx([0]+p_e_1, label = 'Analytical', color ='k')
# settings for x and y labels and ticks and legend
plt.xlabel('Number of Memories (M)', size = 15)
plt.ylabel('Probability of Error ($p_e$)', size = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(prop={'size': 15})
# ensuring that the x-azis is logarithmic
plt.xscale('log')
plt.show()