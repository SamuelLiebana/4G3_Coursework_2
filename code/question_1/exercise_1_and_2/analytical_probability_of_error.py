import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""Plot of analytically computed error probability as a function 
   of the number of encoded memories for N=100 and N=1000."""

# CONSTANTS
# defining model parameters
N_1 = 100
N_2 = 1000
M = np.linspace(2, 1000, 1000)

# CALCULATION
# calculating error probabilities using the definition in
# Equation 1.2.26
p_e_1 = np.ones(1000) - norm.cdf(np.sqrt((N_1-1)/(2*(M-1))))
p_e_2 = np.ones(1000) - norm.cdf(np.sqrt((N_2-1)/(2*(M-1))))

# PLOTTING
# semilog plots of the error probability as a function of M
plt.semilogx([0]+p_e_1, label = 'N=100', color ='k')
plt.semilogx([0]+p_e_2, label = 'N=1000', color= 'k', linestyle = '--' )
# plot ticks and labels
plt.xlabel('Number of Memories (M)', size = 15)
plt.ylabel('Probability of Error ($p_e$)', size = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# vertical lines showing onset of increase in p_e
plt.axvline(x=5, color = 'r', linestyle = 'dotted')
plt.axvline(x=50, color = 'r', linestyle = 'dotted')

# show legend
plt.legend(prop={'size': 15})
plt.show()
