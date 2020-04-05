import numpy as np
import matplotlib.pyplot as plt

"""Plot of analytically computed SNR as a function 
   of the number of encoded memories for N=100 and N=1000."""

# CONSTANTS
# defining model parameters
N_1 = 100
N_2 = 1000
M = np.linspace(1, 100000, 100000)

# CALCULATION
# calculating SNR using the definition in
# Equation 1.2.27
SNR_1 = 0.5*(N_1 -1)/(M-1)
SNR_2 = 0.5*(N_2 -1)/(M-1)

# PLOTTING
# semilog plots of the SNR as a function of M
plt.loglog(SNR_1, label = 'N=100', color ='k')
plt.loglog(SNR_2, label = 'N=1000', color= 'k', linestyle = '--' )
# plot ticks and labels
plt.xlabel('Number of Memories (M)', size = 15)
plt.ylabel('SNR (dB)', size = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# vertical lines showing minimum SNR for reliable communication
plt.hlines(y= 10, xmin=0 , xmax= 50, color = 'r', linestyle = 'dotted')
plt.vlines(x=5, ymin=0, ymax=10, color = 'r', linestyle = 'dotted')
plt.vlines(x=50, ymin=0, ymax=10, color = 'r', linestyle = 'dotted')

# show legend
plt.legend(prop={'size': 15})
plt.show()
