import scipy as sp
import pylab as plt
from scipy.signal import find_peaks
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np
from matplotlib.patches import Rectangle

class HH():
    """
    Complete Hodgkin-Huxley Model with periodic input current.
    """

    # Na maximum conductance density normalised to membrane capacitance (uS/nF)
    g_Na = 120.0
    # K maximum conductance density normalised to membrane capacitance (uS/nF)
    g_K  =  36.0
    # Leak maximum conductance density normalised to membrane capacitance (uS/nF)
    g_l  =  0.3
    # Na Nernst reversal potential (mV)
    E_Na =  50.0
    # K Nernst reversal potential (mV)
    E_K  = -77.0
    # Leak Nernst reversal potential (mV)
    E_l  = -54.4
    # Integration timestep (ms)
    delta_t = 0.001
    # Timespan to integrate over
    t = sp.arange(0.0, 500.0, delta_t)


    def alpha_m(self, Vm):
        """Rate of ion channel gating as a function of voltage."""
        return 0.1*(Vm+40.0)/(1.0 - sp.exp(-(Vm+40.0) / 10.0))

    def beta_m(self, Vm):
        """Rate of ion channel gating as a function of voltage."""
        return 4.0*sp.exp(-(Vm+65.0) / 18.0)

    def alpha_h(self, Vm):
        """Rate of ion channel gating as a function of voltage."""
        return 0.07*sp.exp(-(Vm+65.0) / 20.0)

    def beta_h(self, Vm):
        """Rate of ion channel gating as a function of voltage."""
        return 1.0/(1.0 + sp.exp(-(Vm+35.0) / 10.0))

    def alpha_n(self, Vm):
        """Rate of ion channel gating as a function of voltage."""
        return 0.01*(Vm+55.0)/(1.0 - sp.exp(-(Vm+55.0) / 10.0))

    def beta_n(self, Vm):
        """Rate of ion channel gating as a function of voltage."""
        return 0.125*sp.exp(-(Vm+65) / 80.0)

    def I_Na(self, Vm, m, h):
        """
        Membrane current (in mA/nF)
        Sodium (Na = element name)

        |  :param Vm:
        |  :param m:
        |  :param h:
        |  :return: sodium current
        """
        return self.g_Na * m**3 * h * (Vm - self.E_Na)

    def I_K(self, Vm, n):
        """
        Membrane current (in mA/nF)
        Potassium (K = element name)

        |  :param Vm:
        |  :param h:
        |  :return: potassium current
        """
        return self.g_K  * n**4 * (Vm - self.E_K)
    #  Leak
    def I_L(self, Vm):
        """
        Membrane current (in mA/nF)
        Leak

        |  :param Vm:
        |  :param h:
        |  :return: leakage current
        """
        return self.g_l * (Vm - self.E_l)

    def I_ext(self, t, i_amp, T, p):
        """
        External Current, a periodic square pulse with

        |  :param t: time
        |  :param i_amp: amplitude
        |  :param T: period
        |  :param p: pulse width
        |  :return: a periodic input current
        """        
        # intitialise variable
        periodic_train = 0

        # generate input pulse train with characteristics 
        # defined by the input parameters
        for i in range(1, int(len(self.t)/(10**3*T)), 1):
            periodic_train += i_amp*(t>i*T) - i_amp*(t>(i*T+p))
        
        return periodic_train
       

    @staticmethod
    def dALLdt(X, t, i_amp, T, p, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        Vm, m, h, n = X

        dVdt = (self.I_ext(t, i_amp, T, p) - self.I_Na(Vm, m, h) - self.I_K(Vm, n) - self.I_L(Vm)) 
        dmdt = self.alpha_m(Vm)*(1.0-m) - self.beta_m(Vm)*m
        dhdt = self.alpha_h(Vm)*(1.0-h) - self.beta_h(Vm)*h
        dndt = self.alpha_n(Vm)*(1.0-n) - self.beta_n(Vm)*n
        return dVdt, dmdt, dhdt, dndt

    def all_variable_plot(self):
        """
        Main demo for the Hodgkin Huxley neuron model.
        """
        # initialise parameters for input current pulse train
        i_amp = -5
        p = 5
        T = 12

        # integrate the HH equations using euler forward integration
        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(i_amp, T, p, self, ))
        Vm = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        # sodium channel current
        ina = self.I_Na(Vm, m, h)
        # potassium channel current
        ik = self.I_K(Vm, n)
        # leakage current
        il = self.I_L(Vm)

        plt.figure()

        # plot membrane potential
        plt.subplot(4,1,1)
        plt.plot(self.t, Vm, 'k')
        plt.ylabel('$V_{m}$ (mV)')
        plt.ylim(-80, 30)

        # plot channel current
        plt.subplot(4,1,2)
        plt.plot(self.t, ina, 'c', label='$I_{Na}$')
        plt.plot(self.t, ik, 'y', label='$I_{K}$')
        plt.plot(self.t, il, 'm', label='$I_{L}$')
        plt.ylabel('Current (mA/nF)')
        plt.legend()

        # plot state variables
        plt.subplot(4,1,3)
        plt.plot(self.t, m, 'r', label='m')
        plt.plot(self.t, h, 'g', label='h')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Probability')
        plt.legend()

        # plot input current
        plt.subplot(4,1,4)
        i_ext_values = [self.I_ext(t, i_amp, T, p) for t in self.t]
        plt.plot(self.t, i_ext_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{ext}$ (mA/nF)')
        plt.ylim(-10, 1)

        plt.show()

    def membrane_potential_plot(self):
        """
        Only plot the membrane potential.
        """
        # initialise parameters for input current pulse train
        i_amp = 2.3
        p = 5
        T = 14
        # integrate the HH equations using forward euler integration
        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(i_amp, T, p, self,))
        Vm = X[:,0]

        # plot the membrane potential
        plt.plot(self.t, Vm, 'k', label = '$I_{ext} =  14.0$ mA/nF')

        # label, tick and axes settings
        plt.ylabel('$V_{m}$ (mV)', size = 15)
        plt.xlabel('t (ms)', size = 15)
        plt.ylim(-80, 50)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.legend(loc= 'upper right', prop={'size': 15})
        plt.show()

    def parameter_search(self):
        """
        Customisable method to perform parameter searches over T, p and i_amp.
        """

        # input current parameters (can be made lists by changing for loop appropriately)
        T = 19
        p = 5
        i_amp_list = [2.3, 5, 10]

        for i_amp in i_amp_list:
            # integrate HH equations
            X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(i_amp, T, p, self,))
            Vm = X[:, 0]
            # put the external current time series in a list
            i_ext_values = np.array([self.I_ext(t, i_amp, T, p) for t in self.t])

            # axis for membrane potential plot (black)
            fig, ax1 = plt.subplots()
            color = 'tab:black'
            ax1.plot([], [], ' ', label='$T = 19 $ ms, $p = 5$ ms and $I = %1.1f$ mA/nF ' % i_amp)
            ax1.plot(self.t, Vm, 'k')
            ax1.set_ylim(-80, 50)
            ax1.set_ylabel('$V_{m}$ (mV)', size=15)
            ax1.set_xlabel('t (ms)', size=15)

            # axis for overlayed external current plot (red)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.plot(self.t, i_ext_values, 'r')
            ax2.set_ylabel('$I_{ext}$ (mA/nF)', color=color, size = 15)  # we already handled the x-label with ax1
            ax2.set_ylim(-80, 50)
            ax2.tick_params(axis='y', labelcolor=color)

            # settings for ticks, labels and legend
            plt.xticks(size=15)
            plt.yticks(size=15)
            handles,labels = [],[]
            for ax in fig.axes:
                for h,l in zip(*ax.get_legend_handles_labels()):
                    handles.append(h)
                    labels.append(l)
            plt.legend(handles,labels, loc='upper right', prop={'size': 12})

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

            plt.show()

if __name__ == '__main__':
    runner = HH()
    runner.membrane_potential_plot()
