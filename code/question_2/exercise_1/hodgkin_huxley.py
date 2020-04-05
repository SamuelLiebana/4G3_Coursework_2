import scipy as sp
import pylab as plt
from scipy.signal import find_peaks
from scipy.integrate import odeint
import numpy as np

class HH():
    """
    Complete Hodgkin-Huxley Model with step input current.
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
    t = sp.arange(0.0, 300.0, delta_t)


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
   
    def I_l(self, Vm):
        """
        Membrane current (in mA/nF)
        Leak

        |  :param Vm:
        |  :param h:
        |  :return: leakage current
        """
        return self.g_l * (Vm - self.E_l)

    def I_ext(self, t, i_amp):
        """
        External Current

        |  :param t: time
        |  :param i_amp: amplitude of input current pulse
        |  :return: a 200ms-long current pulse from 50ms to 250ms
        """
        return  i_amp*(t>50) - i_amp*(t>250)

    @staticmethod
    def dALLdt(X, t, i_amp, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        Vm, m, h, n = X

        dVdt = (self.I_ext(t, i_amp) - self.I_Na(Vm, m, h) - self.I_K(Vm, n) - self.I_l(Vm)) 
        dmdt = self.alpha_m(Vm)*(1.0-m) - self.beta_m(Vm)*m
        dhdt = self.alpha_h(Vm)*(1.0-h) - self.beta_h(Vm)*h
        dndt = self.alpha_n(Vm)*(1.0-n) - self.beta_n(Vm)*n
        return dVdt, dmdt, dhdt, dndt

    def all_variable_plot(self):
        """
        Main demo for the Hodgkin Huxley neuron model.
        """
        # model initialised at the steady-state values for the 
        # state variables at resting potential
        i_amp = 10
        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(i_amp, self, ))
        Vm = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        # sodium channel current
        ina = self.I_Na(Vm, m, h)
        # potassium channel current
        ik = self.I_K(Vm, n)
        # leakage current
        il = self.I_l(Vm)

        plt.figure()

        # plot membrane potential
        plt.subplot(4,1,1)
        plt.plot(self.t, Vm, 'k')
        plt.ylabel('Vm (mV)')

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
        i_ext_values = [self.I_ext(t, i_amp) for t in self.t]
        plt.plot(self.t, i_ext_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{ext}$ (mA/nF)')
        plt.ylim(-1, 10)

        plt.show()

    def membrane_potential_plot(self):
        """
        Only plot the membrane potential with lines indicating
        beginnning and end of current pulse.
        """
        # amplitude of input current pulse
        i_amp = 5
        # integrate the HH equations using forward euler integration
        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(i_amp, self,))
        Vm = X[:,0]
        # plot the membrane potential
        plt.plot(self.t, Vm, 'k', label = '$I_{ext} = %1.1f $ mA/nF' % i_amp )
        # vertical lines at beginning and end of current pulse
        plt.axvline(x=50, linestyle = '--', color = 'g')
        plt.axvline(x=250, linestyle = '--', color = 'r')
        # label, tick and axes settings
        plt.ylabel('$V_{m}$ (mV)', size = 15)
        plt.xlabel('t (ms)', size = 15)
        plt.ylim(-80, 50)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.legend(loc= 'upper right', prop={'size': 15})
        plt.show()

    def external_current_plot(self):
        """
        Plot of different values of the amplitude of the external current
        in our experiments.
        """
        # reference amplitude of the current 
        i_amp = 1
        i_ext_values = np.array([self.I_ext(t, i_amp) for t in self.t])

        # graded colour plot
        for i in np.linspace(0.1,5, 9):
            plt.plot(self.t, i*i_ext_values, color = (1-50*i/256, 1-50*i/256, 1-50*i/256))

        # axes settings
        plt.xlabel('t (ms)', size = 15)
        plt.ylabel('$I_{ext}$ (mA/nF)', size = 15)
        plt.ylim(-1, 10)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
    
        plt.show()

    def firing_rate_vs_input_amplitude(self):
        """
        Plot of the firing rate vs. Input current pulse amplitude.
        """
        
        # input amplitudes for plot
        input_amps = np.linspace(0, 50, 101)
        # placeholder for corresponding rates
        rates = []
        # iterate over input amplitudes
        for i in input_amps:
            # integrate the HH equations
            X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(i, self,))
            Vm = X[:,0]
            # find the peaks in the resuling membrane potential 
            # which are greater than 0
            locs, _ = find_peaks(Vm, 0)
            # if there are at least 4 action potentials
            # find the firing rate by calculating the time
            # that it took for 4 to fire.
            if len(locs) > 4:
                rates.append(3/((locs[4] -  locs[2])*0.001*10**(-3)))
            else:
                rates.append(0)
        
        # plotting a vertical line at the threshold
        # input current amplitude
        rates = np.array(rates)
        pos = np.where(np.abs(np.diff(rates)) >= 70)[0]+1
        thresh = 0.5*(input_amps[pos] + input_amps[pos-1])
        plt.vlines(x= input_amps[pos], ymin = 0, ymax = rates[pos], linestyle = '--', color = 'k')
        
        # at the discontinuity set value to nan so that 
        # we can plot without a connecting line
        input_amps = np.insert(input_amps, pos, np.nan)
        rates = np.insert(rates, pos, np.nan)

        # plot the main curve
        plt.plot(input_amps, rates, color = 'k')
        
        # x and y ticks and labels
        plt.xlabel('$I_{ext}$ (mA/nF)', size = 15)
        plt.ylabel('Firing Rate (Hz)', size = 15)
        plt.xticks(np.append(np.linspace(0,50, 6), thresh), size = 12)
        plt.xlim(0, 60)
        plt.yticks(size = 12)
        plt.show()


if __name__ == '__main__':
    runner = HH()
    # choose which method you want to run
    runner.firing_rate_vs_input_amplitude()