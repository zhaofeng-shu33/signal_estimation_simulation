#!usr/bin/python
# -*- coding:utf-8 -*-
# author: zhaofeng-shu33
# license: Apache License Version 2.0
# file description: simulation of signal estimation(continuous)
from scipy import signal
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pdb
import logging
from tabulate import tabulate
class model:
    def __init__(self, time_period, power_spectrum, amplitude = None, frequency = None, phase = None):
        # frequency is circular frequency
        self.time_period = time_period
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.power_spectrum = power_spectrum
    def generate_signal(self, model_base_list,time_interval_len): 
        # generate signal with noise added
        # ugly code, the third parameter is time_interval_len
        return self.amplitude * model_base_list + \
        np.random.normal(scale = np.sqrt(self.power_spectrum/time_interval_len),size = max(model_base_list.shape))
    def sine_signal_base(self,sampling_time_list):
        return np.sin(self.frequency * sampling_time_list + self.phase)
    def sawtooth_signal_base(self, sampling_time_list):
        return signal.sawtooth(self.frequency * sampling_time_list + self.phase)
    def square_signal_base(self, sampling_time_list):
        return signal.square(self.frequency * sampling_time_list + self.phase)

def amplitude_estimate(time_interval_len, signal_base_list, observation_list):
    '''
    use trapezoid method to integrate
    --------
    Returns:
      estimated amplitude
    '''
    numerator = time_interval_len * np.sum(signal_base_list * observation_list)
    denominator = signal_base_energy(time_interval_len, signal_base_list)
    return numerator/denominator
def signal_base_energy(time_interval_len, signal_base_list):
    return time_interval_len * np.sum(signal_base_list * signal_base_list)
def plot_signal(signal_name, t, f_t):
    '''
      plot signal with name from ['square', 'sine', 'sawtooth']
      ------
      Parameters:
        t: time series in x axis
        f_t: signal value in y axis
    '''
    if(type(signal_name) == str):
        plt.plot(t,f_t)
        plt.title(signal_name)
    else:
        for index, name in enumerate(signal_name):
            plt.subplot(len(signal_name),1,index+1)
            plt.title(name)                        
            plt.plot(t,f_t[index,:])
    plt.tight_layout()
    plt.show()

def power_spectrum_range(string):
    power_spectrum_list = eval(string)
    if(type(power_spectrum_list) == float):
        power_spectrum_list = [power_spectrum_list]
    return power_spectrum_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action = 'store_true', default = False)
    parser.add_argument('--sample_num', type = int, default = 100)
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--power_spectrum', type= power_spectrum_range, default = [0.05])
    parser.add_argument('--amp_estimate', help = 'Task: estimate amplitude for three kinds of signals',\
     action = 'store_true', default = True)   
    parser.add_argument('--exp_times', help = 'experiment times for estimation', type = int, default = 100 ) 
    parser.add_argument('--output_table',action = 'store_true', default = False)
    args = parser.parse_args()
    
    time_period = 2*np.pi
    power_spectrum_list = args.power_spectrum
    freq = 2.0
    amp = 1.5
    pha = 0.3
    if(args.debug):
        pdb.set_trace()
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)
    continuous_model = model(time_period, power_spectrum_list[0], amplitude = amp, frequency = freq, phase = pha)
    signal_name = ['sine', 'sawtooth', 'square']
    t = np.linspace(0,time_period, num = args.sample_num)
    f_t = np.zeros([3,args.sample_num])    
    for index, name in enumerate(signal_name):
        exec('f_t[index,:] = continuous_model.%s_signal_base(t)'%name)    
    time_interval_len = time_period/args.sample_num                
    if(args.plot):
        plot_signal(signal_name, t, continuous_model.generate_signal(f_t,time_interval_len))
    # Task amp_estiamte
    if(args.amp_estimate):
        table_header = ['','SNR','bias','var(t)','var(ep)'] # len = 10
        table_contents = []            
        estimated_result = np.zeros([3,args.exp_times])
        var_tmp_thy = np.zeros(3)
        for power_spectrum in power_spectrum_list:
            continuous_model.power_spectrum = power_spectrum
            for cnt in range(args.exp_times):
                for index, name in enumerate(signal_name):
                    estimated_result[index,cnt] = amplitude_estimate(time_interval_len, f_t[index,:],\
                    continuous_model.generate_signal(f_t[index,:],time_interval_len))
            
            average_tmp = np.mean(estimated_result, axis = 1)
            var_tmp_emp = np.var(estimated_result, axis = 1)
            
            for index, name in enumerate(signal_name):
                var_tmp_thy[index] = signal_base_energy(time_interval_len, f_t[index,:])
                snr = 10 * np.log10(var_tmp_thy[index]/continuous_model.power_spectrum)
                v_t = continuous_model.power_spectrum/var_tmp_thy[index]                
                table_contents.append([name,snr,average_tmp[index] - continuous_model.amplitude, \
                var_tmp_emp[index], v_t])            
                logging.info('estimating amplitude with snr = %f for signal %s'%(snr,name))                
                logging.info('amplitude estimated result for %s signal is %f'%(name, average_tmp[index]))
                logging.info('empirical variance of estimated amplitude : %f'%(var_tmp_emp[index]))
                logging.info('theorical variance of amplitude estimator : %f'%(v_t))
        if(args.output_table):
            table_str = tabulate(table_contents, headers=table_header, tablefmt='latex_raw', floatfmt='.2E')
            with open('sim_continuous.txt','w') as f:
                f.write(table_str)
                f.close()
            
