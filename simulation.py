#!usr/bin/python
# -*- coding:utf-8 -*-
# author: zhaofeng-shu33
# license: Apache License Version 2.0
# file description: simulation of signal estimation(discrete)
import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse
import logging
from tabulate import tabulate
def scatter_plot(data, plot_file_name='', plot_title='', plt_show=True):
    '''
    for visualization purpose, we choose both theta and n to be two dimensional.
    data is m times 2 ndarray
    '''
    plt.scatter(data[:,0],data[:,1])
    if not(plot_title):
        plt.title(plot_title)
    if not(plot_file_name):
        plt.savefig(plot_file_name)
    if(plt_show):
        plt.show()

class model:
    def __init__(self, mu_theta, cov_theta, cov_n, C):
        self.mu_theta = mu_theta
        self.theta_len = mu_theta.shape[0]
        self.noise_len = cov_n.shape[0]
        self.cov_theta = cov_theta
        self.C = C
        self.mu_z = np.dot(C,mu_theta)
        self._update_cov_z(cov_n)
    def _update_cov_z(self,cov_n):
        '''
        we need to adjust the SNR to test the estimator,
        therefore, this function is necessary
        '''
        self.cov_n = cov_n
        self.cov_z = self.C @ self.cov_theta @ self.C.T + self.cov_n
        
class estimator_base(model):
    def __init__(self,observation_generator):
        '''
          extract the parameters from observation_generator and initialize the same model
        '''
        self.observation_generator = observation_generator
        super(estimator_base,self).__init__(observation_generator.mu_theta, observation_generator.cov_theta, 
        observation_generator.cov_n, observation_generator.C)
    def verify(self):
        '''
          unbiasness hat{theta}- theta to 0, variance to theoretical bound 
        '''
        t_tmp = self.estimated_theta  # pointer copy
        expected_mean_diff = np.mean(t_tmp - self.mu_theta, axis = 0)   
        t_tmp = t_tmp - self.observation_generator.random_theta   
        expected_error_cov = t_tmp.T @ t_tmp / t_tmp.shape[0]
        logging.info('verification for method %s'%(self.__class__.__name__))
        e_m_d = np.linalg.norm(expected_mean_diff)
        t_e_e = np.trace(self.estimator_cov)
        e_e_e = np.trace(expected_error_cov)
        logging.info('expected mean difference:  %f'%e_m_d)
        logging.info('theoretical estimated error: %f'%t_e_e)
        logging.info('empirical estimated error: %f'%e_e_e)
        return (e_m_d,t_e_e,e_e_e)

    def estimate(self):
        '''
          overwrite in the subclass
        '''
        self.estimated_theta = None
    def _error_bound(self):
        '''
          overwrite in the subclass
        '''
        self.estimator_cov = None
class estimator_mean_square_bayes(estimator_base):
    def __init__(self, observation_generator):        
        '''
            compute theoretical error bound at initialization
        '''
        super(estimator_mean_square_bayes,self).__init__(observation_generator)
        self._error_bound()
        
    def _error_bound(self):
        self.estimator_cov = self.cov_theta - self.cov_theta @ self.C.T @ np.linalg.inv(self.cov_z) @ self.C @ self.cov_theta
    def estimate(self):
        '''
        estimate theta for each samples
        '''
        estimated_theta = self.mu_theta.reshape([2,1])  + self.cov_theta @ self.C.T @ np.linalg.inv(self.cov_z) @ (self.observation_generator.random_z.T - self.mu_z.reshape([2,1]) )
        self.estimated_theta = estimated_theta.T

class estimator_least_square(estimator_base):
    def __init__(self, observation_generator):        
        '''
            compute theoretical error bound at initialization
        '''
        super(estimator_least_square,self).__init__(observation_generator)
        self._error_bound()
        
    def _error_bound(self):
        tmp = np.linalg.inv(self.C.T @ self.C)
        self.estimator_cov = tmp @ self.C.T @ self.cov_n @ self.C @ tmp

    def estimate(self):
        '''
        estimate theta for each samples
        '''
        tmp = np.linalg.inv(self.C.T @ self.C)        
        estimated_theta = tmp @ self.C.T @ self.observation_generator.random_z.T
        self.estimated_theta = estimated_theta.T

class estimator_maximum_likelyhood(estimator_base):
    def __init__(self, observation_generator):        
        '''
          compute theoretical error bound at initialization
        '''
        super(estimator_maximum_likelyhood,self).__init__(observation_generator)
        self._error_bound()

    def _error_bound(self):
        self.estimator_cov = np.linalg.inv(self.C.T @ np.linalg.inv(self.cov_n) @ self.C)

    def estimate(self):
        '''
          estimate theta for each samples
        '''
        estimated_theta = self.estimator_cov @ self.C.T @ np.linalg.inv(self.cov_n) @ self.observation_generator.random_z.T
        self.estimated_theta = estimated_theta.T
    
class observation_generator(model):
    def __init__(self, mu_theta, cov_theta, cov_n, C):
        super(observation_generator,self).__init__(mu_theta, cov_theta, cov_n, C)
    def generate(self,num_of_samples = 100):
        '''
        we generator random theta and random noise respectively 
        and then use z = C * theta + n to generate z
        '''
        self.random_theta = np.random.multivariate_normal(self.mu_theta, self.cov_theta,num_of_samples)
        random_noise = np.random.multivariate_normal(np.zeros(self.noise_len), self.cov_n, num_of_samples)
        self.random_z = np.dot(C,self.random_theta.T).T+ random_noise
    def observation_scatter_plot(self):
        scatter_plot(self.random_z)

def task(sigma_list, observation_generator_instance, method, num_of_samples, output_table):
    table_header = ['SNR','','bayes','ml','ls'] # len = 10
    table_contents = []    
    for sigma in sigma_list:
        cov_n = sigma * sigma * np.array([1,0,0,1]).reshape(2,2)
        observation_generator_instance._update_cov_z(cov_n)
        observation_generator_instance.generate(num_of_samples)
        if method == 'all':
            methods = ['bayes','ml','ls']
        else:
            methods = [method]
        snr = 10*np.log10(np.trace(observation_generator_instance.cov_theta)/np.trace(cov_n))
        logging.info('estimating theta for snr = %f'%snr)
        table_contents.extend([['%.2E'%snr,'mean error'],['','theoretical variance'],['','experimental variance']])
        for alg in methods:
            if(alg == 'bayes'):
                estimator = estimator_mean_square_bayes(og)
            elif(alg == 'ml'):
                estimator = estimator_maximum_likelyhood(og)
            elif(alg == 'ls'):
                estimator = estimator_least_square(og)
            else:
                raise NotImplementedError("method %s not implemented!"%alg)
            estimator.estimate()            
            e_m_d,t_e_e,e_e_e = estimator.verify()
            table_contents[-3].append(e_m_d)
            table_contents[-2].append(t_e_e)
            table_contents[-1].append(e_e_e)
    if(output_table):
        table_str = tabulate(table_contents, headers=table_header, tablefmt='latex_raw', floatfmt='.2E')
        with open('sim_discrete.txt','w') as f:
            f.write(table_str)
            f.close()
def sigma_range(string):
    sigma_list = eval(string)
    if(type(sigma_list) == float):
        sigma_list = [sigma_list]
    return sigma_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--no_plot', action = 'store_true')
    parser.add_argument('--sample_num', type = int, default = 100)
    parser.add_argument('--method', choices = ['bayes','ml','ls','all'], default = 'bayes')
    parser.add_argument('--sigma_list', type = sigma_range, default = [0.1])
    parser.add_argument('--output_table', help = 'whether to write result to a latex table, to output a table, \
    method = all must be set.', action = 'store_true', default=False)
    args = parser.parse_args()
    if(args.debug):
        pdb.set_trace() 
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)
        
    # parameter initialization for discrete case    
    C = np.array([0.5,0.5,0.7,-0.3]).reshape(2,2)
    cov_n = np.array([1,0,0,1]).reshape(2,2)
    mu_theta = np.array([1,1])
    cov_theta = np.array([0.1, 0.03, 0.03, 0.05]).reshape(2,2)

    og = observation_generator(mu_theta, cov_theta, cov_n, C)

    task(args.sigma_list, og, args.method, args.sample_num, args.output_table)
