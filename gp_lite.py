import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from utils import *

class gp_lite:

	def __init__(self, space_input=None, space_output=None, kernel = 'SE'):
		self.kernel = kernel

	def init_hypers(self, case = 'canonical', theta = None):
		self.x = None
		self.y = None
		# for the SE kernel	
		if case == 'random':
			self.sigma = np.random.rand()*2
			self.gamma = np.random.rand()
			self.mu = np.random.rand()*0.1
			self.sigma_n = np.random.rand()*0.1
		elif case == 'canonical':
			self.sigma = 10
			self.gamma = 1/2
			self.mu = 0
			self.sigma_n = 0.1
		elif case == 'manual':
			self.sigma = theta[0]
			self.gamma = theta[1]
			self.mu = theta[2]
			self.sigma_n = theta[3]

	def show_hypers(self):
		print(f'gamma: {self.gamma}, i.e., lengthscale = {np.sqrt(1/(2*self.gamma))}')
		print(f'sigma: {self.sigma}')
		print(f'sigma_n: {self.sigma_n}')
		print(f'mu: {self.mu}')


	def sample(self, how_many=1):
		samples =  np.random.multivariate_normal(self.mean, self.cov, size=how_many)
		self.samples = samples.T
		return self.samples

	def load(self, x, y):
		self.Nx = len(x)
		self.x = x
		self.y = y
	
	def compute_posterior(self, dimension=None, where = None):

		if dimension is None: 
			self.N = 100
			self.time = np.linspace(1,100,100)
		elif np.size(dimension) == 1: 
			self.N = dimension
			self.time = np.linspace(1,100,dimension)
		if where is not None:
			self.time = where
			self.N = len(where)

		cov_grid = Spec_Mix(self.time,self.time, self.gamma, self.mu, self.sigma) + 1e-5*np.eye(self.N) + self.sigma_n**2*np.eye(self.N)

		if self.x is None: #no observations 
			self.mean = np.zeros_like(self.time)
			self.cov = cov_grid
		else: #observations
			cov_obs = Spec_Mix(self.x,self.x,self.gamma,self.mu,self.sigma) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
			cov_star = Spec_Mix(self.time,self.x, self.gamma, self.mu, self.sigma)
			self.mean = np.squeeze(cov_star@np.linalg.solve(cov_obs,self.y))
			self.cov =  cov_grid - (cov_star@np.linalg.solve(cov_obs,cov_star.T))

	def nlogp(self, hypers):
		sigma = np.exp(hypers[0])
		gamma = np.exp(hypers[1])
		mu = np.exp(hypers[2])
		sigma_n = np.exp(hypers[3])
		    
		Y = self.y
		Gram = Spec_Mix(self.x,self.x,gamma,mu,sigma)
		K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
		(sign, logdet) = np.linalg.slogdet(K)
		return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))

	def nll(self):
		Y = self.y
		Gram = Spec_Mix(self.x,self.x,self.gamma,self.mu,self.sigma)
		K = Gram + self.sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
		(sign, logdet) = np.linalg.slogdet(K)
		return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))

	def dnlogp(self, hypers):
		sigma = np.exp(hypers[0])
		gamma = np.exp(hypers[1])
		mu = np.exp(hypers[2])
		sigma_n = np.exp(hypers[3])

		Y = self.y
		Gram = Spec_Mix(self.x,self.x,gamma,mu,sigma)
		K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
		h = np.linalg.solve(K,Y).T

		dKdsigma = 2*Gram/sigma
		dKdgamma = -Gram*(outersum(self.x,-self.x)**2)
		dKdmu = -2*np.pi*Spec_Mix_sine(self.x,self.x, gamma, mu, sigma)*outersum(self.x,-self.x)
		dKdsigma_n = 2*sigma_n*np.eye(self.Nx)

		H = (np.outer(h,h) - np.linalg.inv(K))
		dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
		dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
		dlogp_dmu = mu * 0.5*np.trace(H@dKdmu)
		dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
		return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dmu, -dlogp_dsigma_n])

	def train(self, flag = 'quiet'):
		if self.mu == 0:
			self.mu = 0.1
		hypers0 = np.array([np.log(self.sigma), np.log(self.gamma), np.log(self.mu), np.log(self.sigma_n)])
		res = minimize(self.nlogp, hypers0, args=(), method='L-BFGS-B', jac = self.dnlogp, options={'maxiter': 500, 'disp': True})
		self.sigma = np.exp(res.x[0])
		self.gamma = np.exp(res.x[1])
		self.mu = np.exp(res.x[2])
		self.sigma_n = np.exp(res.x[3])
		self.theta = np.array([self.mu, self.gamma, self.sigma_n ])
		if flag != 'quiet':
		    print('Hyperparameters are:')
		    print(f'sigma ={self.sigma}')
		    print(f'gamma ={self.gamma}')
		    print(f'mu ={self.mu}')
		    print(f'sigma_n ={self.sigma_n}')

	def plot_samples(self, linestyle='-', v_axis_lims = None):
		if v_axis_lims == None:
			v_axis_lims = np.max(np.abs(self.samples))
		plt.figure(figsize=(9,4))
		error_bars = 2 * self.sigma
		plt.fill_between(self.time, - error_bars, error_bars, color='blue',alpha=0.1, label='95% error bars')
		plt.plot(self.time,np.zeros_like(self.time),alpha = 0.7,label='mean')
		if self.samples.shape[1] ==1:
			plt.plot(self.time,self.samples,linestyle, c='r', alpha = 1)
		else:
			plt.plot(self.time,self.samples,linestyle, alpha = 1)
		plt.title('GP samples')
		plt.xlabel('time')
		plt.legend(loc=1)
		plt.xlim([min(self.time),max(self.time)])
		plt.ylim([-v_axis_lims,v_axis_lims])
		plt.tight_layout()

	def plot_posterior(self, n_samples = 0, v_axis_lims = None):
		if v_axis_lims == None:
			v_axis_lims = np.max(np.abs(self.samples))

		plt.figure(figsize=(9,3))
		plt.plot(self.time,self.mean, 'b', label='posterior')

		plt.plot(self.x,self.y, '.r', markersize = 8, label='data')
		error_bars = 2 * np.sqrt((np.diag(self.cov)))
		plt.fill_between(self.time, self.mean - error_bars, self.mean + error_bars, color='blue',alpha=0.1, label='95% error bars')
		if n_samples > 0:
			self.compute_posterior(where = self.time)
			self.sample(how_many = n_samples)
			plt.plot(self.time,self.samples,alpha = 0.7)
		plt.title('Posterior')
		plt.xlabel('time')
		plt.legend(loc=1, ncol=3)
		plt.xlim([min(self.time),max(self.time)])
		plt.ylim([-v_axis_lims,v_axis_lims])
		plt.tight_layout()

	def plot_data(self, v_axis_lims = None):
		if v_axis_lims == None:
			v_axis_lims = np.max(np.abs(self.samples))

		plt.figure(figsize=(9,3))

		plt.plot(self.x,self.y, '.r', markersize = 8,label='data')

		plt.title('Posterior')
		plt.xlabel('time')
		plt.legend(loc=1)
		plt.xlim([min(self.time),max(self.time)])
		plt.ylim([-v_axis_lims,v_axis_lims])
		plt.tight_layout()