import numpy as np

#utils 

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

def Spec_Mix(x,y, gamma, mu, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.cos(2*np.pi*mu*outersum(x,-y))

def Spec_Mix_sine(x,y, gamma, mu, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*mu*outersum(x,-y))